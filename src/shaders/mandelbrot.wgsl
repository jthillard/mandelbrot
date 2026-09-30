// Deep-zoom Mandelbrot via perturbation theory with rebasing.
//
// Instead of iterating each pixel's orbit directly (which f32 can't do at deep
// zoom), we iterate the *delta* from a high-precision reference orbit computed
// on the CPU. For a pixel c = c_ref + dc, its orbit y_n = X_n + e_n where:
//
//     e_{n+1} = 2 * X_n * e_n + e_n^2 + dc          (all f32)
//
// The full value y_n = X_n + e_n is used for the escape test. Rebasing
// (Zhuoran's method) keeps the delta small and avoids glitches: whenever the
// true value |y| drops below the delta |e|, or the reference runs out, we reset
// the reference index to 0 and carry the full value as the new delta (valid
// because X_0 = 0).
//
// Past ~1e30 zoom the deltas themselves leave f32's exponent range (smallest
// normal ~1.2e-38), so `DEEP` pipelines start each pixel in a rescaled form,
// e = w * 2^s with an f32 mantissa `w` and an i32 exponent `s`, and hand over
// to the plain f32 loop once the delta is big enough (see `iterate_sample`).

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> ref_orbit: array<vec2<f32>>;
// Only read by `fs_color`'s shadow branch (custom-lights palette); the
// iteration pass (`fs_data`) never touches it.
@group(0) @binding(2) var<uniform> lights: array<Light, 16>;
// Only read by the adaptive-AA refine pass (`fs_refine`): the 1-sample-per-
// pixel data texture written by `fs_data`, which decides where to supersample.
@group(1) @binding(0) var coarse_tex: texture_2d<f32>;
// Per-point binary exponents of the reference orbit (`RefOrbit::exps`): the
// true X[m] is ref_orbit[m] * 2^ref_exp[m]. Non-zero only for points below
// f32's range, which only deep references contain; only `DEEP` pipelines read
// it (see `ref_at`).
@group(0) @binding(3) var<storage, read> ref_exp: array<i32>;
// Bivariate linear approximation table (`fractal::bla`): node i of level l
// maps the delta at reference step 1 + i·2^l to the one 2^l steps later as
// e' = M·e + N·dc, valid while log2|e| < r_log2. M = m·2^m_exp and
// N = n·2^n_exp are real 2x2 matrices, row-major (the Jacobian of the map,
// so the abs/conjugate kinds are covered too). `bla_meta` is
// [min_level, level_count, off_0, ..., off_{level_count}]; only `BLA`
// pipelines read either.
struct Bla {
    m: vec4<f32>,
    n: vec4<f32>,
    m_exp: i32,
    n_exp: i32,
    r_log2: f32,
    _pad: u32,
};
@group(0) @binding(4) var<storage, read> bla_nodes: array<Bla>;
@group(0) @binding(5) var<storage, read> bla_meta: array<u32>;

// Pipeline-overridable specialization constants, set per pipeline from the
// uniforms' `kind` / `is_julia` / `de_coloring` (see `PipelineKey` in
// renderer.rs). Every per-iteration branch on them folds away at pipeline
// creation, so the hot loop only contains the current kind's math instead of
// testing all of them on every step. The matching uniform fields are still
// uploaded (the layout is shared with colorize.wgsl) but this shader must read
// these constants, never `u.kind` / `u.is_julia` / `u.de_coloring`.
override KIND: u32 = 0u;
override IS_JULIA: bool = false;
override DE: bool = false;
// A kind-switch morph is in progress (`u.morph_w > 0`): each step blends in a
// second kind, `u.morph_from`. That one is a runtime value (it only lives for
// the length of the animation), so only MORPH pipelines pay for its branches.
override MORPH: bool = false;
// Deep view (`u.scale_exp != 0`): the per-pixel offset `dc`, the pixel size
// and `u.span` / `u.dc_offset` are all in units of 2^scale_exp, and each pixel
// starts in the rescaled deep phase (see `iterate_sample`).
override DEEP: bool = false;
// Jump over runs of reference steps with the BLA table (every kind but
// Phoenix, no morph; see `bla_lookup`).
override BLA: bool = false;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    // Position within the view, in [-0.5, 0.5] at the visible edges.
    @location(0) centered: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    let ndc = fullscreen_triangle_pos(idx);
    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    // Flip y so +imaginary points up the screen.
    out.centered = vec2<f32>(ndc.x, -ndc.y) * 0.5;
    return out;
}

// Complex conjugate.
fn conj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

// Complex division a / b.
fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let d = dot(b, b);
    return vec2<f32>(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / d;
}

// |c + d| - |c|, evaluated exactly (no catastrophic cancellation even when the
// sum crosses zero). This is what makes the Burning Ship delta correct through
// the sign flips that happen all along the axes, where the ship's detail lives.
fn diffabs(c: f32, d: f32) -> f32 {
    let cd = c + d;
    if c >= 0.0 {
        return select(-(2.0 * c + d), d, cd >= 0.0);
    }
    return select(-d, 2.0 * c + d, cd > 0.0);
}

// Perturbation delta for z -> z^p: (Z+e)^p - Z^p = e * sum_{k=0}^{p-1} (Z+e)^k Z^{p-1-k}.
// The large z^p term is never formed (that would cancel catastrophically), and
// the sum is evaluated Horner-style (s <- s*(Z+e) + Z^j) so it needs neither a
// table of powers (a dynamically indexed local array spills to slow memory on
// most GPUs) nor binomial coefficients. Forming Z+e rounds e away when it's
// tiny, but that only perturbs `s` by a relative f32 epsilon, and the result
// is `e * s`, so the delta keeps full relative precision.
fn multibrot_delta(z: vec2<f32>, e: vec2<f32>, p: u32) -> vec2<f32> {
    return cmul(e, multibrot_sum(z, z + e, p));
}

// The sum in `multibrot_delta`, sum_{k=0}^{p-1} y^k Z^{p-1-k} with y = Z+e.
fn multibrot_sum(z: vec2<f32>, y: vec2<f32>, p: u32) -> vec2<f32> {
    var s = vec2<f32>(1.0, 0.0);
    var zj = vec2<f32>(1.0, 0.0);
    for (var j: u32 = 1u; j < p; j = j + 1u) {
        zj = cmul(zj, z); // Z^j
        s = cmul(s, y) + zj;
    }
    return s;
}

// Maximum number of terms in `complex_multibrot_delta`'s series (matches the
// `cm_coef` uniform array: 8 vec4s = 16 complex coefficients). Truncation, not
// exactness: unlike `multibrot_delta` (a finite sum for an integer power), a
// complex power has no finite expansion, so this converges rather than
// terminates. Fine as long as perturbation's usual invariant (|e| << |z|,
// kept true by rebasing) holds, since each extra term is O(w^k) smaller.
const COMPLEX_MULTIBROT_TERMS: u32 = 16u;

// Complex binomial coefficient C(p, k), k in 1..=16, precomputed on the CPU
// (they depend only on p; see `complex_binomials` in app.rs).
fn cm_coef(k: u32) -> vec2<f32> {
    let v = u.cm_coef[(k - 1u) / 2u];
    return select(v.xy, v.zw, (k & 1u) == 0u);
}

// Perturbation delta for z -> z^p with a complex p: (Z+e)^p - Z^p.
//
// When |e| << |Z| (the common case: it's the whole reason perturbation
// works), forming Z+e directly would round e away in f32, so instead expand
// = Z^p * ((1+w)^p - 1), w = e/Z, as a Taylor series in w: (1+w)^p - 1 =
// sum_{k=1}^N C(p,k) w^k. The series stops as soon as the next w^k is
// negligible against the running sum (below f32 precision) — at deep zoom w
// is tiny, so that's typically after 2-3 terms instead of all 16.
//
// Right after a rebase (or near a reference point close to zero, where w is
// singular), e is *not* small relative to Z — that's normal perturbation
// dynamics, not a deep-zoom edge case — and the series above would diverge.
// But forming Z+e directly is numerically safe exactly there (e isn't many
// orders of magnitude smaller than Z), so fall back to a plain subtraction.
//
// The series is also wrong when Z -> Z+e crosses the principal branch cut
// (negative real axis) that `cpow` and the CPU reference use. There it
// continues Z's branch, but the true map jumps by a factor e^{2πip}. Which
// pixels crossed then depended on the reference's position, so whole disks
// flipped branch while panning. With |w| < 0.5, arg(1+w) is within ±30°, so an
// Im sign flip with Re(Z) < 0 is exactly a cut crossing. The direct form is
// fine there: the true delta across the cut is large, not tiny.
fn complex_multibrot_delta(z: vec2<f32>, e: vec2<f32>, p: vec2<f32>) -> vec2<f32> {
    let y = z + e;
    let crosses_cut = z.x < 0.0 && ((z.y < 0.0) != (y.y < 0.0));
    // |w|^2 = |e|^2 / |Z|^2; inf or nan (Z ~ 0, or both ~ 0) correctly fails
    // the `< 0.25` test below and falls through to the direct branch.
    let w2 = dot(e, e) / dot(z, z);
    if w2 < 0.25 && !crosses_cut {
        let w = cdiv(e, z);
        var wk = w; // w^1
        var acc = vec2<f32>(0.0, 0.0);
        for (var k: u32 = 1u; k <= COMPLEX_MULTIBROT_TERMS; k = k + 1u) {
            acc = acc + cmul(cm_coef(k), wk);
            wk = cmul(wk, w);
            if dot(wk, wk) < 1e-18 * dot(acc, acc) {
                break;
            }
        }
        return cmul(cpow(z, p), acc);
    }
    return cpow(y, p) - cpow(z, p);
}

// One perturbation step of `kind`'s delta: e -> f(Z+e) - f(Z), where `z` is
// the reference orbit value X_m. `step_add` (dc) is added by the caller. Must
// match `FractalKind` on the CPU side.
fn advance_delta_kind(kind: u32, z: vec2<f32>, e: vec2<f32>) -> vec2<f32> {
    if kind == KIND_BURNING_SHIP {
        // (|x| + i|y|)^2 has real part x^2 - y^2 (an ordinary square delta) and
        // imaginary part 2|x y|. The imaginary delta is 2(|x y| - |X Y|); diffabs
        // computes it exactly, even where the product x y changes sign — which the
        // old sign(X)sign(Y) shortcut got wrong whenever the delta was large
        // enough to flip it (all the time at shallow zoom).
        let base = 2.0 * cmul(z, e) + cmul(e, e);
        let dp = z.x * e.y + z.y * e.x + e.x * e.y;
        return vec2<f32>(base.x, 2.0 * diffabs(z.x * z.y, dp));
    } else if kind == KIND_TRICORN {
        let cz = conj(z);
        let ce = conj(e);
        return 2.0 * cmul(cz, ce) + cmul(ce, ce);
    } else if kind == KIND_MULTIBROT {
        return multibrot_delta(z, e, clamp(u.power, 2u, MULTIBROT_MAX_POWER));
    } else if kind == KIND_CELTIC {
        // z^2 delta split: sq.x = delta of Re(z^2), sq.y = delta of Im(z^2).
        // Celtic abs the real output, so |Re(z^2)| delta = diffabs(Re(Z^2), sq.x).
        let sq = 2.0 * cmul(z, e) + cmul(e, e);
        return vec2<f32>(diffabs(z.x * z.x - z.y * z.y, sq.x), sq.y);
    } else if kind == KIND_BUFFALO {
        // Abs both outputs: real |Re(z^2)|, imag -|Im(z^2)| (Im(Z^2) = 2 X Y).
        let sq = 2.0 * cmul(z, e) + cmul(e, e);
        return vec2<f32>(diffabs(z.x * z.x - z.y * z.y, sq.x),
                         -diffabs(2.0 * z.x * z.y, sq.y));
    } else if kind == KIND_PERPENDICULAR {
        // real x^2 - y^2 (ordinary square delta), imag -2 x |y|.
        // d(-2 x |y|) = -2[ X·(|Y+ey|-|Y|) + ex·|Y+ey| ]; diffabs gives |Y+ey|-|Y|.
        let sq = 2.0 * cmul(z, e) + cmul(e, e);
        let da = diffabs(z.y, e.y);        // |Y + ey| - |Y|
        let abs_yf = abs(z.y) + da;        // |Y + ey|
        return vec2<f32>(sq.x, -2.0 * (z.x * da + e.x * abs_yf));
    } else if kind == KIND_LAMBDA {
        // Lambda map: z^{n+1} = λ·z·(1-z). Delta: e = λ·e·(1-2z-e).
        let one_minus_2z_minus_e = vec2<f32>(1.0 - 2.0 * z.x - e.x, -2.0 * z.y - e.y);
        return cmul(u.lambda_l, cmul(e, one_minus_2z_minus_e));
    } else if kind == KIND_COMPLEX_MULTIBROT {
        return complex_multibrot_delta(z, e, u.complex_power);
    }
    return 2.0 * cmul(z, e) + cmul(e, e); // Mandelbrot (and Phoenix square part)
}

// Derivative f'(Z) of the iteration map at the full value Z, used to propagate
// the orbit derivative for distance-estimation shading. Exact for the
// holomorphic kinds (z^2 -> 2Z, z^p -> p Z^{p-1}); for the non-holomorphic
// Burning Ship / Tricorn we use |f'| ~ |2Z|, which keeps the DE magnitude close
// enough to de-speckle filaments.
fn fprime_kind(kind: u32, z: vec2<f32>) -> vec2<f32> {
    if kind == KIND_MULTIBROT {
        let p = clamp(u.power, 2u, MULTIBROT_MAX_POWER);
        var zk = z; // Z^1
        for (var k: u32 = 2u; k < p; k = k + 1u) {
            zk = cmul(zk, z); // -> Z^{p-1}
        }
        return f32(p) * zk;
    } else if kind == KIND_LAMBDA {
        // Lambda: f'(z) = λ·(1-2z).
        return cmul(u.lambda_l, vec2<f32>(1.0 - 2.0 * z.x, -2.0 * z.y));
    } else if kind == KIND_COMPLEX_MULTIBROT {
        // f'(z) = p * z^(p-1).
        return cmul(u.complex_power, cpow(z, u.complex_power - vec2<f32>(1.0, 0.0)));
    }
    return 2.0 * z;
}

// Delta step of the current map. While switching kinds (MORPH), the map is
// blended per iteration, (1 - w)*f_kind + w*f_from; that's linear in the two
// outputs, so its delta is the same blend of both kinds' deltas (the CPU
// reference in reference.rs uses the same blend, so rebasing stays exact).
fn advance_delta(z: vec2<f32>, e: vec2<f32>) -> vec2<f32> {
    let d = advance_delta_kind(KIND, z, e);
    if MORPH {
        return mix(d, advance_delta_kind(u.morph_from, z, e), u.morph_w);
    }
    return d;
}

// Derivative of the current (possibly morphing) map; blended like
// `advance_delta`.
fn fprime(z: vec2<f32>) -> vec2<f32> {
    let d = fprime_kind(KIND, z);
    if MORPH {
        return mix(d, fprime_kind(u.morph_from, z), u.morph_w);
    }
    return d;
}

// ---- Deep (rescaled) phase helpers -------------------------------------
//
// A deep value is an f32 mantissa times 2^exponent. All scaling is by exact
// powers of two, so it never rounds.

const LN2: f32 = 0.6931471805599453;
// Where `ldexp_sat` saturates. Only ever compared against small values
// (see `diffabs_scaled`), and far enough from f32's max that doubling it,
// or squaring a value of the size it's compared with, stays finite.
const LDEXP_SAT: f32 = 1.2676506e30; // 2^100

// x * 2^k for any k. WGSL's `ldexp` is only defined for exponents inside
// f32's range, so go through `frexp`: results below the smallest normal
// flush to 0, results above 2^100 saturate to ±2^100.
fn ldexp_sat(x: f32, k: i32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    // Normal x (the usual case): the same result by adding k to the
    // exponent field, much cheaper than frexp + ldexp in this hot path.
    // frexp's exponent is the biased field - 126; the checks keep the
    // result's field in [1, 226], so the add can't carry into the sign.
    let bits = bitcast<u32>(x);
    let biased = i32((bits >> 23u) & 0xffu);
    if biased != 0 && biased != 255 {
        let ex = biased - 126 + k;
        if ex > 100 {
            return select(-LDEXP_SAT, LDEXP_SAT, x > 0.0);
        }
        if ex < -125 {
            return 0.0;
        }
        return bitcast<f32>(bitcast<i32>(bits) + (k << 23u));
    }
    let f = frexp(x);
    let ex = f.exp + k;
    if ex > 100 {
        return select(-LDEXP_SAT, LDEXP_SAT, x > 0.0);
    }
    if ex < -125 {
        return 0.0;
    }
    return ldexp(f.fract, ex);
}

fn ldexp2_sat(v: vec2<f32>, k: i32) -> vec2<f32> {
    return vec2<f32>(ldexp_sat(v.x, k), ldexp_sat(v.y, k));
}

// A complex number as mantissa * 2^e, with the mantissa's larger component in
// [0.5, 1) (or exactly 0, with e = 0).
struct Fe {
    m: vec2<f32>,
    e: i32,
};

fn fe_make(v: vec2<f32>, e: i32) -> Fe {
    let a = max(abs(v.x), abs(v.y));
    if a == 0.0 {
        return Fe(vec2<f32>(0.0, 0.0), 0);
    }
    let k = frexp(a).exp;
    return Fe(ldexp2_sat(v, -k), e + k);
}

// Reference point X[m] as f32 (points stored normalized flush to 0 here;
// only deep references have any, see `ref_exp`).
fn ref_at(m: u32) -> vec2<f32> {
    let x = ref_orbit[m];
    if DEEP {
        let k = ref_exp[m];
        if k != 0 {
            return ldexp2_sat(x, k);
        }
    }
    return x;
}

// X[m] with its full exponent range (deep phase only).
fn ref_fe(m: u32) -> Fe {
    return fe_make(ref_orbit[m], ref_exp[m]);
}

// Complex log of m * 2^e (m != 0).
fn clog_fe(m: vec2<f32>, e: i32) -> vec2<f32> {
    return vec2<f32>(0.5 * log(dot(m, m)) + f32(e) * LN2, atan2(m.y, m.x));
}

fn cexp(a: vec2<f32>) -> vec2<f32> {
    return exp(a.x) * vec2<f32>(cos(a.y), sin(a.y));
}

// diffabs(c, 2^s * d) / 2^s = diffabs(c / 2^s, d): diffabs is positively
// homogeneous. Saturating c / 2^s is harmless: once it dwarfs |d| the result
// is just ±d.
fn diffabs_scaled(c: f32, d: f32, s: i32) -> f32 {
    return diffabs(ldexp_sat(c, -s), d);
}

// Sum of two `Fe`s, at the larger one's exponent.
fn fe_add(a: Fe, b: Fe) -> Fe {
    if a.m.x == 0.0 && a.m.y == 0.0 {
        return b;
    }
    if b.m.x == 0.0 && b.m.y == 0.0 {
        return a;
    }
    let e = max(a.e, b.e);
    return fe_make(ldexp2_sat(a.m, a.e - e) + ldexp2_sat(b.m, b.e - e), e);
}

// One deep step's delta, (f(X + e) - f(X)) / 2^t: the mantissa `w` and its
// exponent `t`. `t` is the input scale s except next to the critical point,
// where the linear part of the step vanishes and the result is ~e^2, far
// below the input's scale.
struct DeepStep {
    w: vec2<f32>,
    t: i32,
};

// The per-kind formula of a deep step: (f(X + e) - f(X)) / 2^t for the
// delta e = w * 2^s, given X measured in units of 2^u (`x` = X / 2^u) and
// `sc` = 2^se, se = s - u, the delta's scale in those units. The result's
// scale is t = s + (p-1)·u for a degree-p kind, since every kind here but
// Lambda is p-homogeneous in (X, e) jointly (`diffabs_scaled` rescales the
// fold-point comparisons the same way). Usually u = 0 (x = X, sc = 2^s,
// t = s); see `deep_step_kind` for when it isn't. Lambda always gets u = 0.
// Every kind needs an arm here too.
fn advance_delta_scaled_kind(kind: u32, x: vec2<f32>, w: vec2<f32>, sc: f32, se: i32) -> vec2<f32> {
    if kind == KIND_BURNING_SHIP {
        let base = 2.0 * cmul(x, w) + sc * cmul(w, w);
        let dp = x.x * w.y + x.y * w.x + sc * w.x * w.y;
        return vec2<f32>(base.x, 2.0 * diffabs_scaled(x.x * x.y, dp, se));
    } else if kind == KIND_TRICORN {
        let cx = conj(x);
        let cw = conj(w);
        return 2.0 * cmul(cx, cw) + sc * cmul(cw, cw);
    } else if kind == KIND_MULTIBROT {
        return cmul(w, multibrot_sum(x, x + sc * w, clamp(u.power, 2u, MULTIBROT_MAX_POWER)));
    } else if kind == KIND_CELTIC {
        let sq = 2.0 * cmul(x, w) + sc * cmul(w, w);
        return vec2<f32>(diffabs_scaled(x.x * x.x - x.y * x.y, sq.x, se), sq.y);
    } else if kind == KIND_BUFFALO {
        let sq = 2.0 * cmul(x, w) + sc * cmul(w, w);
        return vec2<f32>(diffabs_scaled(x.x * x.x - x.y * x.y, sq.x, se),
                         -diffabs_scaled(2.0 * x.x * x.y, sq.y, se));
    } else if kind == KIND_PERPENDICULAR {
        let sq = 2.0 * cmul(x, w) + sc * cmul(w, w);
        let da = diffabs_scaled(x.y, w.y, se); // (|Y + ey| - |Y|) / 2^s
        let abs_yf = abs(x.y) + sc * da;       // |Y + ey| * 2^(s-t)
        return vec2<f32>(sq.x, -2.0 * (x.x * da + w.x * abs_yf));
    } else if kind == KIND_LAMBDA {
        let t = vec2<f32>(1.0 - 2.0 * x.x - sc * w.x, -2.0 * x.y - sc * w.y);
        return cmul(u.lambda_l, cmul(w, t));
    }
    return 2.0 * cmul(x, w) + sc * cmul(w, w); // Mandelbrot (and Phoenix square part)
}

// Below this exponent a reference point counts as next to the critical point
// 0 (see `deep_step_kind`). Above it, the e^2 terms 2^s * w^2 can only flush
// to 0 when they are below 2^-50 of the linear ones.
const DEEP_X_NEAR_LOG2: i32 = -60;

// One deep step of `kind` for the delta e = w * 2^s from X (`x` as f32, `xf`
// at full range). Usually the input scale is kept (t = s). But when X is
// tiny (next to the critical point, e.g. at a minibrot's period), the linear
// term vanishes and the step's value is ~e^p, which would flush to 0 at
// scale s. There every z^p-like kind is p-homogeneous in (X, e) jointly, so
// both are measured in units of 2^u (u = the larger one's exponent) and the
// result lands at t = s + (p-1)·u (`ue` below).
fn deep_step_kind(kind: u32, x: vec2<f32>, xf: Fe, w: vec2<f32>, sc: f32, s: i32) -> DeepStep {
    if kind == KIND_COMPLEX_MULTIBROT {
        return complex_multibrot_step(xf, w, s);
    }
    let x_zero = xf.m.x == 0.0 && xf.m.y == 0.0;
    // Lambda's critical point is 1/2 and its step has a constant linear
    // term (λ·e), so it never needs this.
    if kind == KIND_LAMBDA || (!x_zero && xf.e >= DEEP_X_NEAR_LOG2) {
        return DeepStep(advance_delta_scaled_kind(kind, x, w, sc, s), s);
    }
    let kw = deep_log2(w, vec2<f32>(0.0, 0.0));
    if kw == DEEP_ZERO {
        return DeepStep(w, s); // e = 0: f(X) - f(X)
    }
    var ue = s + kw;
    if !x_zero {
        ue = max(ue, xf.e);
    }
    var deg = 2;
    if kind == KIND_MULTIBROT {
        deg = i32(clamp(u.power, 2u, MULTIBROT_MAX_POWER));
    }
    let xk = ldexp2_sat(xf.m, xf.e - ue);
    let se = s - ue;
    let dw = advance_delta_scaled_kind(kind, xk, w, ldexp_sat(1.0, se), se);
    return DeepStep(dw, s + (deg - 1) * ue);
}

// Scaled `complex_multibrot_delta` with X as a full-range `Fe`. Same series
// as the f32 version, rewritten as X^(p-1) * w * sum_k C(p,k) r^(k-1)
// (r = e/X) so nothing is formed at the delta's true scale; X^(p-1)'s own
// exponent goes into the result's `t`. When |e/X| >= 0.5, X is itself tiny
// (|X| <= 2|e|), so both terms of the direct form are taken in log space at
// the larger one's scale. Branch-cut crossings leave the deep phase before
// stepping (`deep_cut_crossing`).
fn complex_multibrot_step(xf: Fe, w: vec2<f32>, s: i32) -> DeepStep {
    let p = u.complex_power;
    let wf = fe_make(w, s);
    if wf.m.x == 0.0 && wf.m.y == 0.0 {
        return DeepStep(vec2<f32>(0.0, 0.0), s);
    }
    if xf.m.x == 0.0 && xf.m.y == 0.0 {
        // e^p.
        let l = cmul(p, clog_fe(wf.m, wf.e));
        let k = i32(floor(l.x / LN2));
        return DeepStep(cexp(l - vec2<f32>(f32(k) * LN2, 0.0)), k);
    }
    let r = ldexp2_sat(cdiv(wf.m, xf.m), wf.e - xf.e);
    if dot(r, r) < 0.25 {
        var acc = cm_coef(1u);
        var rk = r; // r^(k-1)
        for (var k: u32 = 2u; k <= COMPLEX_MULTIBROT_TERMS; k = k + 1u) {
            acc = acc + cmul(cm_coef(k), rk);
            rk = cmul(rk, r);
            if dot(rk, rk) < 1e-18 * dot(acc, acc) {
                break;
            }
        }
        // X^(p-1) = cexp(l) = cexp(l - k·ln2) * 2^k.
        let l = cmul(p - vec2<f32>(1.0, 0.0), clog_fe(xf.m, xf.e));
        let k = i32(floor(l.x / LN2));
        let x_pm1 = cexp(l - vec2<f32>(f32(k) * LN2, 0.0));
        return DeepStep(cmul(cmul(w, x_pm1), acc), s + k);
    }
    // (X + e)^p - X^p, both in log space (X + e may be exactly 0).
    let xs = ldexp2_sat(xf.m, xf.e - s);
    let yf = fe_make(xs + w, s);
    let lb = cmul(p, clog_fe(xf.m, xf.e));
    var k = i32(floor(lb.x / LN2));
    var la = vec2<f32>(0.0, 0.0);
    let y_zero = yf.m.x == 0.0 && yf.m.y == 0.0;
    if !y_zero {
        la = cmul(p, clog_fe(yf.m, yf.e));
        k = max(k, i32(floor(la.x / LN2)));
    }
    let kl = vec2<f32>(f32(k) * LN2, 0.0);
    var ya = vec2<f32>(0.0, 0.0);
    if !y_zero {
        ya = cexp(la - kl);
    }
    return DeepStep(ya - cexp(lb - kl), k);
}

// Deep-phase twin of `advance_delta` (same morph blend, at the larger of the
// two kinds' output scales).
fn advance_delta_scaled(x: vec2<f32>, xf: Fe, w: vec2<f32>, sc: f32, s: i32) -> DeepStep {
    let a = deep_step_kind(KIND, x, xf, w, sc, s);
    if MORPH {
        let b = deep_step_kind(u.morph_from, x, xf, w, sc, s);
        let t = max(a.t, b.t);
        return DeepStep(mix(ldexp2_sat(a.w, a.t - t), ldexp2_sat(b.w, b.t - t), u.morph_w), t);
    }
    return a;
}

// Whether this step would take Complex Multibrot's X + e across the branch
// cut (see `complex_multibrot_delta`). The delta then jumps to the size of X,
// so the deep phase ends and the f32 loop takes the step.
fn deep_cut_crossing(xf: Fe, w: vec2<f32>, s: i32) -> bool {
    let cm = KIND == KIND_COMPLEX_MULTIBROT || (MORPH && u.morph_from == KIND_COMPLEX_MULTIBROT);
    if !cm {
        return false;
    }
    let yn = xf.m + ldexp2_sat(w, s - xf.e); // (X + e) / 2^xe
    return xf.m.x < 0.0 && ((xf.m.y < 0.0) != (yn.y < 0.0));
}

// Binary exponent of the largest component of a pair of complex mantissas
// (`frexp` convention: |x| < 2^k), or DEEP_ZERO when both are exactly 0.
const DEEP_ZERO: i32 = -100000;
fn deep_log2(a: vec2<f32>, b: vec2<f32>) -> i32 {
    let m = max(max(abs(a.x), abs(a.y)), max(abs(b.x), abs(b.y)));
    if m == 0.0 {
        return DEEP_ZERO;
    }
    return frexp(m).exp;
}

// f'(y) at the full value y = X + w * 2^s, as an `Fe`. Plain `fprime` in f32
// unless y is below f32's comfortable range, which only happens next to the
// critical point 0 (after a rebase), where X is itself tiny: then y is
// formed in the 2^s-scaled domain and f' ~ p·y^(p-1) keeps its exponent.
fn deep_fprime(xf: Fe, w: vec2<f32>, s: i32, yt: vec2<f32>) -> Fe {
    if MORPH || max(abs(yt.x), abs(yt.y)) >= DEEP_TINY {
        return Fe(fprime(yt), 0);
    }
    let yf = fe_make(ldexp2_sat(xf.m, xf.e - s) + w, s);
    if yf.m.x == 0.0 && yf.m.y == 0.0 {
        return Fe(fprime(vec2<f32>(0.0, 0.0)), 0);
    }
    if KIND == KIND_MULTIBROT {
        let p = clamp(u.power, 2u, MULTIBROT_MAX_POWER);
        var ym = yf.m; // m^(p-1)
        for (var k: u32 = 2u; k < p; k = k + 1u) {
            ym = cmul(ym, yf.m);
        }
        return fe_make(f32(p) * ym, yf.e * i32(p - 1u));
    } else if KIND == KIND_LAMBDA {
        return Fe(fprime(vec2<f32>(0.0, 0.0)), 0); // λ(1 - 2y) ~ λ
    } else if KIND == KIND_COMPLEX_MULTIBROT {
        // p·y^(p-1) = p·exp(l), l = (p-1)·ln y; keep exp(Re l)'s exponent.
        let p = u.complex_power;
        let l = cmul(p - vec2<f32>(1.0, 0.0), clog_fe(yf.m, yf.e));
        let k = i32(floor(l.x / LN2));
        return fe_make(cmul(p, cexp(vec2<f32>(l.x - f32(k) * LN2, l.y))), k);
    }
    // z^2-like kinds: 2y (|f'| = |2y| for the abs variants too, see fprime).
    return Fe(2.0 * yf.m, yf.e);
}

// The deep phase hands over to the f32 loop once the delta's magnitude
// reaches 2^DEEP_EXIT_LOG2: by then |e|^2 is still a normal f32 and the
// pixel offset dc (< 2^-99 on deep views) is below f32 rounding of e. The DE
// derivative only has to be a comfortably normal f32 (DEEP_EXIT_DZ_LOG2).
const DEEP_EXIT_LOG2: i32 = -48;
const DEEP_EXIT_DZ_LOG2: i32 = -100;
// Below this, a full value y goes through `deep_fprime`'s extended path.
const DEEP_TINY: f32 = 7.888609e-31; // 2^-100
// The mantissas are renormalized once their exponent drifts past ±this.
const DEEP_RENORM_LOG2: i32 = 16;
// A reference point can only matter for rebasing when it's within this many
// binades above the delta's scale (|w| < 2^DEEP_RENORM_LOG2).
const DEEP_NEAR_LOG2: i32 = 24;

// Row-major 2x2 matrix times vector (a BLA node's M or N).
fn mat2v(m: vec4<f32>, v: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(dot(m.xy, v), dot(m.zw, v));
}

// log2|v|, from above (|v| <= sqrt(2)·max component). Not via dot(v, v):
// a delta ~2^-100 (a pixel offset at 1e-30 zoom) squares below f32's range,
// and log2(0) = -inf passed every radius test, jumping pixels far too early.
fn log2_mag(v: vec2<f32>) -> f32 {
    return log2(max(abs(v.x), abs(v.y))) + 0.5;
}

// Longest BLA jump available at reference index `m` for a delta of
// log2 magnitude `e_log2`, of at most `budget` steps: (node index, steps),
// steps = 0 when there's none. Nodes start at 1 + i·2^l, so level l needs
// m - 1 aligned to 2^l. A longer run's radius is never larger than its first
// half's, so the search goes up from the shortest and stops at the first
// invalid level (usually right away: one load per failed attempt).
fn bla_lookup(m: u32, e_log2: f32, budget: u32) -> vec2<u32> {
    var best = vec2<u32>(0u, 0u);
    let levels = bla_meta[1];
    if levels == 0u || m == 0u {
        return best;
    }
    let lmin = bla_meta[0];
    let j = m - 1u;
    let tz = countTrailingZeros(j); // 32 when j == 0
    if tz < lmin {
        return best;
    }
    let top = min(tz - lmin, levels - 1u);
    // Level offsets without loading them from `bla_meta` per level (one more
    // dependent load each): level 0 starts at 0 and each level has half the
    // previous one's nodes, rounded down (`bla::build`).
    var base = 0u;
    var count = bla_meta[3];
    for (var k: u32 = 0u; k <= top; k = k + 1u) {
        let l = lmin + k;
        let steps = 1u << l;
        let local = j >> l;
        if steps > budget || local >= count {
            break;
        }
        let idx = base + local;
        if !(e_log2 < bla_nodes[idx].r_log2) {
            break;
        }
        best = vec2<u32>(idx, steps);
        base = base + count;
        count = count >> 1u;
    }
    return best;
}

// Weight of the Phoenix kind's p*z_{n-1} term in the current map: 1 for plain
// Phoenix, its morph share while switching to/from Phoenix, else 0.
fn phoenix_weight() -> f32 {
    var w = 0.0;
    if KIND == KIND_PHOENIX {
        w = select(1.0, 1.0 - u.morph_w, MORPH);
    }
    if MORPH && u.morph_from == KIND_PHOENIX {
        w = w + u.morph_w;
    }
    return w;
}

// Periodicity (interior) detection, Brent-style: windows end at iterations
// PERIOD_FIRST_CHECK, 2x that, 4x ..., and at each window end the window's
// iterate closest to the critical point is saved. Every later iterate is
// compared against the last saved one. Returning within PERIOD_EPS2
// (relative, squared) means the orbit has closed a cycle.
//
// Why the closest-to-critical iterate rather than the one at the window end:
// near a deep minibrot, an orbit follows the minibrot's cycle with a
// deviation at the minibrot's own scale. At an arbitrary phase |z| ~ 1, so
// that deviation is far below the relative tolerance and *any* nearby
// exterior orbit "returned" (a large black disk around a minibrot at
// ~1e-13 zoom). At the phase nearest the critical point, z itself is at the
// minibrot's scale, so the relative tolerance measures the actual return.
//
// A close return alone isn't trusted. A pixel just *outside* the set (at a
// minibrot's edge, or a cusp) can shadow a cycle for thousands of iterations
// before escaping. So three safeguards apply, tuned against an f64 simulation
// of this exact algorithm and f64 ground truth on cusp, bulb-contact,
// minibrot-edge and deep seahorse views:
// * Multiplier: |product of f'(z)|^2 over the steps since the save must be
//   < PERIOD_MAX_MULT2, so the cycle it closed is clearly attracting. Plain
//   "< 1" let near-parabolic exterior points (|multiplier| ~ 1) through at
//   cusps; the margin fixes that.
// * Confirmation: the contracting return must happen in
//   PERIOD_CONFIRMATIONS consecutive windows, each twice as long as the
//   last. Exterior orbits passing near the critical point can look strongly
//   contracting for one window (seen: flagged at iteration 245, escaped at
//   2275). A second, longer window rules that out.
// * Tolerance: PERIOD_EPS2 is relative and near f32 precision.
// Every kind here except Phoenix is
// (piecewise) conformal, so |f'| from `fprime` is the exact local scale
// factor, including the abs-folding kinds, whose folds are isometries.
// Phoenix's two-term map would need a 2x2 Jacobian, so it's excluded (as
// is a kind-switch morph, for the same reason). So is
// Complex Multibrot without DE, where `fprime` would add a second `cpow`
// (log/atan2/exp) per step for a check that rarely fires on its views.
const PERIOD_FIRST_CHECK: u32 = 16u;
const PERIOD_EPS2: f32 = 1e-12;
const PERIOD_MAX_MULT2: f32 = 0.25;
const PERIOD_CONFIRMATIONS: u32 = 2u;

// Whether `iterate_sample` runs periodicity detection for this kind (folds to
// a constant per pipeline).
fn periodic_enabled() -> bool {
    // A blend of two maps isn't conformal, so |f'| isn't its scale factor.
    if MORPH || KIND == KIND_PHOENIX {
        return false;
    }
    if KIND == KIND_COMPLEX_MULTIBROT && !DE {
        return false;
    }
    return true;
}

// Critical point of the current map (where f' = 0), the periodicity save
// point's reference: 0 for every z^p-like kind, 1/2 for Lambda's λz(1-z).
fn critical_point() -> vec2<f32> {
    if KIND == KIND_LAMBDA {
        return vec2<f32>(0.5, 0.0);
    }
    return vec2<f32>(0.0, 0.0);
}

// Escape data for one sample: `ci` is the (color-independent) palette parameter,
// `de` the distance-estimate darkening factor in [0,1], `escaped` false for the
// interior of the set. Splitting iteration from coloring lets a colour change be
// remapped cheaply (see the colourise pass) without re-iterating.
struct Sample {
    ci: f32,
    de: f32,
    escaped: bool,
};

// Perturbation iterate a single sample. `offset` is the per-pixel offset in
// complex units. For Mandelbrot it is the c-plane offset added every step (delta
// starts at 0); for Julia it is the z-plane offset that seeds the initial delta
// (c is fixed, so nothing is added per step). In `DEEP` pipelines both
// `offset` and `px` are in units of 2^u.scale_exp.
fn iterate_sample(offset: vec2<f32>, px: f32) -> Sample {
    // Loop invariants, read once instead of on every iteration.
    let max_iter = u.max_iter;
    let bailout_sq = u.bailout_sq;
    let ref_len = u.ref_len;
    let z0 = ref_at(0u); // reference start (0 for Mandelbrot, center for Julia)

    // Main cardioid / period-2 bulb bypass: those points never escape, so skip
    // iterating them (they'd otherwise all burn the full max_iter). `offset` is
    // relative to the reference center; the absolute c is recovered from the
    // orbit itself, since X_1 = X_0^2 + C_ref = C_ref. That's only f32-accurate,
    // so skip the test once a pixel is smaller than that error (deep zoom),
    // where it could misclassify pixels right at the boundary.
    if KIND == KIND_MANDELBROT && !MORPH && !IS_JULIA && !DEEP && ref_len > 1u && px > 1e-6 {
        let c = ref_orbit[1] + offset;
        let xq = c.x - 0.25;
        let q = xq * xq + c.y * c.y;
        let in_cardioid = q * (q + xq) <= 0.25 * c.y * c.y;
        let xb = c.x + 1.0;
        let in_bulb = xb * xb + c.y * c.y <= 0.0625;
        if in_cardioid || in_bulb {
            return Sample(0.0, 1.0, false); // interior of the set
        }
    }

    // Set plane: delta starts at 0 and gains dc every step. Julia: the offset
    // seeds the delta and nothing is added per step.
    var step_add = offset;
    var e = vec2<f32>(0.0, 0.0);
    // Pixel size in complex units (`px` is pre-scaled in DEEP pipelines).
    var px_t = px;
    // Orbit derivative for distance estimation, pre-multiplied by the pixel
    // size `px`. For the set plane it is px·d/dc (starts at 0, gains +px each
    // step); for Julia it is px·d/dz0 (starts at px). The raw derivative grows
    // like 1/px, so unscaled its square overflows f32 at deep zoom (~1e-12),
    // which zeroed DE along iteration bands; scaled, it stays ~|z|ln|z| / DE
    // in pixels at any depth.
    var dzs = vec2<f32>(0.0, 0.0);
    if IS_JULIA {
        step_add = vec2<f32>(0.0, 0.0);
        e = offset;
        dzs = vec2<f32>(px_t, 0.0);
    }
    // Previous-iterate state for the Phoenix two-term recurrence (delta of
    // y_{n-1}, and its scaled derivative for DE). Both start at 0 (y_{-1} = 0).
    var e_prev = vec2<f32>(0.0, 0.0);
    var dzs_prev = vec2<f32>(0.0, 0.0);
    let phoenix_w = phoenix_weight();

    var m: u32 = 0u;              // reference index; invariant: y_n = xm + e, xm = X[m]
    var n: u32 = 0u;              // total iteration count
    var xm = z0;                  // X[m], carried so each step loads the orbit once
    var z = xm + e;               // full value y_n, kept for coloring
    var z2 = dot(z, z);
    var escaped = false;

    // Periodicity detection (see PERIOD_FIRST_CHECK): last saved orbit value,
    // |f'|^2 product of the steps since it was saved, next save iteration.
    let periodic = periodic_enabled();
    // Plus whether this window already had a contracting return, and how many
    // consecutive windows have.
    var z_saved = z;
    var mult2 = 1.0;
    var check_at = PERIOD_FIRST_CHECK;
    var period_hit = false;
    var period_streak = 0u;
    // This window's save candidate: its iterate closest to the critical
    // point, that distance squared, and the |f'|^2 product since it.
    let crit = critical_point();
    var z_cand = z;
    var cand_d2 = 3.0e38;
    var mult2_cand = 1.0;

    // Deep phase (DEEP pipelines only): iterate the delta as e = w * 2^sx,
    // with `w` an f32 mantissa kept near 1 by renormalizing and `sx` an i32
    // exponent, while |e| is too small for f32 (it starts at the pixel offset,
    // ~2^scale_exp). The DE derivative is linear in the same way and carried
    // as dzs = v * 2^sv, with its own exponent: near the critical point (after
    // a rebase) f' is tiny, so dzs and e can drift far apart. Once both are
    // big enough (or a rebase makes the delta large), the state is converted
    // to plain f32 and the loop below carries on from the same n / m.
    if DEEP {
        let scale_e = u.scale_exp;
        var sx = scale_e;
        var sv = scale_e;
        var sc = ldexp_sat(1.0, sx);
        var w = vec2<f32>(0.0, 0.0);
        var v = vec2<f32>(0.0, 0.0);
        var w_prev = vec2<f32>(0.0, 0.0);
        var v_prev = vec2<f32>(0.0, 0.0);
        // Per-step additions (dc and px for the set plane), in units of 2^sx
        // and 2^sv.
        var d = offset;
        var pd = px;
        if IS_JULIA {
            w = offset;
            v = vec2<f32>(px, 0.0);
            d = vec2<f32>(0.0, 0.0);
            pd = 0.0;
        }
        let z0f = ref_fe(0u);
        var xf = z0f;
        var xt = z0;
        var xf_old = xf;
        var xt_old = xt;
        var w_old = w;
        var rebase_exit = false;
        var cut_exit = false;
        loop {
            // Full value y = X + e as f32 (e flushes to 0 when negligible).
            let yt = xt + ldexp2_sat(w, sx);
            if dot(yt, yt) > bailout_sq {
                escaped = true;
                break;
            }
            if n >= max_iter {
                break;
            }
            if deep_cut_crossing(xf, w, sx) {
                cut_exit = true;
                break;
            }

            // BLA: jump the whole run the table covers, straight to its
            // output scale (usually many binades up, towards the f32 hand-off).
            var jumped = false;
            if BLA {
                let e_log2 = f32(sx) + log2_mag(w);
                let hit = bla_lookup(m, e_log2, min(max_iter - n, ref_len - 1u - m));
                if hit.y != 0u {
                    let nd = bla_nodes[hit.x];
                    if DE {
                        var accv = fe_make(mat2v(nd.m, v), nd.m_exp + sv);
                        if !IS_JULIA {
                            accv = fe_add(accv, fe_make(mat2v(nd.n, vec2<f32>(px, 0.0)), nd.n_exp + scale_e));
                        }
                        v = accv.m;
                        if accv.m.x != 0.0 || accv.m.y != 0.0 {
                            sv = accv.e;
                            if !IS_JULIA {
                                pd = ldexp_sat(px, scale_e - sv);
                            }
                        }
                    }
                    var acc = fe_make(mat2v(nd.m, w), nd.m_exp + sx);
                    if !IS_JULIA {
                        acc = fe_add(acc, fe_make(mat2v(nd.n, offset), nd.n_exp + scale_e));
                    }
                    w = acc.m;
                    if acc.m.x != 0.0 || acc.m.y != 0.0 {
                        sx = acc.e;
                        sc = ldexp_sat(1.0, sx);
                        if !IS_JULIA {
                            d = ldexp2_sat(offset, scale_e - sx);
                        }
                    }
                    w_old = w;
                    m = m + hit.y;
                    n = n + hit.y;
                    jumped = true;
                }
            }

            if !jumped {
            if DE {
                let fp = deep_fprime(xf, w, sx, yt);
                if fp.e == 0 {
                    var v_new = cmul(fp.m, v);
                    if !IS_JULIA {
                        v_new.x = v_new.x + pd;
                    }
                    if phoenix_w > 0.0 {
                        v_new = v_new + phoenix_w * cmul(u.phoenix_p, v_prev);
                        v_prev = v;
                    }
                    v = v_new;
                } else {
                    // f' is below f32's range (next to the critical point):
                    // sum the terms at the largest one's scale and move
                    // there, like the delta below.
                    var acc = fe_make(cmul(fp.m, v), fp.e + sv);
                    if !IS_JULIA {
                        acc = fe_add(acc, fe_make(vec2<f32>(px, 0.0), scale_e));
                    }
                    if phoenix_w > 0.0 {
                        acc = fe_add(acc, fe_make(phoenix_w * cmul(u.phoenix_p, v_prev), sv));
                    }
                    if acc.m.x == 0.0 && acc.m.y == 0.0 {
                        if phoenix_w > 0.0 {
                            v_prev = v;
                        }
                        v = acc.m;
                    } else {
                        if phoenix_w > 0.0 {
                            v_prev = ldexp2_sat(v, sv - acc.e);
                        }
                        v = acc.m;
                        sv = acc.e;
                        if !IS_JULIA {
                            pd = ldexp_sat(px, scale_e - sv);
                        }
                    }
                }
            }
            w_old = w;
            let st = advance_delta_scaled(xt, xf, w, sc, sx);
            if st.t == sx {
                w = st.w + d;
                if phoenix_w > 0.0 {
                    w = w + phoenix_w * cmul(u.phoenix_p, w_prev);
                    w_prev = w_old;
                }
            } else {
                // The step's value is at another scale (next to the
                // critical point it is ~e^2, far below 2^sx): add dc and the
                // Phoenix term at the largest addend's scale and move there.
                var acc = fe_make(st.w, st.t);
                if !IS_JULIA {
                    acc = fe_add(acc, fe_make(offset, scale_e));
                }
                if phoenix_w > 0.0 {
                    acc = fe_add(acc, fe_make(phoenix_w * cmul(u.phoenix_p, w_prev), sx));
                }
                if acc.m.x == 0.0 && acc.m.y == 0.0 {
                    w = acc.m;
                    if phoenix_w > 0.0 {
                        w_prev = w_old;
                    }
                } else {
                    // w_old (the pre-step delta) is still needed at the new
                    // scale: it's the next previous delta, and a rebase reads it.
                    w_old = ldexp2_sat(w_old, sx - acc.e);
                    if phoenix_w > 0.0 {
                        w_prev = w_old;
                    }
                    w = acc.m;
                    sx = acc.e;
                    sc = ldexp_sat(1.0, sx);
                    if !IS_JULIA {
                        d = ldexp2_sat(offset, scale_e - sx);
                    }
                }
            }
            m = m + 1u;
            n = n + 1u;
            }

            if m >= ref_len {
                escaped = true; // see the f32 loop's reference-exhausted case
                break;
            }
            xf_old = xf;
            xt_old = xt;
            xf = ref_fe(m);
            xt = ldexp2_sat(xf.m, xf.e);

            // Rebase test |X + e| < |e|, in units of 2^sx. Only possible when
            // |X| is within a few binades of |e| (|w| < 2^DEEP_RENORM_LOG2).
            if xf.e - sx < DEEP_NEAR_LOG2 {
                let q = ldexp2_sat(xf.m, xf.e - sx) + w;
                if dot(q, q) < dot(w, w) {
                    // The new delta y - X[0] stays tiny only if X[0] is
                    // (always, for the set plane), and for Phoenix only if the
                    // previous full value y_{n-1} (its new previous delta) is.
                    let z0_small = (z0f.m.x == 0.0 && z0f.m.y == 0.0)
                        || z0f.e - sx < DEEP_NEAR_LOG2;
                    let prev_small = phoenix_w == 0.0 || xf_old.e - sx < DEEP_NEAR_LOG2;
                    if !(z0_small && prev_small) {
                        rebase_exit = true;
                        break;
                    }
                    if phoenix_w > 0.0 {
                        w_prev = ldexp2_sat(xf_old.m, xf_old.e - sx) + w_old;
                    }
                    w = q - ldexp2_sat(z0f.m, z0f.e - sx);
                    xf = z0f;
                    xt = z0;
                    m = 0u;
                }
            }

            // Leave once both the delta and the derivative fit in f32 (an
            // exactly-zero one always does); otherwise renormalize the
            // mantissas when they drift (exact: powers of two only).
            let kw = deep_log2(w, w_prev);
            let kv = deep_log2(v, v_prev);
            let w_ok = kw == DEEP_ZERO || sx + kw > DEEP_EXIT_LOG2;
            let v_ok = !DE || kv == DEEP_ZERO || sv + kv > DEEP_EXIT_DZ_LOG2;
            if w_ok && v_ok {
                break;
            }
            if kw != DEEP_ZERO && abs(kw) > DEEP_RENORM_LOG2 {
                w = ldexp2_sat(w, -kw);
                w_prev = ldexp2_sat(w_prev, -kw);
                sx = sx + kw;
                sc = ldexp_sat(1.0, sx);
                if !IS_JULIA {
                    d = ldexp2_sat(offset, scale_e - sx);
                }
            }
            if DE && kv != DEEP_ZERO && abs(kv) > DEEP_RENORM_LOG2 {
                v = ldexp2_sat(v, -kv);
                v_prev = ldexp2_sat(v_prev, -kv);
                sv = sv + kv;
                if !IS_JULIA {
                    pd = ldexp_sat(px, scale_e - sv);
                }
            }
        }

        // Hand over to the f32 loop. dc and px may flush to 0 here: they
        // are below f32 rounding of the (now large enough) delta and
        // derivative.
        px_t = ldexp_sat(px, scale_e);
        if !IS_JULIA {
            step_add = ldexp2_sat(offset, scale_e);
        }
        e = ldexp2_sat(w, sx);
        e_prev = ldexp2_sat(w_prev, sx);
        dzs = ldexp2_sat(v, sv);
        dzs_prev = ldexp2_sat(v_prev, sv);
        xm = xt;
        if rebase_exit {
            // Rebase in plain f32: the new delta (y - X[0], and for Phoenix
            // the previous full value) is no longer tiny.
            if phoenix_w > 0.0 {
                e_prev = xt_old + ldexp2_sat(w_old, sx);
            }
            e = (xt + e) - z0;
            xm = z0;
            m = 0u;
        }
        if cut_exit && e.y == 0.0 && w.y != 0.0 {
            // Keep the side of the cut the pixel is on even if e.y flushed
            // to 0: that decides the branch in the next (f32) step.
            e.y = select(-1.17549435e-38, 1.17549435e-38, w.y > 0.0);
        }
        z = xm + e;
        z2 = dot(z, z);
        // Restart periodicity detection from here (check_at must stay ahead
        // of n, or no window would ever close). Nothing is saved until the
        // first window closes: `z` here is at an arbitrary phase, and the
        // pixel still shadows the reference (exactly periodic when it's a
        // minibrot nucleus), so comparing against it would flag exterior
        // pixels as interior (see PERIOD_FIRST_CHECK). The sentinel is far
        // outside the bailout radius, so no return can match it.
        z_saved = vec2<f32>(1e18, 1e18);
        z_cand = z;
        while check_at <= n {
            check_at = check_at * 2u;
        }
    }

    loop {
        // (`escaped` may already be set by the deep phase.)
        if escaped || z2 > bailout_sq {
            escaped = true;
            break;
        }
        if n >= max_iter {
            break; // interior
        }

        // Propagate the derivative of the full orbit (unaffected by rebasing,
        // which only re-expresses the same value). Only when DE is enabled.
        // Phoenix's two-term map adds p·dz_{n-1} and carries the previous dz.
        // f'(z) of this step, shared by DE and the periodicity multiplier.
        // BLA jump (see the deep phase). dc and px are taken unscaled from
        // `offset`/`px`: at depth `step_add`/`px_t` have flushed to 0, but
        // N·dc hasn't. The skipped steps' |f'|^2 product is |det M| (the
        // steps are conformal, or folds of conformal maps). No skipped
        // iterate can be the window's closest to the critical point: a run's
        // radius is below ε·|X| for every X in it.
        //
        // With DE, dzs goes through M, the map's true Jacobian, where the
        // plain loop multiplies by `fprime` (|2Z| for the abs kinds): same
        // magnitude, since the folds are isometries.
        let e_old = e;
        let z_old = z;
        var jumped = false;
        if BLA && m != 0u {
            let hit = bla_lookup(m, log2_mag(e), min(max_iter - n, ref_len - 1u - m));
            if hit.y != 0u {
                let nd = bla_nodes[hit.x];
                let se = select(0, u.scale_exp, DEEP);
                if periodic {
                    let det = abs(nd.m.x * nd.m.w - nd.m.y * nd.m.z);
                    let a2 = ldexp_sat(det, 2 * nd.m_exp);
                    mult2 = mult2 * a2;
                    mult2_cand = mult2_cand * a2;
                }
                if DE {
                    var dn = ldexp2_sat(mat2v(nd.m, dzs), nd.m_exp);
                    if !IS_JULIA {
                        dn = dn + ldexp2_sat(mat2v(nd.n, vec2<f32>(px, 0.0)), nd.n_exp + se);
                    }
                    dzs = dn;
                }
                var en = ldexp2_sat(mat2v(nd.m, e), nd.m_exp);
                if !IS_JULIA {
                    en = en + ldexp2_sat(mat2v(nd.n, offset), nd.n_exp + se);
                }
                e = en;
                m = m + hit.y;
                n = n + hit.y;
                jumped = true;
            }
        }

        if !jumped {
        var fp = vec2<f32>(0.0, 0.0);
        if DE || periodic {
            fp = fprime(z);
        }
        if periodic {
            let fp2 = dot(fp, fp);
            mult2 = mult2 * fp2;
            mult2_cand = mult2_cand * fp2;
        }
        if DE {
            var dzs_new = cmul(fp, dzs);
            if !IS_JULIA {
                dzs_new.x = dzs_new.x + px_t;
            }
            if phoenix_w > 0.0 {
                dzs_new = dzs_new + phoenix_w * cmul(u.phoenix_p, dzs_prev);
                dzs_prev = dzs;
            }
            dzs = dzs_new;
        }

        // Advance the delta by this fractal's formula (+ dc for the set plane).
        // Phoenix additionally adds p·e_{n-1} and carries the previous delta.
        e = advance_delta(xm, e) + step_add;
        if phoenix_w > 0.0 {
            e = e + phoenix_w * cmul(u.phoenix_p, e_prev);
            e_prev = e_old;
        }
        m = m + 1u;
        n = n + 1u;
        }

        // Keep the reference index valid and the delta small.
        if m >= ref_len {
            // Reference exhausted: any pixel that followed it this far has
            // effectively escaped (interior pixels rebase before reaching here).
            z = xm + e;
            escaped = true;
            break;
        }
        xm = ref_at(m);
        z = xm + e;
        z2 = dot(z, z);
        if z2 < dot(e, e) {
            // Rebase to index 0: carry the full value as the new delta. Valid
            // because y_n = X[0] + (y_n - X[0]); for Mandelbrot X[0]=0. The
            // full value `z` (and `z2`) is unchanged by the re-expression.
            // Phoenix: after rebasing the implied previous reference is Y[-1]=0,
            // so the previous delta becomes the full previous value y_{n-1}.
            if phoenix_w > 0.0 {
                e_prev = z_old;
            }
            e = z - z0;
            xm = z0;
            m = 0u;
        }

        if periodic {
            // Closed an attracting cycle in enough consecutive windows:
            // interior (see PERIOD_FIRST_CHECK).
            let d = z - z_saved;
            if !period_hit && mult2 < PERIOD_MAX_MULT2 && dot(d, d) <= PERIOD_EPS2 * z2 {
                period_hit = true;
                period_streak = period_streak + 1u;
                if period_streak >= PERIOD_CONFIRMATIONS {
                    break;
                }
            }
            let dc2 = dot(z - crit, z - crit);
            if dc2 < cand_d2 {
                cand_d2 = dc2;
                z_cand = z;
                mult2_cand = 1.0;
            }
            if n >= check_at {
                if !period_hit {
                    period_streak = 0u;
                }
                period_hit = false;
                z_saved = z_cand;
                mult2 = mult2_cand;
                cand_d2 = 3.0e38;
                // (A BLA jump can pass several window ends at once.)
                while check_at <= n {
                    check_at = check_at * 2u;
                }
            }
        }
    }

    if !escaped {
        return Sample(0.0, 1.0, false); // interior of the set
    }

    // Both escape formulas below (smooth count, DE) assume |f(z)| ~ |z|^2 near
    // escape. Lambda's λz(1-z) + c ~ -λz^2 adds a factor |λ| per step, which
    // made both jump at every band boundary (contour lines in shadow/3D).
    // w = -λz conjugates it to an exact w^2 + C, so measure |w| and |dw|.
    var dzs_esc = dzs;
    if KIND == KIND_LAMBDA {
        z = cmul(u.lambda_l, z);
        dzs_esc = cmul(u.lambda_l, dzs);
    }
    z2 = dot(z, z);

    // Continuous (smooth) iteration count.
    let log_zn = 0.5 * log(max(z2, 1.0));
    let nu = log2(log_zn * INV_LN2);
    let smooth_i = f32(n) + 1.0 - nu;

    // sqrt compresses the huge iteration counts of deep zooms so the palette
    // varies smoothly instead of aliasing into speckle.
    let ci = sqrt(max(smooth_i, 0.0));

    var de = 1.0;
    if DE {
        // Exterior distance estimate |z|·ln|z| / |dz|, already in pixels since
        // `dzs` = px·dz. We darken toward the boundary (< ~1 px away) so
        // filaments stay crisp instead of aliasing into speckle. If |dzs|
        // overflowed (far sub-pixel from the set), de -> 0 and the boundary
        // simply reads as dark, which is the correct limit.
        // Shadow/3D use DE as a height field, so it must stay unclamped: any
        // cap flattens everything farther than that from the set into a
        // uniform plateau (a visible circle around the set when zoomed out).
        // 3D saturates heights smoothly itself (`sdf` in colorize.wgsl).
        let zmag = sqrt(max(z2, 1.0));
        let dzmag = sqrt(max(dot(dzs_esc, dzs_esc), 1e-30));
        // |z|·ln|z|/|dz| is G/|G'| (G the potential, ln|z|/2^n). Far from the
        // set G ~ ln r, so it grows like r·ln r rather than r: zoomed far out,
        // the 3D cone got steeper with every zoom step, down to a texel-wide
        // needle cut off above the plateau. Replacing G by 2(1 - e^(-G/2))
        // leaves it unchanged near the set (G -> 0, and ~0.8x at the edge of
        // the default view) but caps it at 2, so far away DE grows like 2r
        // and the cone stops narrowing. G underflows to 0 past ~150
        // iterations, where the factor is 1 anyway. G divides by the map's
        // degree d per step, not 2: with 2^-n, pixels either side of a band
        // boundary got G off by d/2, a DE seam on every band for d != 2.
        let g = log(zmag) * pow(escape_degree(), -f32(n));
        let far = select(1.0, 2.0 * (1.0 - exp(-g / 2.0)) / g, g > 1e-4);
        let max_de = select(1.0, 1e30, u.shadow != 0u);
        de = clamp(zmag * log(zmag) / dzmag * far, 0.0, max_de);
    }
    return Sample(ci, de, true);
}

// Degree d of the current map at infinity (|f(z)| ~ |z|^d), which sets how
// fast the potential G = ln|z_n| / d^n shrinks per step. Complex Multibrot's
// |z^p| = |z|^Re(p)·e^(-Im(p)·arg z) grows like |z|^Re(p) (arg is bounded).
// A morph blend is dominated by the higher degree.
fn kind_degree(kind: u32) -> f32 {
    if kind == KIND_MULTIBROT {
        return f32(clamp(u.power, 2u, MULTIBROT_MAX_POWER));
    } else if kind == KIND_COMPLEX_MULTIBROT {
        return max(u.complex_power.x, 1.0);
    }
    return 2.0;
}

fn escape_degree() -> f32 {
    let d = kind_degree(KIND);
    if MORPH {
        return max(d, kind_degree(u.morph_from));
    }
    return d;
}

// 1 / ln(2), for the smooth iteration count's log2(ln|z| / ln 2).
const INV_LN2: f32 = 1.4426950408889634;

// Map a sample's escape data through the palette (+ DE darkening). This is the
// only color-dependent step, so it can be redone without re-iterating. Interior
// samples are black.
fn color_sample(s: Sample) -> vec3<f32> {
    if !s.escaped {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return classic_color(s.ci, s.de);
}

// Supersampled escape data at one point: average (ci, DE factor) over an
// `aa`×`aa` grid's escaped sub-samples, plus the fraction that landed in the
// interior. Shared by `fs_data` (1 sample), `fs_refine` (the AA grid, only on
// pixels that need it) and `fs_color`'s shadow branch (used both at the pixel
// and at its two neighbours, to build a DE height field without a texture
// round-trip).
fn aggregate_sample(base: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, px: f32, aa: u32) -> vec3<f32> {
    let inv = 1.0 / f32(aa);
    var ci_sum = 0.0;
    var de_sum = 0.0;
    var escaped_n = 0u;
    for (var sy: u32 = 0u; sy < aa; sy = sy + 1u) {
        for (var sx: u32 = 0u; sx < aa; sx = sx + 1u) {
            let jx = (f32(sx) + 0.5) * inv - 0.5;
            let jy = (f32(sy) + 0.5) * inv - 0.5;
            let s = iterate_sample(base + jx * dx + jy * dy, px);
            if s.escaped {
                ci_sum = ci_sum + s.ci;
                de_sum = de_sum + s.de;
                escaped_n = escaped_n + 1u;
            }
        }
    }
    let total = f32(aa * aa);
    let ci_avg = select(0.0, ci_sum / f32(escaped_n), escaped_n > 0u);
    let de_avg = select(1.0, de_sum / f32(escaped_n), escaped_n > 0u);
    let interior_frac = 1.0 - f32(escaped_n) / total;
    return vec3<f32>(ci_avg, de_avg, interior_frac);
}

// Pixel footprint in complex units, |(|dx| + |dy|)|. Not `length()` directly:
// that squares its argument, and below ~1e-19 per pixel (half-height ~1e-16,
// sooner for the 2x-resolution 3D texture) the square drops under f32's
// smallest normal and flushes to 0, making px = 0 and DE meaningless.
// Normalizing by the largest component first keeps the square near 1.
fn pixel_size(dx: vec2<f32>, dy: vec2<f32>) -> f32 {
    let a = abs(dx) + abs(dy);
    let m = max(a.x, a.y);
    if m == 0.0 {
        return 0.0;
    }
    return m * length(a / m);
}

// Iteration pass: write per-pixel escape data (color-independent) so a colour
// change is remapped by the cheap colourise pass without re-iterating.
//   R = ci (palette parameter), G = DE factor, B = interior fraction (for AA).
// Always one sample per pixel: anti-aliasing is added afterwards, only where
// it matters, by `fs_refine`.
@fragment
fn fs_data(in: VsOut) -> @location(0) vec4<f32> {
    let base = in.centered * u.span + u.dc_offset;
    let dx = dpdx(base);
    let dy = dpdy(base);
    let px = pixel_size(dx, dy);

    return vec4<f32>(aggregate_sample(base, dx, dy, px, 1u), 1.0);
}

// Adaptive-AA thresholds for `fs_refine`: a pixel is supersampled only if a
// 4-neighbour's 1-spp sample differs from its own by more than this. `ci`
// steps are palette-phase steps of `ci * color_scale` (color_scale <= 1 in the
// UI), so 0.02 keeps anything visibly banded; DE is compared relative to its
// own magnitude (it's in pixels, unbounded for shadow/3D height fields).
const AA_CI_EPS: f32 = 0.02;
const AA_DE_EPS: f32 = 0.1;

fn aa_differs(c: vec4<f32>, n: vec4<f32>) -> bool {
    if c.b != n.b {
        return true; // interior / exterior boundary
    }
    if c.b != 0.0 {
        return false; // both interior: uniformly black
    }
    return abs(n.r - c.r) > AA_CI_EPS || abs(n.g - c.g) > AA_DE_EPS * max(c.g, 0.1);
}

// Adaptive anti-aliasing pass (only run when AA is on): reads `fs_data`'s
// 1-spp texture and re-iterates the full AA grid only for pixels whose
// neighbourhood isn't smooth (set boundary, filaments, palette discontinuities).
// Everywhere else the centre sample already equals the grid average to within
// the thresholds above, so it's copied — which skips the AA cost entirely for
// the interior (the most expensive pixels, each burning max_iter) and for the
// smooth exterior.
@fragment
fn fs_refine(in: VsOut) -> @location(0) vec4<f32> {
    // Derivatives first, while control flow is still uniform.
    let base = in.centered * u.span + u.dc_offset;
    let dx = dpdx(base);
    let dy = dpdy(base);
    let px = pixel_size(dx, dy);

    let p = vec2<i32>(in.pos.xy);
    let hi = vec2<i32>(textureDimensions(coarse_tex)) - vec2<i32>(1, 1);
    let c = textureLoad(coarse_tex, p, 0);
    let l = textureLoad(coarse_tex, max(p - vec2<i32>(1, 0), vec2<i32>(0, 0)), 0);
    let r = textureLoad(coarse_tex, min(p + vec2<i32>(1, 0), hi), 0);
    let t = textureLoad(coarse_tex, max(p - vec2<i32>(0, 1), vec2<i32>(0, 0)), 0);
    let b = textureLoad(coarse_tex, min(p + vec2<i32>(0, 1), hi), 0);
    if aa_differs(c, l) || aa_differs(c, r) || aa_differs(c, t) || aa_differs(c, b) {
        return vec4<f32>(aggregate_sample(base, dx, dy, px, max(u.aa_level, 1u)), 1.0);
    }
    return c;
}

// Combined iterate + colour in a single pass, for PNG export (which never needs
// incremental recolouring). The interactive path uses fs_data (+ fs_refine) +
// the colourise pass so colour changes skip iteration. Export always runs the
// full AA grid on every pixel, for maximum quality.
@fragment
fn fs_color(in: VsOut) -> @location(0) vec4<f32> {
    let base = in.centered * u.span + u.dc_offset;
    let dx = dpdx(base);
    let dy = dpdy(base);
    let px = pixel_size(dx, dy);
    let aa = max(u.aa_level, 1u);

    if u.shadow != 0u {
        // No data texture to sample neighbours from (this pass never runs
        // one), so build the same DE height field colorize.wgsl reads from
        // the texture by aggregating live, at the pixel and its two
        // neighbours a `dx`/`dy` step away.
        let here = aggregate_sample(base, dx, dy, px, aa);
        if here.z != 0.0 {
            return vec4<f32>(shadow_interior_color(), 1.0);
        }
        let right = aggregate_sample(base + dx, dx, dy, px, aa);
        let down = aggregate_sample(base + dy, dx, dy, px, aa);
        let normal = normal_from_heights(here.y, right.y, down.y);
        return vec4<f32>(shadow_color(normal, here.x), 1.0);
    }

    let inv = 1.0 / f32(aa);
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var sy: u32 = 0u; sy < aa; sy = sy + 1u) {
        for (var sx: u32 = 0u; sx < aa; sx = sx + 1u) {
            let jx = (f32(sx) + 0.5) * inv - 0.5;
            let jy = (f32(sy) + 0.5) * inv - 0.5;
            acc = acc + color_sample(iterate_sample(base + jx * dx + jy * dy, px));
        }
    }
    return vec4<f32>(acc / f32(aa * aa), 1.0);
}
