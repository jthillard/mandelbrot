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

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> ref_orbit: array<vec2<f32>>;
// Only read by `fs_color`'s shadow branch (custom-lights palette); the
// iteration pass (`fs_data`) never touches it.
@group(0) @binding(2) var<uniform> lights: array<Light, 16>;
// Only read by the adaptive-AA refine pass (`fs_refine`): the 1-sample-per-
// pixel data texture written by `fs_data`, which decides where to supersample.
@group(1) @binding(0) var coarse_tex: texture_2d<f32>;

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
    let y = z + e;
    var s = vec2<f32>(1.0, 0.0);
    var zj = vec2<f32>(1.0, 0.0);
    for (var j: u32 = 1u; j < p; j = j + 1u) {
        zj = cmul(zj, z); // Z^j
        s = cmul(s, y) + zj;
    }
    return cmul(e, s);
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
fn complex_multibrot_delta(z: vec2<f32>, e: vec2<f32>, p: vec2<f32>) -> vec2<f32> {
    // |w|^2 = |e|^2 / |Z|^2; inf or nan (Z ~ 0, or both ~ 0) correctly fails
    // the `< 0.25` test below and falls through to the direct branch.
    let w2 = dot(e, e) / dot(z, z);
    if w2 < 0.25 {
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
    return cpow(z + e, p) - cpow(z, p);
}

// One perturbation step of the current fractal's delta: e -> f(Z+e) - f(Z),
// where `z` is the reference orbit value X_m. `step_add` (dc) is added by the
// caller. Must match `FractalKind` on the CPU side.
fn advance_delta(z: vec2<f32>, e: vec2<f32>) -> vec2<f32> {
    if KIND == KIND_BURNING_SHIP {
        // (|x| + i|y|)^2 has real part x^2 - y^2 (an ordinary square delta) and
        // imaginary part 2|x y|. The imaginary delta is 2(|x y| - |X Y|); diffabs
        // computes it exactly, even where the product x y changes sign — which the
        // old sign(X)sign(Y) shortcut got wrong whenever the delta was large
        // enough to flip it (all the time at shallow zoom).
        let base = 2.0 * cmul(z, e) + cmul(e, e);
        let dp = z.x * e.y + z.y * e.x + e.x * e.y;
        return vec2<f32>(base.x, 2.0 * diffabs(z.x * z.y, dp));
    } else if KIND == KIND_TRICORN {
        let cz = conj(z);
        let ce = conj(e);
        return 2.0 * cmul(cz, ce) + cmul(ce, ce);
    } else if KIND == KIND_MULTIBROT {
        return multibrot_delta(z, e, clamp(u.power, 2u, 8u));
    } else if KIND == KIND_CELTIC {
        // z^2 delta split: sq.x = delta of Re(z^2), sq.y = delta of Im(z^2).
        // Celtic abs the real output, so |Re(z^2)| delta = diffabs(Re(Z^2), sq.x).
        let sq = 2.0 * cmul(z, e) + cmul(e, e);
        return vec2<f32>(diffabs(z.x * z.x - z.y * z.y, sq.x), sq.y);
    } else if KIND == KIND_BUFFALO {
        // Abs both outputs: real |Re(z^2)|, imag -|Im(z^2)| (Im(Z^2) = 2 X Y).
        let sq = 2.0 * cmul(z, e) + cmul(e, e);
        return vec2<f32>(diffabs(z.x * z.x - z.y * z.y, sq.x),
                         -diffabs(2.0 * z.x * z.y, sq.y));
    } else if KIND == KIND_PERPENDICULAR {
        // real x^2 - y^2 (ordinary square delta), imag -2 x |y|.
        // d(-2 x |y|) = -2[ X·(|Y+ey|-|Y|) + ex·|Y+ey| ]; diffabs gives |Y+ey|-|Y|.
        let sq = 2.0 * cmul(z, e) + cmul(e, e);
        let da = diffabs(z.y, e.y);        // |Y + ey| - |Y|
        let abs_yf = abs(z.y) + da;        // |Y + ey|
        return vec2<f32>(sq.x, -2.0 * (z.x * da + e.x * abs_yf));
    } else if KIND == KIND_LAMBDA {
        // Lambda map: z^{n+1} = λ·z·(1-z). Delta: e = λ·e·(1-2z-e).
        let one_minus_2z_minus_e = vec2<f32>(1.0 - 2.0 * z.x - e.x, -2.0 * z.y - e.y);
        return cmul(u.lambda_l, cmul(e, one_minus_2z_minus_e));
    } else if KIND == KIND_COMPLEX_MULTIBROT {
        return complex_multibrot_delta(z, e, u.complex_power);
    }
    return 2.0 * cmul(z, e) + cmul(e, e); // Mandelbrot (and Phoenix square part)
}

// Derivative f'(Z) of the iteration map at the full value Z, used to propagate
// the orbit derivative for distance-estimation shading. Exact for the
// holomorphic kinds (z^2 -> 2Z, z^p -> p Z^{p-1}); for the non-holomorphic
// Burning Ship / Tricorn we use |f'| ~ |2Z|, which keeps the DE magnitude close
// enough to de-speckle filaments.
fn fprime(z: vec2<f32>) -> vec2<f32> {
    if KIND == KIND_MULTIBROT {
        let p = clamp(u.power, 2u, 8u);
        var zk = z; // Z^1
        for (var k: u32 = 2u; k < p; k = k + 1u) {
            zk = cmul(zk, z); // -> Z^{p-1}
        }
        return f32(p) * zk;
    } else if KIND == KIND_LAMBDA {
        // Lambda: f'(z) = λ·(1-2z).
        return cmul(u.lambda_l, vec2<f32>(1.0 - 2.0 * z.x, -2.0 * z.y));
    } else if KIND == KIND_COMPLEX_MULTIBROT {
        // f'(z) = p * z^(p-1).
        return cmul(u.complex_power, cpow(z, u.complex_power - vec2<f32>(1.0, 0.0)));
    }
    return 2.0 * z;
}

// Periodicity (interior) detection, Brent-style: the full orbit value is
// saved at iterations PERIOD_FIRST_CHECK, 2x that, 4x ..., and every later
// iterate is compared against the last saved one. Returning within
// PERIOD_EPS2 (relative, squared) means the orbit has closed a cycle.
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
// Phoenix's two-term map would need a 2x2 Jacobian, so it's excluded. So is
// Complex Multibrot without DE, where `fprime` would add a second `cpow`
// (log/atan2/exp) per step for a check that rarely fires on its views.
const PERIOD_FIRST_CHECK: u32 = 16u;
const PERIOD_EPS2: f32 = 1e-12;
const PERIOD_MAX_MULT2: f32 = 0.25;
const PERIOD_CONFIRMATIONS: u32 = 2u;

// Whether `iterate_sample` runs periodicity detection for this kind (folds to
// a constant per pipeline).
fn periodic_enabled() -> bool {
    if KIND == KIND_PHOENIX {
        return false;
    }
    if KIND == KIND_COMPLEX_MULTIBROT && !DE {
        return false;
    }
    return true;
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
// (c is fixed, so nothing is added per step).
fn iterate_sample(offset: vec2<f32>, px: f32) -> Sample {
    // Loop invariants, read once instead of on every iteration.
    let max_iter = u.max_iter;
    let bailout_sq = u.bailout_sq;
    let ref_len = u.ref_len;
    let z0 = ref_orbit[0]; // reference start (0 for Mandelbrot, center for Julia)

    // Main cardioid / period-2 bulb bypass: those points never escape, so skip
    // iterating them (they'd otherwise all burn the full max_iter). `offset` is
    // relative to the reference center; the absolute c is recovered from the
    // orbit itself, since X_1 = X_0^2 + C_ref = C_ref. That's only f32-accurate,
    // so skip the test once a pixel is smaller than that error (deep zoom),
    // where it could misclassify pixels right at the boundary.
    if KIND == KIND_MANDELBROT && !IS_JULIA && ref_len > 1u && px > 1e-6 {
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
    // Orbit derivative for distance estimation. For the set plane it is d/dc
    // (starts at 0, gains +1 each step); for Julia it is d/dz0 (starts at 1).
    var dz = vec2<f32>(0.0, 0.0);
    if IS_JULIA {
        step_add = vec2<f32>(0.0, 0.0);
        e = offset;
        dz = vec2<f32>(1.0, 0.0);
    }
    // Previous-iterate state for the Phoenix two-term recurrence (delta of
    // y_{n-1}, and its derivative for DE). Both start at 0 (y_{-1} = 0).
    var e_prev = vec2<f32>(0.0, 0.0);
    var dz_prev = vec2<f32>(0.0, 0.0);

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

    loop {
        if z2 > bailout_sq {
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
        var fp = vec2<f32>(0.0, 0.0);
        if DE || periodic {
            fp = fprime(z);
        }
        if periodic {
            mult2 = mult2 * dot(fp, fp);
        }
        if DE {
            var dz_new = cmul(fp, dz);
            if !IS_JULIA {
                dz_new.x = dz_new.x + 1.0;
            }
            if KIND == KIND_PHOENIX {
                dz_new = dz_new + cmul(u.phoenix_p, dz_prev);
                dz_prev = dz;
            }
            dz = dz_new;
        }

        // Advance the delta by this fractal's formula (+ dc for the set plane).
        // Phoenix additionally adds p·e_{n-1} and carries the previous delta.
        let e_old = e;
        let z_old = z;
        e = advance_delta(xm, e) + step_add;
        if KIND == KIND_PHOENIX {
            e = e + cmul(u.phoenix_p, e_prev);
            e_prev = e_old;
        }
        m = m + 1u;
        n = n + 1u;

        // Keep the reference index valid and the delta small.
        if m >= ref_len {
            // Reference exhausted: any pixel that followed it this far has
            // effectively escaped (interior pixels rebase before reaching here).
            z = xm + e;
            escaped = true;
            break;
        }
        xm = ref_orbit[m];
        z = xm + e;
        z2 = dot(z, z);
        if z2 < dot(e, e) {
            // Rebase to index 0: carry the full value as the new delta. Valid
            // because y_n = X[0] + (y_n - X[0]); for Mandelbrot X[0]=0. The
            // full value `z` (and `z2`) is unchanged by the re-expression.
            // Phoenix: after rebasing the implied previous reference is Y[-1]=0,
            // so the previous delta becomes the full previous value y_{n-1}.
            if KIND == KIND_PHOENIX {
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
            if n == check_at {
                if !period_hit {
                    period_streak = 0u;
                }
                period_hit = false;
                z_saved = z;
                mult2 = 1.0;
                check_at = check_at * 2u;
            }
        }
    }

    if !escaped {
        return Sample(0.0, 1.0, false); // interior of the set
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
        // Exterior distance estimate (complex-plane units): |z|·ln|z| / |dz|.
        // Divided by the pixel footprint it becomes a distance in pixels; we
        // darken toward the boundary (< ~1 px away) so filaments stay crisp
        // instead of aliasing into speckle. If |dz| overflowed, de -> 0 and the
        // boundary simply reads as dark, which is the correct limit.
        let zmag = sqrt(max(z2, 1.0));
        let dzmag = sqrt(max(dot(dz, dz), 1e-20));
        let d = zmag * log(zmag) / dzmag;
        let max_de = select(1.0, 1000.0, u.shadow != 0u);
        de = clamp(d / max(px, 1e-30), 0.0, max_de);
    }
    return Sample(ci, de, true);
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
    let px = length(abs(dx) + abs(dy));

    return vec4<f32>(aggregate_sample(base, dx, dy, px, 1u), 1.0);
}

// Adaptive-AA thresholds for `fs_refine`: a pixel is supersampled only if a
// 4-neighbour's 1-spp sample differs from its own by more than this. `ci`
// steps are palette-phase steps of `ci * color_scale` (color_scale <= 1 in the
// UI), so 0.02 keeps anything visibly banded; DE is compared relative to its
// own magnitude (it's in pixels, up to 1000 for shadow/3D height fields).
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
    let px = length(abs(dx) + abs(dy));

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
    let px = length(abs(dx) + abs(dy));
    let aa = max(u.aa_level, 1u);

    if u.shadow != 0u {
        // No data texture to sample neighbours from (this pass never runs
        // one), so build the same DE height field colorize.wgsl reads from
        // the texture by aggregating live, at the pixel and its two
        // neighbours a `dx`/`dy` step away.
        let here = aggregate_sample(base, dx, dy, px, aa);
        if here.z != 0.0 {
            return vec4<f32>(0.1, 0.1, 0.1, 1.0);
        }
        let right = aggregate_sample(base + dx, dx, dy, px, aa);
        let down = aggregate_sample(base + dy, dx, dy, px, aa);
        let normal = normal_from_heights(here.y, right.y, down.y);
        return vec4<f32>(shadow_color(normal), 1.0);
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
