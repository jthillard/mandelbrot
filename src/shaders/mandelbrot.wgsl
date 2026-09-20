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

// Binomial coefficient C(n, k) as f32 (exact for the small powers we use).
fn binom(n: u32, k: u32) -> f32 {
    var num = 1.0;
    var den = 1.0;
    for (var i: u32 = 0u; i < k; i = i + 1u) {
        num = num * f32(n - i);
        den = den * f32(i + 1u);
    }
    return num / den;
}

// Perturbation delta for z -> z^p: sum_{k=1}^{p} C(p,k) Z^{p-k} e^k. Expanded so
// the large z^p term is never formed (that would cancel catastrophically).
fn multibrot_delta(z: vec2<f32>, e: vec2<f32>, p: u32) -> vec2<f32> {
    var zp: array<vec2<f32>, 9>; // Z^0 .. Z^8
    zp[0] = vec2<f32>(1.0, 0.0);
    for (var j: u32 = 1u; j <= p; j = j + 1u) {
        zp[j] = cmul(zp[j - 1u], z);
    }
    var acc = vec2<f32>(0.0, 0.0);
    var ek = vec2<f32>(1.0, 0.0); // e^0
    for (var k: u32 = 1u; k <= p; k = k + 1u) {
        ek = cmul(ek, e); // e^k
        acc = acc + binom(p, k) * cmul(zp[p - k], ek);
    }
    return acc;
}

// Number of terms kept in `complex_multibrot_delta`'s series. Truncation, not
// exactness: unlike `multibrot_delta` (a finite binomial sum for an integer
// power), a complex power has no finite expansion, so this converges rather
// than terminates. Fine as long as perturbation's usual invariant (|e| << |z|,
// kept true by rebasing) holds, since each extra term is O(w^k) smaller.
const COMPLEX_MULTIBROT_TERMS: u32 = 16u;

// Perturbation delta for z -> z^p with a complex p: (Z+e)^p - Z^p.
//
// When |e| << |Z| (the common case: it's the whole reason perturbation
// works), forming Z+e directly would round e away in f32, so instead expand
// = Z^p * ((1+w)^p - 1), w = e/Z, as a Taylor series in w: (1+w)^p - 1 =
// sum_{k=1}^N C(p,k) w^k, with the complex binomial coefficient built up
// incrementally: C(p,k) = C(p,k-1) * (p-(k-1)) / k. Unlike `multibrot_delta`
// (a finite binomial sum for an integer power), this only *converges* — and
// only for |w| < 1 — rather than terminating exactly.
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
        var wk = vec2<f32>(1.0, 0.0); // w^0
        var coef = vec2<f32>(1.0, 0.0); // C(p,0)
        var acc = vec2<f32>(0.0, 0.0);
        for (var k: u32 = 1u; k <= COMPLEX_MULTIBROT_TERMS; k = k + 1u) {
            coef = cdiv(cmul(coef, p - vec2<f32>(f32(k - 1u), 0.0)), vec2<f32>(f32(k), 0.0));
            wk = cmul(wk, w);
            acc = acc + cmul(coef, wk);
        }
        return cmul(cpow(z, p), acc);
    }
    return cpow(z + e, p) - cpow(z, p);
}

// One perturbation step of the current fractal's delta: e -> f(Z+e) - f(Z),
// where `z` is the reference orbit value X_m. `step_add` (dc) is added by the
// caller. Must match `FractalKind` on the CPU side.
fn advance_delta(z: vec2<f32>, e: vec2<f32>) -> vec2<f32> {
    if u.kind == KIND_BURNING_SHIP {
        // (|x| + i|y|)^2 has real part x^2 - y^2 (an ordinary square delta) and
        // imaginary part 2|x y|. The imaginary delta is 2(|x y| - |X Y|); diffabs
        // computes it exactly, even where the product x y changes sign — which the
        // old sign(X)sign(Y) shortcut got wrong whenever the delta was large
        // enough to flip it (all the time at shallow zoom).
        let base = 2.0 * cmul(z, e) + cmul(e, e);
        let dp = z.x * e.y + z.y * e.x + e.x * e.y;
        return vec2<f32>(base.x, 2.0 * diffabs(z.x * z.y, dp));
    } else if u.kind == KIND_TRICORN {
        let cz = conj(z);
        let ce = conj(e);
        return 2.0 * cmul(cz, ce) + cmul(ce, ce);
    } else if u.kind == KIND_MULTIBROT {
        return multibrot_delta(z, e, clamp(u.power, 2u, 8u));
    } else if u.kind == KIND_CELTIC {
        // z^2 delta split: sq.x = delta of Re(z^2), sq.y = delta of Im(z^2).
        // Celtic abs the real output, so |Re(z^2)| delta = diffabs(Re(Z^2), sq.x).
        let sq = 2.0 * cmul(z, e) + cmul(e, e);
        return vec2<f32>(diffabs(z.x * z.x - z.y * z.y, sq.x), sq.y);
    } else if u.kind == KIND_BUFFALO {
        // Abs both outputs: real |Re(z^2)|, imag -|Im(z^2)| (Im(Z^2) = 2 X Y).
        let sq = 2.0 * cmul(z, e) + cmul(e, e);
        return vec2<f32>(diffabs(z.x * z.x - z.y * z.y, sq.x),
                         -diffabs(2.0 * z.x * z.y, sq.y));
    } else if u.kind == KIND_PERPENDICULAR {
        // real x^2 - y^2 (ordinary square delta), imag -2 x |y|.
        // d(-2 x |y|) = -2[ X·(|Y+ey|-|Y|) + ex·|Y+ey| ]; diffabs gives |Y+ey|-|Y|.
        let sq = 2.0 * cmul(z, e) + cmul(e, e);
        let da = diffabs(z.y, e.y);        // |Y + ey| - |Y|
        let abs_yf = abs(z.y) + da;        // |Y + ey|
        return vec2<f32>(sq.x, -2.0 * (z.x * da + e.x * abs_yf));
    } else if u.kind == KIND_LAMBDA {
        // Lambda map: z^{n+1} = λ·z·(1-z). Delta: e = λ·e·(1-2z-e).
        let one_minus_2z_minus_e = vec2<f32>(1.0 - 2.0 * z.x - e.x, -2.0 * z.y - e.y);
        return cmul(u.lambda_l, cmul(e, one_minus_2z_minus_e));
    } else if u.kind == KIND_COMPLEX_MULTIBROT {
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
    if u.kind == KIND_MULTIBROT {
        let p = clamp(u.power, 2u, 8u);
        var zk = vec2<f32>(1.0, 0.0); // Z^0
        for (var k: u32 = 1u; k < p; k = k + 1u) {
            zk = cmul(zk, z); // -> Z^{p-1}
        }
        return f32(p) * zk;
    } else if u.kind == KIND_LAMBDA {
        // Lambda: f'(z) = λ·(1-2z).
        return cmul(u.lambda_l, vec2<f32>(1.0 - 2.0 * z.x, -2.0 * z.y));
    } else if u.kind == KIND_COMPLEX_MULTIBROT {
        // f'(z) = p * z^(p-1).
        return cmul(u.complex_power, cpow(z, u.complex_power - vec2<f32>(1.0, 0.0)));
    }
    return 2.0 * z;
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
    let z0 = ref_orbit[0]; // reference start (0 for Mandelbrot, center for Julia)

    var step_add = offset;
    var e = vec2<f32>(0.0, 0.0);
    // Orbit derivative for distance estimation. For the set plane it is d/dc
    // (starts at 0, gains +1 each step); for Julia it is d/dz0 (starts at 1).
    var dz = vec2<f32>(0.0, 0.0);
    var dz_seed = vec2<f32>(1.0, 0.0);
    // Previous-iterate state for the Phoenix two-term recurrence (delta of
    // y_{n-1}, and its derivative for DE). Both start at 0 (y_{-1} = 0).
    var e_prev = vec2<f32>(0.0, 0.0);
    var dz_prev = vec2<f32>(0.0, 0.0);
    if u.is_julia != 0u {
        step_add = vec2<f32>(0.0, 0.0);
        e = offset;
        dz = vec2<f32>(1.0, 0.0);
        dz_seed = vec2<f32>(0.0, 0.0);
    }

    var m: u32 = 0u;              // reference index; invariant: y_n = X[m] + e
    var n: u32 = 0u;              // total iteration count
    var z = vec2<f32>(0.0, 0.0);  // full value y_n, kept for coloring
    var escaped = false;

        loop {
            let xm = ref_orbit[m];
            z = xm + e;

            let z2 = dot(z, z);
            if z2 > u.bailout_sq {
                escaped = true;
                break;
            }
            if n >= u.max_iter {
                break; // interior
            }

        // Propagate the derivative of the full orbit (unaffected by rebasing,
        // which only re-expresses the same value). Only when DE is enabled.
        // Phoenix's two-term map adds p·dz_{n-1} and carries the previous dz.
            if u.de_coloring != 0u {
                var dz_new = cmul(fprime(z), dz) + dz_seed;
                if u.kind == KIND_PHOENIX {
                    dz_new = dz_new + cmul(u.phoenix_p, dz_prev);
                    dz_prev = dz;
                }
                dz = dz_new;
            }

        // Advance the delta by this fractal's formula (+ dc for the set plane).
        // Phoenix additionally adds p·e_{n-1} and carries the previous delta.
            let e_old = e;
            e = advance_delta(xm, e) + step_add;
            if u.kind == KIND_PHOENIX {
                e = e + cmul(u.phoenix_p, e_prev);
                e_prev = e_old;
            }
            m = m + 1u;
            n = n + 1u;

        // Keep the reference index valid and the delta small.
            if m >= u.ref_len {
            // Reference exhausted: any pixel that followed it this far has
            // effectively escaped (interior pixels rebase before reaching here).
                z = ref_orbit[u.ref_len - 1u] + e;
                escaped = true;
                break;
            }
            let y = ref_orbit[m] + e;
            if dot(y, y) < dot(e, e) {
            // Rebase to index 0: carry the full value as the new delta. Valid
            // because y_n = X[0] + (y_n - X[0]); for Mandelbrot X[0]=0.
            // Phoenix: after rebasing the implied previous reference is Y[-1]=0,
            // so the previous delta becomes the full previous value y_n (= z).
                if u.kind == KIND_PHOENIX {
                    e_prev = z;
                }
                e = y - z0;
                m = 0u;
            }
        }

    if !escaped {
        return Sample(0.0, 1.0, false); // interior of the set
    }

    let z2 = dot(z, z);

    // Continuous (smooth) iteration count.
    let log_zn = 0.5 * log(max(z2, 1.0));
    let nu = log2(log_zn / log(2.0));
    let smooth_i = f32(n) + 1.0 - nu;

    // sqrt compresses the huge iteration counts of deep zooms so the palette
    // varies smoothly instead of aliasing into speckle.
    let ci = sqrt(max(smooth_i, 0.0));

    var de = 1.0;
    if u.de_coloring != 0u {
        // Exterior distance estimate (complex-plane units): |z|·ln|z| / |dz|.
        // Divided by the pixel footprint it becomes a distance in pixels; we
        // darken toward the boundary (< ~1 px away) so filaments stay crisp
        // instead of aliasing into speckle. If |dz| overflowed, de -> 0 and the
        // boundary simply reads as dark, which is the correct limit.
        let zmag = sqrt(max(z2, 1.0));
        let dzmag = sqrt(max(dot(dz, dz), 1e-20));
        let d = zmag * log(zmag) / dzmag;
        var max_de = 1.;
        if u.shadow != 0u {
            max_de = 1000.;
        }
        de = clamp(d / max(px, 1e-30), 0.0, max_de);
    }
    return Sample(ci, de, true);
}

// Map a sample's escape data through the palette (+ DE darkening). This is the
// only color-dependent step, so it can be redone without re-iterating. Interior
// samples are black.
fn color_sample(s: Sample) -> vec3<f32> {
    if !s.escaped {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    return classic_color(s.ci, s.de);
}

// Supersampled escape data at one point: average (ci, DE factor) over the
// AA grid's escaped sub-samples, plus the fraction that landed in the
// interior. Shared by `fs_data` (writes it straight to the data texture) and
// `fs_color`'s shadow branch (used both at the pixel and at its two
// neighbours, to build a DE height field without a texture round-trip).
fn aggregate_sample(base: vec2<f32>, dx: vec2<f32>, dy: vec2<f32>, px: f32) -> vec3<f32> {
    let aa = max(u.aa_level, 1u);
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
// AA is grid-supersampled here; the interior fraction lets the colourise pass
// anti-alias the set boundary (blend toward black) after the fact.
@fragment
fn fs_data(in: VsOut) -> @location(0) vec4<f32> {
    let base = in.centered * u.span + u.dc_offset;
    let dx = dpdx(base);
    let dy = dpdy(base);
    let px = length(abs(dx) + abs(dy));

    return vec4<f32>(aggregate_sample(base, dx, dy, px), 1.0);
}

// Combined iterate + colour in a single pass, for PNG export (which never needs
// incremental recolouring). The interactive path uses fs_data + the colourise
// pass so colour changes skip iteration.
@fragment
fn fs_color(in: VsOut) -> @location(0) vec4<f32> {
    let base = in.centered * u.span + u.dc_offset;
    let dx = dpdx(base);
    let dy = dpdy(base);
    let px = length(abs(dx) + abs(dy));

    if u.shadow != 0u {
        // No data texture to sample neighbours from (this pass never runs
        // one), so build the same DE height field colorize.wgsl reads from
        // the texture by aggregating live, at the pixel and its two
        // neighbours a `dx`/`dy` step away.
        let here = aggregate_sample(base, dx, dy, px);
        if here.z != 0.0 {
            return vec4<f32>(0.1, 0.1, 0.1, 1.0);
        }
        let right = aggregate_sample(base + dx, dx, dy, px);
        let down = aggregate_sample(base + dy, dx, dy, px);
        let normal = normal_from_heights(here.y, right.y, down.y);
        return vec4<f32>(shadow_color(normal), 1.0);
    }

    let aa = max(u.aa_level, 1u);
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
