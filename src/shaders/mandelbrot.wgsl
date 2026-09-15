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

struct Uniforms {
    span: vec2<f32>,
    max_iter: u32,
    ref_len: u32,
    color_offset: f32,
    color_scale: f32,
    bailout_sq: f32,
    is_julia: u32,
    palette_id: u32,
    aa_level: u32,
    // Iteration formula (see the KIND_* constants below).
    kind: u32,
    // Exponent for the Multibrot kind.
    power: u32,
    dc_offset: vec2<f32>,
    // Distortion constant p for the Phoenix map (z^2 + c + p*z_{n-1}); unused
    // by other kinds. Placed by dc_offset so both vec2s stay 8-byte aligned.
    phoenix_p: vec2<f32>,
    // 0 = escape-time coloring, 1 = distance-estimation shading.
    de_coloring: u32,
};

const KIND_MANDELBROT: u32 = 0u;
const KIND_BURNING_SHIP: u32 = 1u;
const KIND_TRICORN: u32 = 2u;
const KIND_MULTIBROT: u32 = 3u;
const KIND_CELTIC: u32 = 4u;
const KIND_PERPENDICULAR: u32 = 5u;
const KIND_BUFFALO: u32 = 6u;
const KIND_PHOENIX: u32 = 7u;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> ref_orbit: array<vec2<f32>>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    // Position within the view, in [-0.5, 0.5] at the visible edges.
    @location(0) centered: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    var verts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let ndc = verts[idx];
    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    // Flip y so +imaginary points up the screen.
    out.centered = vec2<f32>(ndc.x, -ndc.y) * 0.5;
    return out;
}

// Complex multiply.
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex conjugate.
fn conj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

// |c + d| - |c|, evaluated exactly (no catastrophic cancellation even when the
// sum crosses zero). This is what makes the Burning Ship delta correct through
// the sign flips that happen all along the axes, where the ship's detail lives.
fn diffabs(c: f32, d: f32) -> f32 {
    let cd = c + d;
    if (c >= 0.0) {
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

// One perturbation step of the current fractal's delta: e -> f(Z+e) - f(Z),
// where `z` is the reference orbit value X_m. `step_add` (dc) is added by the
// caller. Must match `FractalKind` on the CPU side.
fn advance_delta(z: vec2<f32>, e: vec2<f32>) -> vec2<f32> {
    if (u.kind == KIND_BURNING_SHIP) {
        // (|x| + i|y|)^2 has real part x^2 - y^2 (an ordinary square delta) and
        // imaginary part 2|x y|. The imaginary delta is 2(|x y| - |X Y|); diffabs
        // computes it exactly, even where the product x y changes sign — which the
        // old sign(X)sign(Y) shortcut got wrong whenever the delta was large
        // enough to flip it (all the time at shallow zoom).
        let base = 2.0 * cmul(z, e) + cmul(e, e);
        let dp = z.x * e.y + z.y * e.x + e.x * e.y;
        return vec2<f32>(base.x, 2.0 * diffabs(z.x * z.y, dp));
    } else if (u.kind == KIND_TRICORN) {
        let cz = conj(z);
        let ce = conj(e);
        return 2.0 * cmul(cz, ce) + cmul(ce, ce);
    } else if (u.kind == KIND_MULTIBROT) {
        return multibrot_delta(z, e, clamp(u.power, 2u, 8u));
    } else if (u.kind == KIND_CELTIC) {
        // z^2 delta split: sq.x = delta of Re(z^2), sq.y = delta of Im(z^2).
        // Celtic abs the real output, so |Re(z^2)| delta = diffabs(Re(Z^2), sq.x).
        let sq = 2.0 * cmul(z, e) + cmul(e, e);
        return vec2<f32>(diffabs(z.x * z.x - z.y * z.y, sq.x), sq.y);
    } else if (u.kind == KIND_BUFFALO) {
        // Abs both outputs: real |Re(z^2)|, imag -|Im(z^2)| (Im(Z^2) = 2 X Y).
        let sq = 2.0 * cmul(z, e) + cmul(e, e);
        return vec2<f32>(diffabs(z.x * z.x - z.y * z.y, sq.x),
                         -diffabs(2.0 * z.x * z.y, sq.y));
    } else if (u.kind == KIND_PERPENDICULAR) {
        // real x^2 - y^2 (ordinary square delta), imag -2 x |y|.
        // d(-2 x |y|) = -2[ X·(|Y+ey|-|Y|) + ex·|Y+ey| ]; diffabs gives |Y+ey|-|Y|.
        let sq = 2.0 * cmul(z, e) + cmul(e, e);
        let da = diffabs(z.y, e.y);        // |Y + ey| - |Y|
        let abs_yf = abs(z.y) + da;        // |Y + ey|
        return vec2<f32>(sq.x, -2.0 * (z.x * da + e.x * abs_yf));
    }
    return 2.0 * cmul(z, e) + cmul(e, e); // Mandelbrot (and Phoenix square part)
}

// Derivative f'(Z) of the iteration map at the full value Z, used to propagate
// the orbit derivative for distance-estimation shading. Exact for the
// holomorphic kinds (z^2 -> 2Z, z^p -> p Z^{p-1}); for the non-holomorphic
// Burning Ship / Tricorn we use |f'| ~ |2Z|, which keeps the DE magnitude close
// enough to de-speckle filaments.
fn fprime(z: vec2<f32>) -> vec2<f32> {
    if (u.kind == KIND_MULTIBROT) {
        let p = clamp(u.power, 2u, 8u);
        var zk = vec2<f32>(1.0, 0.0); // Z^0
        for (var k: u32 = 1u; k < p; k = k + 1u) {
            zk = cmul(zk, z); // -> Z^{p-1}
        }
        return f32(p) * zk;
    }
    return 2.0 * z;
}

// Smooth cyclic palettes (Inigo Quilez cosine palettes), selected by id.
fn palette(id: u32, t: f32) -> vec3<f32> {
    if (id == 4u) {
        return vec3<f32>(t, t, t); // grayscale
    }
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    var c = vec3<f32>(1.0, 1.0, 1.0);
    var d = vec3<f32>(0.00, 0.10, 0.20); // 0: amber / blue
    if (id == 1u) {
        d = vec3<f32>(0.00, 0.33, 0.67); // rainbow
    } else if (id == 2u) {
        d = vec3<f32>(0.30, 0.20, 0.20); // warm ember
    } else if (id == 3u) {
        c = vec3<f32>(1.0, 1.0, 0.5);
        d = vec3<f32>(0.80, 0.90, 0.30); // lime / magenta
    }
    return a + b * cos(6.28318530718 * (c * t + d));
}

// Perturbation iterate + color a single sample. `offset` is the per-pixel
// offset in complex units. For Mandelbrot it is the c-plane offset added every
// step (delta starts at 0); for Julia it is the z-plane offset that seeds the
// initial delta (c is fixed, so nothing is added per step). Interior pixels
// return black.
fn shade(offset: vec2<f32>, px: f32) -> vec3<f32> {
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
    if (u.is_julia != 0u) {
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
        if (z2 > u.bailout_sq) {
            escaped = true;
            break;
        }
        if (n >= u.max_iter) {
            break; // interior
        }

        // Propagate the derivative of the full orbit (unaffected by rebasing,
        // which only re-expresses the same value). Only when DE is enabled.
        // Phoenix's two-term map adds p·dz_{n-1} and carries the previous dz.
        if (u.de_coloring != 0u) {
            var dz_new = cmul(fprime(z), dz) + dz_seed;
            if (u.kind == KIND_PHOENIX) {
                dz_new = dz_new + cmul(u.phoenix_p, dz_prev);
                dz_prev = dz;
            }
            dz = dz_new;
        }

        // Advance the delta by this fractal's formula (+ dc for the set plane).
        // Phoenix additionally adds p·e_{n-1} and carries the previous delta.
        let e_old = e;
        e = advance_delta(xm, e) + step_add;
        if (u.kind == KIND_PHOENIX) {
            e = e + cmul(u.phoenix_p, e_prev);
            e_prev = e_old;
        }
        m = m + 1u;
        n = n + 1u;

        // Keep the reference index valid and the delta small.
        if (m >= u.ref_len) {
            // Reference exhausted: any pixel that followed it this far has
            // effectively escaped (interior pixels rebase before reaching here).
            z = ref_orbit[u.ref_len - 1u] + e;
            escaped = true;
            break;
        }
        let y = ref_orbit[m] + e;
        if (dot(y, y) < dot(e, e)) {
            // Rebase to index 0: carry the full value as the new delta. Valid
            // because y_n = X[0] + (y_n - X[0]); for Mandelbrot X[0]=0.
            // Phoenix: after rebasing the implied previous reference is Y[-1]=0,
            // so the previous delta becomes the full previous value y_n (= z).
            if (u.kind == KIND_PHOENIX) {
                e_prev = z;
            }
            e = y - z0;
            m = 0u;
        }
    }

    if (!escaped) {
        return vec3<f32>(0.0, 0.0, 0.0); // interior of the set
    }

    let z2 = dot(z, z);

    // Continuous (smooth) iteration count.
    let log_zn = 0.5 * log(max(z2, 1.0));
    let nu = log2(log_zn / log(2.0));
    let smooth_i = f32(n) + 1.0 - nu;

    // sqrt compresses the huge iteration counts of deep zooms so the palette
    // varies smoothly instead of aliasing into speckle.
    let ci = sqrt(max(smooth_i, 0.0));
    let t = fract(ci * u.color_scale + u.color_offset);
    var col = palette(u.palette_id, t);

    if (u.de_coloring != 0u) {
        // Exterior distance estimate (complex-plane units): |z|·ln|z| / |dz|.
        // Divided by the pixel footprint it becomes a distance in pixels; we
        // darken toward the boundary (< ~1 px away) so filaments stay crisp
        // instead of aliasing into speckle. If |dz| overflowed, de -> 0 and the
        // boundary simply reads as dark, which is the correct limit.
        let zmag = sqrt(max(z2, 1.0));
        let dzmag = sqrt(max(dot(dz, dz), 1e-20));
        let de = zmag * log(zmag) / dzmag;
        let de_px = de / max(px, 1e-30);
        col = col * clamp(de_px, 0.0, 1.0);
    }
    return col;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let base = in.centered * u.span + u.dc_offset;

    // Screen-space complex-units-per-pixel. Derivatives must be evaluated in
    // uniform control flow, so take them here; used to place sub-pixel AA
    // samples and to convert the distance estimate into pixels.
    let dx = dpdx(base);
    let dy = dpdy(base);
    let px = length(abs(dx) + abs(dy)); // ~ complex units per pixel (footprint)

    let aa = max(u.aa_level, 1u);
    if (aa <= 1u) {
        return vec4<f32>(shade(base, px), 1.0);
    }

    var acc = vec3<f32>(0.0, 0.0, 0.0);
    let inv = 1.0 / f32(aa);
    for (var sy: u32 = 0u; sy < aa; sy = sy + 1u) {
        for (var sx: u32 = 0u; sx < aa; sx = sx + 1u) {
            // Sample centers evenly spread across the pixel, jitter in (-0.5, 0.5).
            let jx = (f32(sx) + 0.5) * inv - 0.5;
            let jy = (f32(sy) + 0.5) * inv - 0.5;
            acc = acc + shade(base + jx * dx + jy * dy, px);
        }
    }
    return vec4<f32>(acc / f32(aa * aa), 1.0);
}
