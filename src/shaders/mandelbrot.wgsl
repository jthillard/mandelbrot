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
    dc_offset: vec2<f32>,
};

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
fn shade(offset: vec2<f32>) -> vec3<f32> {
    let z0 = ref_orbit[0]; // reference start (0 for Mandelbrot, center for Julia)

    var step_add = offset;
    var e = vec2<f32>(0.0, 0.0);
    if (u.is_julia != 0u) {
        step_add = vec2<f32>(0.0, 0.0);
        e = offset;
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

        // Advance the delta: e = 2*X_m*e + e^2 (+ dc for Mandelbrot).
        e = 2.0 * cmul(xm, e) + cmul(e, e) + step_add;
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
            e = y - z0;
            m = 0u;
        }
    }

    if (!escaped) {
        return vec3<f32>(0.0, 0.0, 0.0); // interior of the set
    }

    // Continuous (smooth) iteration count.
    let log_zn = 0.5 * log(max(dot(z, z), 1.0));
    let nu = log2(log_zn / log(2.0));
    let smooth_i = f32(n) + 1.0 - nu;

    // sqrt compresses the huge iteration counts of deep zooms so the palette
    // varies smoothly instead of aliasing into speckle.
    let ci = sqrt(max(smooth_i, 0.0));
    let t = fract(ci * u.color_scale + u.color_offset);
    return palette(u.palette_id, t);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let base = in.centered * u.span + u.dc_offset;

    let aa = max(u.aa_level, 1u);
    if (aa <= 1u) {
        return vec4<f32>(shade(base), 1.0);
    }

    // Screen-space complex-units-per-pixel, used to place sub-pixel samples.
    // Derivatives must be evaluated in uniform control flow, so take them here.
    let dx = dpdx(base);
    let dy = dpdy(base);

    var acc = vec3<f32>(0.0, 0.0, 0.0);
    let inv = 1.0 / f32(aa);
    for (var sy: u32 = 0u; sy < aa; sy = sy + 1u) {
        for (var sx: u32 = 0u; sx < aa; sx = sx + 1u) {
            // Sample centers evenly spread across the pixel, jitter in (-0.5, 0.5).
            let jx = (f32(sx) + 0.5) * inv - 0.5;
            let jy = (f32(sy) + 0.5) * inv - 0.5;
            acc = acc + shade(base + jx * dx + jy * dy);
        }
    }
    return vec4<f32>(acc / f32(aa * aa), 1.0);
}
