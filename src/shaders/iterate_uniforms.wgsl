// Shared by mandelbrot.wgsl (writes the per-pixel data texture) and
// colorize.wgsl (reads it): the iteration pass and the colour remap pass
// must agree on both the uniform layout and the palette function.

// Must match the Rust `Uniforms` struct in renderer.rs field-for-field,
// including padding.
struct Uniforms {
    span: vec2<f32>,
    max_iter: u32,
    ref_len: u32,
    color_offset: f32,
    color_scale: f32,
    bailout_sq: f32,
    is_julia: u32,
    palette_id: u32,
    shadow_palette_id: u32,
    aa_level: u32,
    // Iteration formula (see the KIND_* constants in common.wgsl).
    kind: u32,
    // Exponent for the Multibrot kind.
    power: u32,
    // Kind-switch morph: the kind blended *from* (see morph_w).
    morph_from: u32,
    dc_offset: vec2<f32>,
    // Distortion constant p for the Phoenix map (z^2 + c + p*z_{n-1}); unused
    // by other kinds. Placed by dc_offset so both vec2s stay 8-byte aligned.
    phoenix_p: vec2<f32>,
    // Distortion constant l for the Lambda map (l*z(1 - z_{n-1})); unused
    // by other kinds.
    lambda_l: vec2<f32>,
    // Complex exponent for the Complex Multibrot kind (z^power + c); unused
    // by other kinds.
    complex_power: vec2<f32>,
    // 0 = escape-time coloring, 1 = distance-estimation shading.
    de_coloring: u32,
    // 0 = classic colors, 1 = shadows, 2 = 3D raymarching rendering
    shadow: u32,
    // camera direction vector
    camera_direction: vec3<f32>,
    // Number of live entries at the start of `lights` (fills the vec3's tail
    // padding slot).
    light_count: u32,
    // inverse of the camera's view-projection matrix, for reconstructing a
    // world-space ray origin per pixel in the raymarcher
    camera_inv_proj: mat4x4<f32>,
    // Screen dimensions
    screen_dim: vec2<f32>,
    // Kind-switch morph weight: each step is (1 - w)*f_kind + w*f_morph_from;
    // 0 = no morph. Only read by MORPH pipelines (see mandelbrot.wgsl).
    morph_w: f32,
    // Complex binomial coefficients C(complex_power, k) for k = 1..16, two per
    // vec4 (k odd in .xy, k even in .zw), for the Complex Multibrot delta
    // series. Precomputed on the CPU since they only depend on the power.
    cm_coef: array<vec4<f32>, 8>,
};

// Smooth cyclic palettes (Inigo Quilez cosine palettes), selected by id.
fn palette(id: u32, t: f32) -> vec3<f32> {
    if id == 4u {
        return vec3<f32>(t, t, t); // grayscale
    }
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    var c = vec3<f32>(1.0, 1.0, 1.0);
    var d = vec3<f32>(0.00, 0.10, 0.20); // 0: amber / blue
    if id == 1u {
        d = vec3<f32>(0.00, 0.33, 0.67); // rainbow
    } else if id == 2u {
        d = vec3<f32>(0.30, 0.20, 0.20); // warm ember
    } else if id == 3u {
        c = vec3<f32>(1.0, 1.0, 0.5);
        d = vec3<f32>(0.80, 0.90, 0.30); // lime / magenta
    }
    return a + b * cos(6.28318530718 * (c * t + d));
}

// Classic (non-shadow) escape colouring: palette lookup at the smoothed
// iteration count `ci`, darkened by the distance-estimate factor `de`
// (sqrt-compressed so the darkening falls off more gently near the
// boundary). Shared by the colourise pass's classic branch (colorize.wgsl,
// applied to an already-averaged data texel) and the PNG-export pass
// (mandelbrot.wgsl's `fs_color`, applied per sub-sample pre-AA) — the two
// places a fully escaped point is turned into a final pixel colour.
fn classic_color(ci: f32, de: f32) -> vec3<f32> {
    let t = fract(ci * u.color_scale + u.color_offset);
    return palette(u.palette_id, t) * sqrt(de);
}

// A single directional light, built on the CPU from the UI's light list
// (`GpuLight` in lights.rs): `dir` is the unit direction toward the light
// (precomputed from azimuth/altitude so the shader does no trig), `color` a
// packed RGBA8 whose alpha doubles as intensity. Only the first
// `u.light_count` entries are live, all with a non-zero colour. Each shader
// that binds a `lights: array<Light, 16>` uniform (colorize.wgsl,
// mandelbrot.wgsl's export shadow path) uses this same layout.
struct Light {
    dir: vec3<f32>,
    color: u32,
};

// Lambertian term for a unit `light` direction.
fn compute_light(normal: vec3<f32>, light: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(max(0., dot(normal, light)));
}

fn uncharted2tonemap(x: vec3<f32>) -> vec3<f32> {
    let A = 0.15; // Shoulder strength
    let B = 0.50; // Linear strength
    let C = 0.10; // Linear angle
    let D = 0.20; // Toe strength
    let E = 0.02; // Toe numerator / shoarder angle/etc.
    let F = 0.30; // Toe denominator

    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn filmic(color: vec3<f32>, white_point: f32) -> vec3<f32> {
    let exposure_bias = 2.0;
    let curr = uncharted2tonemap(color * exposure_bias);

    // Valeur blanche maximale de référence
    let white_scale = vec3(1.0) / uncharted2tonemap(vec3(white_point));
    return curr * white_scale;
}

fn s(color: vec3<f32>, k: f32, c: f32) -> vec3<f32> {
    return 1. / (1. + exp(-k * (color - c)));
}

fn contrast(color: vec3<f32>, k: f32, c: f32) -> vec3<f32> {
    let color_c = s(color, k, c);

    return (color_c - s(vec3<f32>(0), k, c)) / (s(vec3<f32>(1), k, c) - s(vec3<f32>(0), k, c));
}

// Surface normal from three height samples (`h0` at the pixel, `h1` one pixel
// to the right, `h2` one pixel down), treating DE as a height field. Only the
// differences matter, so callers don't need to pass pixel coordinates — a
// texture-backed caller (colorize.wgsl) and a live-sampled caller
// (mandelbrot.wgsl's export shadow path) can share this.
fn normal_from_heights(h0: f32, h1: f32, h2: f32) -> vec3<f32> {
    let d0 = vec3<f32>(0.0, 0.0, h0);
    let d1 = vec3<f32>(1.0, 0.0, h1);
    let d2 = vec3<f32>(0.0, 1.0, h2);
    return normalize(cross(d1 - d0, d2 - d0));
}

// Shade a DE-derived surface normal per `u.shadow_palette_id`: 0 = grayscale
// key light, 1 = red/blue two-tone, 2 = the user's custom `lights` list,
// 3 = the classic escape-time palette at `ci` (the smoothed iteration count),
// lit by the grayscale key light.
// Shared by the interactive shadow pass (colorize.wgsl) and the PNG-export
// shadow path (mandelbrot.wgsl's `fs_color`), which must render identically.
fn shadow_color(normal: vec3<f32>, ci: f32) -> vec3<f32> {
    var color: vec3<f32>;
    if u.shadow_palette_id == 0u {
        color = compute_light(normal, vec3<f32>(0.57735027, 0.57735027, 0.57735027)) + vec3<f32>(0.58, 0.85, 1.) * 0.2;

        color = filmic(color, 2.5);
        color = contrast(color, 4., 0.67);
    } else if u.shadow_palette_id == 1u {
        color = compute_light(normal, vec3<f32>(0., 0.70710678, 0.70710678)) * vec3<f32>(1., 0.5, 0.5) + compute_light(normal, vec3<f32>(0.70710678, 0., 0.70710678)) * vec3<f32>(0.5, 1., 1.);

        color = filmic(color, 4.2);
    } else if u.shadow_palette_id == 3u {
        // No DE darkening as in `classic_color`: in shadow/3D modes the DE
        // is a height (clamped to 1000, not 1), and the lighting already
        // shows the relief.
        let t = fract(ci * u.color_scale + u.color_offset);
        let ambient = 0.25;
        let light = compute_light(normal, vec3<f32>(0.57735027, 0.57735027, 0.57735027));
        color = palette(u.palette_id, t) * (ambient + (1.0 - ambient) * light);
    } else {
        color = vec3<f32>(0);
        let light_count = min(u.light_count, 16u);
        for (var i = 0u; i < light_count; i++) {
            let light_color = unpack4x8unorm(lights[i].color);
            color += compute_light(normal, lights[i].dir) * light_color.xyz * light_color.a;
        }

        color = filmic(color, 1. + f32(light_count));
    }
    return color;
}

// Colour of an interior (non-escaped) pixel in shadow/3D modes: black under
// the classic palette, like classic 2D mode, otherwise a dark gray plateau.
fn shadow_interior_color() -> vec3<f32> {
    if u.shadow_palette_id == 3u {
        return vec3<f32>(0.0);
    }
    return vec3<f32>(0.1);
}
