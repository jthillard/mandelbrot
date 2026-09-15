// Colourise pass: map the iteration pass's per-pixel escape data (from
// `mandelbrot.wgsl`'s `fs_data`) through the palette. This is the only
// color-dependent step, so changing the palette / colour scale / offset (e.g.
// colour cycling) re-runs just this cheap pass — the expensive perturbation
// iteration in the data texture is reused untouched.
//
// The data texture holds, per texel: R = ci (palette parameter), G = DE
// darkening factor, B = interior fraction (for boundary anti-aliasing). It is
// the same resolution as this pass's target, so we read it with `textureLoad`
// at the fragment's integer pixel coordinate (nearest — iteration data must not
// be linearly filtered across escape boundaries).

// Must match `Uniforms` in mandelbrot.wgsl / the Rust `Uniforms` struct.
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
    kind: u32,
    power: u32,
    dc_offset: vec2<f32>,
    phoenix_p: vec2<f32>,
    de_coloring: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var data_tex: texture_2d<f32>;

// Smooth cyclic palettes (Inigo Quilez cosine palettes). Must match the palette
// in mandelbrot.wgsl.
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

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var verts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(verts[idx], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let d = textureLoad(data_tex, vec2<i32>(i32(pos.x), i32(pos.y)), 0);
    let ci = d.r;
    let de = d.g;
    let interior_frac = d.b;

    let t = fract(ci * u.color_scale + u.color_offset);
    var col = palette(u.palette_id, t) * de;
    // Anti-alias the set boundary: fade toward black by the fraction of the
    // pixel's sub-samples that landed in the interior.
    col = col * (1.0 - interior_frac);
    return vec4<f32>(col, 1.0);
}
