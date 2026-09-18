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
    shadow_palette_id: u32,
    aa_level: u32,
    kind: u32,
    power: u32,
    dc_offset: vec2<f32>,
    phoenix_p: vec2<f32>,
    lambda_l: vec2<f32>,
    de_coloring: u32,
    shadow: u32,
};

struct Light {
    azimuth: f32,
    altitude: f32,
    color: u32,
    _pad: u32
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var data_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> lights: array<Light, 16>;

// Smooth cyclic palettes (Inigo Quilez cosine palettes). Must match the palette
// in mandelbrot.wgsl.
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

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var verts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(verts[idx], 0.0, 1.0);
}

fn load(x: i32, y: i32) -> vec3<f32> {
    let dist = textureLoad(data_tex, vec2<i32>(x, y), 0).g;
    return vec3<f32>(f32(x), f32(y), dist);
}

fn compute_light(normal: vec3<f32>, light: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(max(0., dot(normal, normalize(light))));
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

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    if u.shadow != 0u {
        if textureLoad(data_tex, vec2<i32>(i32(pos.x), i32(pos.y)), 0).b != 0. {
            return vec4<f32>(0.1, 0.1, 0.1, 1.0);
        } else {
            let d = array<vec3<f32>, 3>(load(i32(pos.x), i32(pos.y)), load(i32(pos.x + 1), i32(pos.y)), load(i32(pos.x), i32(pos.y + 1)));

            let normal = normalize(cross(d[1] - d[0], d[2] - d[0]));

            var color: vec3<f32>;
            if u.shadow_palette_id == 0u {
                color = compute_light(normal,vec3<f32>(.5, .5, .5)) + vec3<f32>(0.58, 0.85, 1.) * 0.2;

                color = filmic(color, 2.5);
                color = contrast(color, 4., 0.67);
            } else if u.shadow_palette_id == 1u {
                color = compute_light(normal, vec3<f32>(0., .5, .5)) * vec3<f32>(1., 0.5, 0.5) + compute_light(normal, vec3<f32>(0.5, 0., .5)) * vec3<f32>(0.5, 1., 1.);

                color = filmic(color, 4.2);
            } else {
                color = vec3<f32>(0);
                var light_count = 0;
                for (var i = 0u ; i < 16; i++) {
                    let light_color = unpack4x8unorm(lights[i].color);
                    if any(light_color != vec4<f32>(0)) {
                        light_count       += 1;
                    }

                    color       += compute_light(normal, vec3<f32>(
                        cos(lights[i].azimuth) * cos(lights[i].altitude),
                        sin(lights[i].azimuth) * cos(lights[i].altitude),
                        sin(lights[i].altitude))) * light_color.xyz * light_color.a;
                }

                color = filmic(color, 1. + f32(light_count));
            }

            return vec4<f32>(color, 1.0);
        }
    } else {
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
}
