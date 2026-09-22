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

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var data_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> lights: array<Light, 16>;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(fullscreen_triangle_pos(idx), 0.0, 1.0);
}

fn shadow_fragment(pos: vec3<f32>) -> vec4<f32> {

    let x = i32(pos.x);
    let y = i32(pos.y);
    if textureLoad(data_tex, vec2<i32>(x, y), 0).b != 0. {
        return vec4<f32>(0.1, 0.1, 0.1, 1.0);
    } else {
        let h0 = textureLoad(data_tex, vec2<i32>(x, y), 0).g;
        let h1 = textureLoad(data_tex, vec2<i32>(x + 1, y), 0).g;
        let h2 = textureLoad(data_tex, vec2<i32>(x, y + 1), 0).g;
        let normal = normal_from_heights(h0, h1, h2);
        return vec4<f32>(shadow_color(normal), 1.0);
    }
}
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    if u.dimension3 == 0u {
        if u.shadow != 0u {
            return shadow_fragment(pos.xyz);
        } else {
            let d = textureLoad(data_tex, vec2<i32>(i32(pos.x), i32(pos.y)), 0);
            let ci = d.r;
            let de = d.g;
            let interior_frac = d.b;

            var col = classic_color(ci, de);
            // Anti-alias the set boundary: fade toward black by the fraction of the
            // pixel's sub-samples that landed in the interior.
            col = col * (1.0 - interior_frac);
            return vec4<f32>(col, 1.0);
        }
    } else {
        return ray_marching(pos);
    }
}

fn sdf(pos: vec3<f32>) -> f32 {
    let texture_pos = vec2<i32>(i32(pos.x), i32(pos.y));
    let px = textureLoad(data_tex, texture_pos, 0);
    let de = px.g;
    let z = pos.z;
    if px.b != 0. {
        return 0.;
    } else {
        return sqrt(z * z + (de * de));
    }
}

fn ray_marching(pos: vec4<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(data_tex));
    let in_texture = vec2<f32>(
        pos.x / 1980. * size.x,
        pos.y / 1080. * size.y,
    );

    var world_pos = u.camera_inv_proj * vec4<f32>(in_texture, pos.z - 10, 1.0);

    let ray_origin = world_pos.xyz;
    let ray_dir = u.camera_direction;

    var p = ray_origin;
    var i = 0u;
    var dist = 0.0;

    while i < 1000u {
        dist = sdf(p);
        if dist < 0.0001 {
            break;
        }
        p         += dist * ray_dir;
        i         += 1u;
    }

    if dist < 0.0001 {
        return shadow_fragment(p);
    }
    return vec4<f32>(1., 0., 0., 1.);
}
