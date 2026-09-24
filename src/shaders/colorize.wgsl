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

fn shadow_fragment(pos: vec2<f32>) -> vec4<f32> {
    let x = i32(pos.x);
    let y = i32(pos.y);
    let size = textureDimensions(data_tex);
    let here = textureLoad(data_tex, vec2<i32>(x, y), 0);
    if here.b != 0. {
        return vec4<f32>(0.1, 0.1, 0.1, 1.0);
    }
    // Forward differences, except on the last column/row where x+1 / y+1
    // is off the texture: fall back to a backward difference, mirrored
    // (h0 + (h0 - h[-1])) so the slope keeps the sign normal_from_heights
    // expects — plugging h[-1] in directly would flip the normal there.
    let h0 = here.g;
    var h1: f32;
    if x + 1 < i32(size.x) {
        h1 = textureLoad(data_tex, vec2<i32>(x + 1, y), 0).g;
    } else {
        h1 = 2.0 * h0 - textureLoad(data_tex, vec2<i32>(x - 1, y), 0).g;
    }
    var h2: f32;
    if y + 1 < i32(size.y) {
        h2 = textureLoad(data_tex, vec2<i32>(x, y + 1), 0).g;
    } else {
        h2 = 2.0 * h0 - textureLoad(data_tex, vec2<i32>(x, y - 1), 0).g;
    }
    let normal = normal_from_heights(h0, h1, h2);
    return vec4<f32>(shadow_color(normal), 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    if u.shadow == 2u {
        return ray_marching(pos);
    } else if u.shadow == 1u {
        return shadow_fragment(pos.xy);
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
}

// Per-frame constants of the raymarch, computed once per pixel in
// `ray_marching` rather than on each of the up-to-100 `sdf` steps.
struct MarchConsts {
    size: vec2<f32>,
    // (size.x / aspect_ratio, size.y): world xy -> texel scale.
    to_texel: vec2<f32>,
    size_i: vec2<i32>,
    inv_size_y: f32,
};

fn sdf(pos: vec3<f32>, k: MarchConsts) -> f32 {
    let texture_pos_f32 = pos.xy * k.to_texel;
    let texture_pos = clamp(vec2<i32>(texture_pos_f32), vec2<i32>(0, 0), k.size_i - vec2<i32>(1, 1));

    let to_texture = max(-min(texture_pos_f32, vec2(0.)), max(texture_pos_f32 - k.size, vec2(0.)));
    let dist_to_texture = length(to_texture) * k.inv_size_y;

    let px = textureLoad(data_tex, texture_pos, 0);
    let de = (px.g * k.inv_size_y) * 0.5;
    // Height is measured toward -z, the side the camera sits on (it looks
    // along +z), so the terrain is solid on +z: interior plateau at z = 0,
    // exterior sloping away from the camera as `de` grows.
    let signed_z = -pos.z;
    let z = max(signed_z, 0.);
    var d: f32;
    if px.b != 0. {
        d = z;
    } else {
        d = min(sqrt(z * z + de * de), signed_z + 1. - exp(-de * 5.));
    }
    // Outside the texture footprint, `d` is the distance from the clamped
    // point q on the footprint's edge. The terrain lies over the (convex)
    // footprint, so |p - x|² ≥ |q - x|² + |p - q|² for every terrain point x:
    // combine in quadrature (not by adding, which overshoots). p can't be in
    // the solid out here, so a negative `d` counts as 0.
    if dist_to_texture > 0. {
        let d_pos = max(d, 0.);
        return sqrt(d_pos * d_pos + dist_to_texture * dist_to_texture);
    }
    return d;
}

fn ray_marching(pos: vec4<f32>) -> vec4<f32> {
    let size_i = vec2<i32>(textureDimensions(data_tex));
    let size = vec2<f32>(size_i);
    let aspect_ratio = u.screen_dim.x / u.screen_dim.y;
    let k = MarchConsts(size, vec2<f32>(size.x / aspect_ratio, size.y), size_i, 1.0 / size.y);

    let in_texture = vec2<f32>(
        (pos.x / size.x) * 2. - 1.,
        (pos.y / size.y) * 2. - 1.,
    );

    var world_pos = u.camera_inv_proj * vec4<f32>(in_texture, 0., 1.0);

    let ray_origin = world_pos.xyz;
    let ray_dir = u.camera_direction;

    let z_intersect = ray_origin.z / ray_dir.z;
    var p = ray_origin - ray_dir * z_intersect;
    var i = 0u;
    var dist = 0.0;

    let dist_threshold = 0.000001;
    while i < 100u {
        let from_origin = p - ray_origin;
        if dot(from_origin, from_origin) > 9. {
            break;
        }
        dist = sdf(p, k);
        if dist < dist_threshold {
            break;
        }
        p += dist * ray_dir;
        i += 1u;
    }

    if dist < dist_threshold {
        return shadow_fragment(p.xy * k.to_texel);
    }
    return vec4<f32>(1., 0., 0., 1.);
}
