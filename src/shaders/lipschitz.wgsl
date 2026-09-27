// Distance field for the shadow and 3D views of kinds whose map is
// discontinuous (Complex Multibrot's principal-branch z^p). The per-pixel
// DE, |z|·ln|z|/|dz| = G/|G'|, is only a distance when the potential G is
// continuous. Across a branch-cut preimage G jumps (|z| jumps by
// e^(2π·Im p)), each side's DE describes the set as continued on its own
// branch: seams in shadow mode, walls in the raymarched terrain. A true
// distance to the set is continuous, so this rebuilds one from the pixels that are unambiguously
// next to the set ("seeds": interior texels, and exterior texels whose DE is
// below SEED_DE, i.e. within about a pixel of it):
//     h(x) = min over seeds y of  DE(y) + DIST_SLOPE·|x - y|
// computed by jump flooding: `fs_seed` marks seeds, `fs_jump` runs with
// `step` halving from about half the texture size down to 1 (plus one more
// step-1 pass), each texel keeping the best seed among itself and its 8
// neighbours at ±step, and `fs_compose` writes the result, eased by
// `ease_distance`, as the data texture's G channel. Seeds are carried as coordinates, so the distance is
// Euclidean rather than an 8-direction chamfer approximation.
//
// The set only counts where it's in view: next to the image edge, set just
// outside it is missed and the height there reads a bit high.
//
// Texel layout of the seed textures: (seed x, seed y, seed DE, 1), or all 0
// for "no seed yet". R (palette parameter) and B (interior fraction) of the
// data texture pass through untouched.

struct LipschitzStep {
    step: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
};

@group(0) @binding(0) var data_tex: texture_2d<f32>;
@group(0) @binding(1) var seed_tex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> ls: LipschitzStep;

// DE units per texel of a true distance: DE is in units of `pixel_size`,
// the texel's (|dx| + |dy|) footprint, i.e. sqrt(2) texels.
const DIST_SLOPE: f32 = 0.70710678;
// Exterior texels with a DE below this (in DE units) are seeds.
const SEED_DE: f32 = 1.0;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(fullscreen_triangle_pos(idx), 0.0, 1.0);
}

@fragment
fn fs_seed(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let d = textureLoad(data_tex, p, 0);
    if d.b != 0.0 {
        return vec4<f32>(vec2<f32>(p), 0.0, 1.0);
    }
    if d.g < SEED_DE {
        return vec4<f32>(vec2<f32>(p), d.g, 1.0);
    }
    return vec4<f32>(0.0);
}

// Height at `p` through seed `s` (a seed texel), or +inf for "no seed".
fn through(p: vec2<i32>, s: vec4<f32>) -> f32 {
    if s.a == 0.0 {
        return 3.0e38;
    }
    return s.z + DIST_SLOPE * distance(vec2<f32>(p), s.xy);
}

@fragment
fn fs_jump(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let hi = vec2<i32>(textureDimensions(seed_tex)) - vec2<i32>(1, 1);
    let st = ls.step;
    var best = textureLoad(seed_tex, p, 0);
    var best_h = through(p, best);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let q = p + st * vec2<i32>(dx, dy);
            if any(q < vec2<i32>(0, 0)) || any(q > hi) {
                continue;
            }
            let s = textureLoad(seed_tex, q, 0);
            let h = through(p, s);
            if h < best_h {
                best = s;
                best_h = h;
            }
        }
    }
    return best;
}

// Scale of `ease_distance`, in screen heights.
const EASE_K: f32 = 0.02;

// Concave easing of the distance field `h` (DE units, `size_y` texels
// tall): K·ln(1 + x/K) in screen heights, the unit `sdf` in colorize.wgsl
// reads. Slope 1 at the set, flattening far from it, so the terrain rises
// steeply out of the set and then levels off. The slope never exceeds 1,
// so the field stays a valid (conservative) distance for the sphere tracer.
fn ease_distance(h: f32, size_y: f32) -> f32 {
    return EASE_K * log(1.0 + h / (size_y * EASE_K)) * size_y;
}

@fragment
fn fs_compose(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let d = textureLoad(data_tex, p, 0);
    if d.b != 0.0 {
        return d;
    }
    // Not min(h, DE): that keeps the too-low side of every jump, so the
    // walls come back as shards.
    let h = through(p, textureLoad(seed_tex, p, 0));
    if h >= 3.0e38 {
        return d; // no set anywhere in view: keep the DE
    }
    return vec4<f32>(d.r, ease_distance(h, f32(textureDimensions(data_tex).y)), d.b, d.a);
}
