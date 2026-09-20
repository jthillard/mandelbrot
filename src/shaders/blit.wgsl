// Passthrough blit: sample the cached fractal texture and write it to egui's
// target. Kept separate from the fractal shader so the expensive per-pixel
// iteration runs only when the cache is (re)rendered, while every egui frame
// pays just this cheap textured fullscreen triangle.

@group(0) @binding(0) var cache_tex: texture_2d<f32>;
@group(0) @binding(1) var cache_samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    let p = fullscreen_triangle_pos(idx);
    var out: VsOut;
    out.pos = vec4<f32>(p, 0.0, 1.0);
    // Map NDC to texture UV. v is flipped so the cache's top row (rendered at
    // NDC y = +1) shows at the top of the screen.
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(cache_tex, cache_samp, in.uv, 0.0);
}
