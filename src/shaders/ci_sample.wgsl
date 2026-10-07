// Auto colour scale without compute shaders (WebGL2): point-samples the data
// texture onto a small CI_SAMPLE_DIM² grid, read back and binned on the CPU
// like `ci_stats.wgsl` (`CiHistogram::Readback` in renderer.rs). The grid
// stretches to the source's aspect; percentiles don't care.

@group(0) @binding(0) var data_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(fullscreen_triangle_pos(idx), 0.0, 1.0);
}

// Must match `CI_SAMPLE_DIM` in renderer.rs.
const CI_SAMPLE_DIM: f32 = 256.0;

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(data_tex));
    let p = min(vec2<i32>(floor(pos.xy * size / CI_SAMPLE_DIM)), vec2<i32>(size) - 1);
    let d = textureLoad(data_tex, p, 0);
    return vec4<f32>(d.r, d.b, 0.0, 0.0);
}
