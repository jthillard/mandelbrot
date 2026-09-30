// Histogram of the escaped pixels' smoothed iteration count `ci`, for the
// auto colour scale (`CiStats` in renderer.rs). Reads the same data texture
// the colourise pass reads (R = ci, B = interior fraction). Bins are uniform
// in log2(1 + ci) over [0, CI_LOG2_MAX), so one fixed layout covers every
// zoom depth (`MAX_ITERATIONS` = 2^24). Must match `CI_BINS`/`CI_LOG2_MAX`
// in renderer.rs.

const CI_BINS: u32 = 1024u;
const CI_LOG2_MAX: f32 = 25.0;

@group(0) @binding(0) var data_tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>, 1024>;

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(data_tex);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let d = textureLoad(data_tex, vec2<i32>(id.xy), 0);
    if d.b >= 1.0 {
        return; // fully interior: no escape time
    }
    let x = log2(1.0 + max(d.r, 0.0)) * (f32(CI_BINS) / CI_LOG2_MAX);
    let bin = min(u32(max(x, 0.0)), CI_BINS - 1u);
    atomicAdd(&hist[bin], 1u);
}
