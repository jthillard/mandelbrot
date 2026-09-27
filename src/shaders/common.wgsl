// Shared helpers, concatenated into every shader at build time via
// `concat!`/`include_str!` (see renderer.rs / buddhabrot.rs). Keep this file
// free of anything that differs between pipelines (e.g. a `Uniforms` struct —
// mandelbrot/colorize and buddhabrot each have their own shape) since every
// shader gets the whole thing spliced in.

// Fullscreen triangle vertex position: one triangle that covers the whole
// viewport (cheaper than a quad's two), shared by every full-screen vertex
// shader in this project.
fn fullscreen_triangle_pos(idx: u32) -> vec2<f32> {
    var verts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return verts[idx];
}

// Complex multiply.
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// z^p for a complex exponent p, via the principal branch z^p = exp(p * ln z),
// ln z = ln|z| + i*arg(z). z = 0 maps to 0 (the correct limit for the
// Re(p) > 0 region the UI exposes; ln(0) would otherwise be -inf).
fn cpow(z: vec2<f32>, p: vec2<f32>) -> vec2<f32> {
    let r2 = dot(z, z);
    if r2 < 1e-30 {
        return vec2<f32>(0.0, 0.0);
    }
    let ln_r = 0.5 * log(r2);
    let theta = atan2(z.y, z.x);
    let mag = exp(p.x * ln_r - p.y * theta);
    let ang = p.x * theta + p.y * ln_r;
    return mag * vec2<f32>(cos(ang), sin(ang));
}

// Iteration formula selector, shared by the perturbation (mandelbrot.wgsl)
// and direct (buddhabrot.wgsl) iteration paths. Must match `FractalKind` in
// reference.rs.
const KIND_MANDELBROT: u32 = 0u;
const KIND_BURNING_SHIP: u32 = 1u;
const KIND_TRICORN: u32 = 2u;
const KIND_MULTIBROT: u32 = 3u;
// Highest Multibrot power (the UI/CLI/share-link clamp in app.rs matches).
// `bailout_sq` in app.rs shrinks the bailout with the power so z^p stays a
// finite f32.
const MULTIBROT_MAX_POWER: u32 = 20u;
const KIND_CELTIC: u32 = 4u;
const KIND_PERPENDICULAR: u32 = 5u;
const KIND_BUFFALO: u32 = 6u;
const KIND_PHOENIX: u32 = 7u;
const KIND_LAMBDA: u32 = 8u;
const KIND_COMPLEX_MULTIBROT: u32 = 9u;
