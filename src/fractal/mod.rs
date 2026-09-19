//! GPU fractal rendering: wgpu pipeline, uniforms, reference orbit, and the
//! egui paint callback.

pub mod buddhabrot;
pub mod reference;
pub mod renderer;
pub mod share;

pub use buddhabrot::{BuddhabrotCallback, BuddhabrotRenderer, BuddhabrotUniforms};
pub use reference::{FractalKind, compute_reference, compute_set_reference};
#[cfg(target_arch = "wasm32")]
pub use renderer::encode_png_with_progress;
#[cfg(not(target_arch = "wasm32"))]
pub use renderer::export_to_png_blocking;
pub use renderer::{ExportRender, FractalCallback, FractalRenderer, MAX_REF_POINTS, Uniforms};
pub use share::ShareState;
