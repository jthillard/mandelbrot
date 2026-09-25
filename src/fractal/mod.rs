//! GPU fractal rendering: wgpu pipeline, uniforms, reference orbit, and the
//! egui paint callback.

pub mod buddhabrot;
pub mod kind;
pub mod reference;
pub mod renderer;
pub mod share;

pub use buddhabrot::{BuddhabrotCallback, BuddhabrotRenderer, BuddhabrotUniforms};
pub use kind::FractalKind;
pub use reference::{compute_reference, compute_set_reference};
#[cfg(not(target_arch = "wasm32"))]
pub use renderer::PipelineKey;
#[cfg(target_arch = "wasm32")]
pub use renderer::encode_png_with_progress;
pub use renderer::{ExportRender, FractalCallback, FractalRenderer, MAX_REF_POINTS, Uniforms};
#[cfg(not(target_arch = "wasm32"))]
pub use renderer::{encode_png, export_to_png_blocking, render_readback_blocking};
pub use share::ShareState;
