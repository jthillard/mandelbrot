//! GPU fractal rendering: wgpu pipeline, uniforms, reference orbit, and the
//! egui paint callback.

pub mod buddhabrot;
pub mod reference;
pub mod renderer;
pub mod share;

pub use buddhabrot::{BuddhabrotCallback, BuddhabrotRenderer, BuddhabrotUniforms};
pub use reference::{FractalKind, compute_reference, compute_set_reference};
pub use renderer::{
    ExportRender, FractalCallback, FractalRenderer, MAX_REF_POINTS, Uniforms,
    encode_png_with_progress,
};
pub use share::ShareState;
