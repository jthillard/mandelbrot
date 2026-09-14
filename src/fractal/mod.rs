//! GPU fractal rendering: wgpu pipeline, uniforms, reference orbit, and the
//! egui paint callback.

pub mod reference;
pub mod renderer;
pub mod share;

pub use reference::{compute_mandelbrot_reference, compute_reference};
pub use renderer::{FractalCallback, FractalRenderer, Uniforms, MAX_REF_POINTS, encode_png};
pub use share::ShareState;
