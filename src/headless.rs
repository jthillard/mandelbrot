// Headless PNG rendering: parse the CLI, build the exact same view/state the
// windowed app would from it, then render straight to a file. No window, no
// event loop, no worker-thread debounce (nothing to debounce for a one-shot
// render); it just creates its own wgpu device, computes the reference orbit
// once, and renders through the same `ExportRender` path the "Export PNG"
// button uses.

use eframe::egui_wgpu::wgpu;

use crate::app::{FractalApp, unix_timestamp};
use crate::cli::Cli;
use crate::fractal::{ExportRender, FractalRenderer, export_to_png_blocking};

/// Cap on the output image dimension (px), to stay within GPU texture limits.
const MAX_DIM: u32 = 8192 * 16;

pub fn run(cli: Cli) -> Result<(), String> {
    if cli.buddhabrot {
        return Err("headless mode doesn't support --buddhabrot yet".into());
    }

    let width = cli.width.clamp(16, MAX_DIM);
    let height = cli.height.clamp(16, MAX_DIM);
    let export_path = cli
        .export_path
        .clone()
        .unwrap_or_else(|| format!("fractal-{}.png", unix_timestamp()));

    let mut app = FractalApp::default_state();
    app.apply_cli(cli);

    eprintln!("computing reference orbit…");
    app.compute_reference_blocking();

    let (device, queue) = pollster::block_on(request_device())?;
    let format = wgpu::TextureFormat::Bgra8Unorm;
    let renderer = FractalRenderer::new(&device, format);
    let (pipeline, bind_group_layout, format) = renderer.export_handles();

    let uniforms = app.make_uniforms(width as f64 / height as f64);
    let er = ExportRender::new(
        &device,
        &queue,
        pipeline,
        &bind_group_layout,
        format,
        width,
        height,
        uniforms,
        app.reference_points(),
    );

    eprintln!("rendering {width}×{height}…");
    let png = export_to_png_blocking(&device, &queue, &er, |phase, fraction| {
        eprint!("\r{phase} {:>3.0}%", fraction * 100.0);
    });
    eprintln!();

    std::fs::write(&export_path, &png).map_err(|e| format!("save failed: {e}"))?;
    println!("saved {export_path} ({width}×{height})");
    Ok(())
}

/// Set up a wgpu device with no surface/window attached, matching the limits
/// `main::wgpu_options` requests for the windowed app (the fractal fragment
/// shader needs storage buffers, which downlevel/WebGL-style limits disallow).
async fn request_device() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(|e| format!("no compatible GPU adapter: {e}"))?;
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("headless fractal device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .map_err(|e| format!("failed to create device: {e}"))
}
