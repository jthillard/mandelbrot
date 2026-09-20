// Headless PNG rendering: parse the CLI, build the exact same view/state the
// windowed app would from it, then render straight to a file. No window, no
// event loop, no worker-thread debounce (nothing to debounce for a one-shot
// render); it just creates its own wgpu device, computes the reference orbit
// once, and renders through the same `ExportRender` path the "Export PNG"
// button uses.

use eframe::egui_wgpu::wgpu;

use crate::app::{FractalApp, unix_timestamp};
use crate::cli::Cli;
use crate::fractal::{ExportRender, FractalRenderer, ShareState, export_to_png_blocking};
use crate::view::{
    ViewState, big_from_decimal_str, interpolate_view, parse_view_spec, precision_for,
};

/// Cap on the output image dimension (px), to stay within GPU texture limits.
const MAX_DIM: u32 = 8192 * 16;

pub fn run(cli: Cli) -> Result<(), String> {
    if cli.buddhabrot {
        return Err("headless mode doesn't support --buddhabrot yet".into());
    }

    let width = cli.width.clamp(16, MAX_DIM);
    let height = cli.height.clamp(16, MAX_DIM);

    // These drive the animation path below; grab them before `apply_cli`
    // consumes `cli` to build the start state.
    let to_view = cli.to_view.clone();
    let to_share = cli.to_share.clone();
    let frames_arg = cli.frames;
    let fps = cli.fps;
    let duration = cli.duration;
    let linear = cli.linear;
    let export_path = cli.export_path.clone();

    let mut app = FractalApp::default_state();
    app.apply_cli(cli);

    if to_view.is_some() || to_share.is_some() {
        return run_animation(
            app,
            to_view,
            to_share,
            frames_arg,
            fps,
            duration,
            linear,
            width,
            height,
            export_path,
        );
    }

    let export_path = export_path.unwrap_or_else(|| format!("fractal-{}.png", unix_timestamp()));

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
        app.lights(),
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

/// Render a sequence of frames sweeping the camera from the app's current
/// (start) view to an end view, for feeding into ffmpeg. Everything other
/// than the view (kind, colors, iteration cap policy, ...) stays fixed at
/// whatever `apply_cli` set up for the start; only the camera moves.
#[allow(clippy::too_many_arguments)]
fn run_animation(
    mut app: FractalApp,
    to_view: Option<String>,
    to_share: Option<String>,
    frames_arg: Option<u32>,
    fps: f64,
    duration: Option<f64>,
    linear: bool,
    width: u32,
    height: u32,
    export_path: Option<String>,
) -> Result<(), String> {
    let frames = match frames_arg {
        Some(n) => n,
        None => {
            let dur = duration.ok_or("animation needs --frames, or --duration (with --fps)")?;
            ((fps * dur).round() as u32).max(2)
        }
    };
    if frames < 2 {
        return Err("animation needs at least 2 frames".into());
    }

    let to = parse_animation_target(to_view.as_deref(), to_share.as_deref())?;
    let from = app.view_state().clone();
    // Iteration count auto-scales with zoom depth per frame, the same way it
    // does while zooming interactively — no need to interpolate it by hand.
    app.set_auto_iterations(true);

    let out_dir = export_path.unwrap_or_else(|| format!("frames-{}", unix_timestamp()));
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("failed to create {out_dir}: {e}"))?;

    let (device, queue) = pollster::block_on(request_device())?;
    let format = wgpu::TextureFormat::Bgra8Unorm;
    let renderer = FractalRenderer::new(&device, format);
    let (pipeline, bind_group_layout, format) = renderer.export_handles();

    for i in 0..frames {
        let raw_t = i as f64 / (frames - 1) as f64;
        let t = if linear { raw_t } else { smoothstep(raw_t) };
        app.set_view(interpolate_view(&from, &to, t));

        eprintln!("[{:>4}/{frames}] computing reference orbit…", i + 1);
        app.compute_reference_blocking();

        let uniforms = app.make_uniforms(width as f64 / height as f64);
        let er = ExportRender::new(
            &device,
            &queue,
            pipeline.clone(),
            &bind_group_layout,
            format,
            width,
            height,
            uniforms,
            app.reference_points(),
            app.lights(),
        );

        let png = export_to_png_blocking(&device, &queue, &er, |phase, fraction| {
            eprint!(
                "\r[{:>4}/{frames}] {phase} {:>3.0}%",
                i + 1,
                fraction * 100.0
            );
        });
        eprintln!();

        let path = format!("{out_dir}/frame-{:05}.png", i + 1);
        std::fs::write(&path, &png).map_err(|e| format!("save failed: {e}"))?;
    }

    println!("saved {frames} frames to {out_dir}/ ({width}×{height})");
    println!(
        "tip: ffmpeg -framerate {fps} -i {out_dir}/frame-%05d.png -c:v libx264 -pix_fmt yuv420p out.mp4"
    );
    Ok(())
}

/// Parse `--to-view`/`--to-share` (exactly one must be set) into the end
/// view of an animation. Only position/zoom/iterations are pulled from a
/// share fragment — the rest of its state (kind, colors, ...) is ignored, so
/// pasting a link from the app doesn't unexpectedly change the fractal kind
/// mid-animation.
fn parse_animation_target(
    to_view: Option<&str>,
    to_share: Option<&str>,
) -> Result<ViewState, String> {
    if let Some(spec) = to_view {
        return parse_view_spec(spec)
            .map(|(view, _)| view)
            .ok_or_else(|| format!("invalid --to-view spec: {spec}"));
    }
    let frag = to_share.expect("run_animation only called with one of to_view/to_share set");
    let state =
        ShareState::decode(frag).ok_or_else(|| format!("invalid --to-share fragment: {frag}"))?;
    let bits = precision_for(state.half_height);
    let re =
        big_from_decimal_str(&state.center_re, bits).ok_or("invalid --to-share center (re)")?;
    let im =
        big_from_decimal_str(&state.center_im, bits).ok_or("invalid --to-share center (im)")?;
    Ok(ViewState::with_center(re, im, state.half_height))
}

/// Ease-in/ease-out pacing: slow at both ends, fast through the middle.
fn smoothstep(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
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
