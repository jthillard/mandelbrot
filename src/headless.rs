// Headless PNG rendering: parse the CLI, build the exact same view/state the
// windowed app would from it, then render straight to a file. No window, no
// event loop, no worker-thread debounce (nothing to debounce for a one-shot
// render); it just creates its own wgpu device, computes the reference orbit
// once, and renders through the same `ExportRender` path the "Export PNG"
// button uses. `--export-path -` writes to stdout instead: the PNG for a
// single image, or a raw RGBA8 video stream for an animation (for piping
// into ffmpeg).

use std::collections::{BTreeMap, HashMap};
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, mpsc};

use crate::app::{FractalApp, RefJob, parse_complex_pair, unix_timestamp};
use crate::cli::Cli;
use crate::fractal::bla;
use crate::fractal::{
    ExportRender, FractalKind, FractalRenderer, PipelineKey, ShareState, encode_png,
    export_to_png_blocking, render_readback_blocking, unpad_rgba,
};
use crate::view::{
    Scale, ViewState, big_from_decimal_str, interpolate_f64, interpolate_view, parse_view_spec,
    precision_for,
};

/// Cap on the output image dimension (px), to stay within GPU texture limits.
const MAX_DIM: u32 = 8192 * 16;

/// `--export-path` value meaning "write to stdout".
const STDOUT_PATH: &str = "-";

/// Refuse to dump binary image data onto a terminal.
fn check_stdout_piped() -> Result<(), String> {
    if std::io::stdout().is_terminal() {
        return Err(
            "--export-path - writes binary data to stdout; pipe it somewhere \
                    (e.g. `| ffmpeg ...`)"
                .into(),
        );
    }
    Ok(())
}

pub fn run(cli: Cli) -> Result<(), String> {
    if cli.buddhabrot {
        return Err("headless mode doesn't support --buddhabrot yet".into());
    }

    let width = cli.width.clamp(16, MAX_DIM);
    let height = cli.height.clamp(16, MAX_DIM);

    // These drive the animation path below; grab them before `apply_cli`
    // consumes `cli` to build the start state.
    let targets = AnimTargets::from_cli(&cli)?;
    let export_path = cli.export_path.clone();
    if export_path.as_deref() == Some(STDOUT_PATH) {
        check_stdout_piped()?;
    }

    if targets.shard.is_some() && !targets.any() {
        return Err("--shard/--shards only apply to animations (give a --to-* target)".into());
    }

    let mut app = FractalApp::default_state();
    app.apply_cli(cli);
    app.set_output_size(width, height);

    if targets.any() {
        return run_animation(app, targets, width, height, export_path);
    }

    let export_path = export_path.unwrap_or_else(|| format!("fractal-{}.png", unix_timestamp()));

    eprintln!("computing reference orbit…");
    app.compute_reference_blocking();

    let (device, queue) = pollster::block_on(request_device())?;
    let format = wgpu::TextureFormat::Bgra8Unorm;
    let renderer = FractalRenderer::new(&device, format);
    let uniforms = app.make_uniforms(width as f64 / height as f64, height as f64);
    let auto_color = app.auto_color_active();
    let handles = renderer.export_handles(&device, &uniforms, auto_color);

    let mut er = ExportRender::new(
        &device,
        &queue,
        &handles,
        width,
        height,
        uniforms,
        app.reference_points(),
        &bla::for_uniforms(app.reference_points(), &uniforms, app.use_bla()),
        app.lights(),
    );
    if auto_color {
        match fit_auto_color(&mut app, &mut er, &device, &queue) {
            Some((lo, hi)) => eprintln!("auto color scale: ci {lo:.1}–{hi:.1}"),
            None => eprintln!("auto color scale: nothing escaped, keeping the default"),
        }
    }

    eprintln!("rendering {width}×{height}…");
    let png = export_to_png_blocking(&device, &queue, &er, |phase, fraction| {
        eprint!("\r{phase} {:>3.0}%", fraction * 100.0);
    });
    eprintln!();

    if export_path == STDOUT_PATH {
        let mut out = std::io::stdout().lock();
        out.write_all(&png)
            .and_then(|()| out.flush())
            .map_err(|e| format!("writing to stdout failed: {e}"))?;
        eprintln!("wrote PNG to stdout ({width}×{height})");
    } else {
        std::fs::write(&export_path, &png).map_err(|e| format!("save failed: {e}"))?;
        println!("saved {export_path} ({width}×{height})");
    }
    Ok(())
}

/// The `--to-*` end state of a headless animation, plus its pacing. Each
/// target is optional; anything left unset stays at its start value.
struct AnimTargets {
    to_view: Option<String>,
    to_share: Option<String>,
    to_iterations: Option<u32>,
    to_julia: Option<(f64, f64)>,
    to_phoenix_p: Option<(f64, f64)>,
    to_lambda_l: Option<(f64, f64)>,
    /// Complex Multibrot exponent, per component (either may move alone).
    to_cpow_re: Option<f64>,
    to_cpow_im: Option<f64>,
    to_kind: Option<FractalKind>,
    /// 3D camera, degrees.
    to_yaw: Option<f32>,
    to_pitch: Option<f32>,
    to_color_scale: Option<f32>,
    /// Zoom depth where `to_color_scale` is reached (else: at the end).
    to_color_scale_at: Option<Scale>,
    frames: Option<u32>,
    fps: f64,
    duration: Option<f64>,
    linear: bool,
    /// `--shard K --shards N`: render only the K-th (1-based) of N parts.
    shard: Option<(u32, u32)>,
}

impl AnimTargets {
    fn from_cli(cli: &Cli) -> Result<Self, String> {
        let pair = |flag: &str, v: &Option<String>| -> Result<Option<(f64, f64)>, String> {
            v.as_deref()
                .map(|s| parse_complex_pair(s).ok_or_else(|| format!("invalid --{flag}: {s}")))
                .transpose()
        };
        let to_cpow = pair("to-complex-power", &cli.to_complex_power)?;
        let shard = match (cli.shard, cli.shards) {
            (None, None) | (None, Some(1)) => None,
            (Some(k), Some(n)) if (1..=n).contains(&k) => Some((k, n)),
            (Some(k), Some(n)) => {
                return Err(format!(
                    "--shard {k} is out of range 1..={n} (--shards {n})"
                ));
            }
            _ => return Err("--shard and --shards must be given together".into()),
        };
        if let Some(s) = cli.to_color_scale {
            if !(s.is_finite() && s > 0.0) {
                return Err(format!("invalid --to-color-scale: {s}"));
            }
            if cli.auto_color_scale {
                return Err(
                    "--to-color-scale can't be combined with --auto-color-scale \
                     (the auto fit sets the scale on every frame)"
                        .into(),
                );
            }
        }
        let to_color_scale_at = cli
            .to_color_scale_at
            .as_deref()
            .map(|s| {
                s.parse::<Scale>()
                    .map_err(|_| format!("invalid --to-color-scale-at: {s}"))
            })
            .transpose()?;
        if to_color_scale_at.is_some() && cli.to_color_scale.is_none() {
            return Err("--to-color-scale-at needs --to-color-scale".into());
        }
        Ok(Self {
            to_view: cli.to_view.clone(),
            to_share: cli.to_share.clone(),
            to_iterations: cli.to_iterations,
            to_julia: pair("to-julia", &cli.to_julia)?,
            to_phoenix_p: pair("to-phoenix-p", &cli.to_phoenix_p)?,
            to_lambda_l: pair("to-lambda-l", &cli.to_lambda_l)?,
            to_cpow_re: cli.to_complex_power_re.or(to_cpow.map(|p| p.0)),
            to_cpow_im: cli.to_complex_power_im.or(to_cpow.map(|p| p.1)),
            to_kind: cli.to_kind.map(Into::into),
            to_yaw: cli.to_yaw,
            to_pitch: cli.to_pitch,
            to_color_scale: cli.to_color_scale,
            to_color_scale_at,
            frames: cli.frames,
            fps: cli.fps,
            duration: cli.duration,
            linear: cli.linear,
            shard,
        })
    }

    /// Whether any end state was given, i.e. this is an animation.
    fn any(&self) -> bool {
        self.to_view.is_some()
            || self.to_share.is_some()
            || self.to_iterations.is_some()
            || self.to_julia.is_some()
            || self.to_phoenix_p.is_some()
            || self.to_lambda_l.is_some()
            || self.to_cpow_re.is_some()
            || self.to_cpow_im.is_some()
            || self.to_kind.is_some()
            || self.to_yaw.is_some()
            || self.to_pitch.is_some()
            || self.to_color_scale.is_some()
    }
}

/// Render a sequence of frames interpolating from the app's current (start)
/// state to `targets`, for feeding into ffmpeg: the camera, iteration count,
/// per-kind constants (c, p, λ, complex power) and, through a kind morph,
/// the iteration formula, the 3D camera angles and the colour scale.
/// Everything else (palette, offset, ...) stays fixed at whatever
/// `apply_cli` set up for the start. With `--export-path -`, frames are
/// streamed in order to stdout as raw RGBA8 (for ffmpeg's `rawvideo`
/// demuxer) instead of being written as PNGs.
fn run_animation(
    mut app: FractalApp,
    targets: AnimTargets,
    width: u32,
    height: u32,
    export_path: Option<String>,
) -> Result<(), String> {
    let fps = targets.fps;
    let frames = match targets.frames {
        Some(n) => n,
        None => {
            let dur = targets
                .duration
                .ok_or("animation needs --frames, or --duration (with --fps)")?;
            ((fps * dur).round() as u32).max(2)
        }
    };
    if frames < 2 {
        return Err("animation needs at least 2 frames".into());
    }
    // Frames are still timed against the whole animation (`apply_frame` takes
    // the global index); a shard only picks which of them this run renders.
    let range = match targets.shard {
        Some((_, n)) if n > frames => {
            return Err(format!("--shards {n} is more than the {frames} frames"));
        }
        Some((k, n)) => shard_range(frames, k, n),
        None => 0..frames,
    };
    let (first, count) = (range.start, range.len());

    let from = app.view_state().clone();
    let (to, to_iterations_share) =
        parse_animation_target(targets.to_view.as_deref(), targets.to_share.as_deref())?
            .unwrap_or_else(|| (from.clone(), None));
    let to_iterations = targets.to_iterations.or(to_iterations_share);
    let from_iterations = app.max_iterations();
    if to_iterations.is_none() {
        // Iteration count auto-scales with zoom depth per frame, the same way it
        // does while zooming interactively — no need to interpolate it by hand.
        app.set_auto_iterations(true);
    }

    let from_consts = app.constants();
    let [c0, p0, l0, cp0] = from_consts;
    let to_consts = [
        targets.to_julia.unwrap_or(c0),
        targets.to_phoenix_p.unwrap_or(p0),
        targets.to_lambda_l.unwrap_or(l0),
        (
            targets.to_cpow_re.unwrap_or(cp0.0),
            targets.to_cpow_im.unwrap_or(cp0.1),
        ),
    ];
    let from_kind = app.kind();
    let to_kind = targets.to_kind.unwrap_or(from_kind);
    let (yaw0, pitch0) = app.camera_angles();
    let yaw1 = targets.to_yaw.map_or(yaw0, f32::to_radians);
    let pitch1 = targets.to_pitch.map_or(pitch0, f32::to_radians);
    let (scale0, ci_lo0) = app.color_fit();
    let scale1 = targets.to_color_scale.map(|s| s.clamp(1e-4, 1.0) as f64);
    // With `--to-color-scale-at`, the scale follows the zoom depth (log2 of
    // the half-height) from the start view's to that one, instead of `t`.
    let depth0 = from.half_height.log2();
    let depth_at = targets.to_color_scale_at.map(Scale::log2);
    if depth_at == Some(depth0) {
        return Err("--to-color-scale-at is the start view's depth".into());
    }

    let stream = export_path.as_deref() == Some(STDOUT_PATH);
    let out_dir = export_path.unwrap_or_else(|| format!("frames-{}", unix_timestamp()));
    if stream {
        eprintln!(
            "streaming raw video to stdout; ffmpeg input: \
             -f rawvideo -pix_fmt rgba -s {width}x{height} -r {fps} -i -"
        );
    } else {
        std::fs::create_dir_all(&out_dir)
            .map_err(|e| format!("failed to create {out_dir}: {e}"))?;
    }

    // Everything about frame `i` is a pure function of its `t`, so the app can
    // be put into any frame's state at any time, in any order.
    let apply_frame = |app: &mut FractalApp, i: u32| {
        let raw_t = i as f64 / (frames - 1) as f64;
        let t = if targets.linear {
            raw_t
        } else {
            smoothstep(raw_t)
        };
        if let Some(to) = to_iterations {
            app.set_max_iterations(
                interpolate_f64(from_iterations as f64, to as f64, t).round() as u32,
            );
        }
        let view = interpolate_view(&from, &to, t);
        let depth = view.half_height.log2();
        app.set_view(view);
        app.set_constants(std::array::from_fn(|k| {
            (
                interpolate_f64(from_consts[k].0, to_consts[k].0, t),
                interpolate_f64(from_consts[k].1, to_consts[k].1, t),
            )
        }));
        app.set_kind_morph(from_kind, to_kind, t);
        app.set_camera_angles(
            interpolate_f64(yaw0 as f64, yaw1 as f64, t) as f32,
            interpolate_f64(pitch0 as f64, pitch1 as f64, t) as f32,
        );
        // Geometrically: the useful range spans decades.
        if let Some(scale1) = scale1 {
            // The zoom is already eased; don't ease its depth again.
            let u = depth_at.map_or(t, |at| ((depth - depth0) / (at - depth0)).clamp(0.0, 1.0));
            let scale = interpolate_f64((scale0 as f64).ln(), scale1.ln(), u).exp();
            app.set_color_fit((scale as f32, ci_lo0));
        }
    };

    // Snapshot every frame's reference-orbit job up front (cheap: just the
    // parameters), so the orbits themselves can be computed in parallel.
    // `jobs[j]` is global frame `first + j`; the pipeline below works in
    // local indices `j`, so the stdout writer's ordering is per shard.
    let jobs: Vec<RefJob> = range
        .clone()
        .map(|i| {
            apply_frame(&mut app, i);
            app.reference_job()
        })
        .collect();

    let (device, queue) = pollster::block_on(request_device())?;
    let format = wgpu::TextureFormat::Bgra8Unorm;
    let renderer = FractalRenderer::new(&device, format);
    let aspect = width as f64 / height as f64;
    // Refit per frame from the same start, so each frame's colours depend only
    // on that frame (frames render out of order, and shards must join up).
    let auto_color = app.auto_color_active();
    let color_fit0 = app.color_fit();

    // Three-stage pipeline, connected by bounded channels (which also cap
    // memory): `threads` workers compute reference orbits (CPU, the expensive
    // part at deep zoom) → this thread renders each frame on the GPU → `threads`
    // workers PNG-encode and write frames. Frames flow through out of order
    // (at most ~`threads` apart); each is written under its own index.
    //
    // When streaming, the last stage instead unpads frames to raw RGBA and a
    // single writer thread puts them back in order before writing to stdout.
    // Its reorder buffer can't apply backpressure (blocking it while waiting
    // for frame `k` could stall the pipeline before `k` gets through), so the
    // orbit workers bound it instead: they don't start a frame more than
    // `window` ahead of the last one written.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let window = threads * 4;
    let next_job = AtomicUsize::new(0);
    let saved = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let error: Mutex<Option<String>> = Mutex::new(None);
    let fail = |e: String| {
        failed.store(true, Ordering::Relaxed);
        error.lock().unwrap().get_or_insert(e);
    };

    if let Some((k, n)) = targets.shard {
        eprintln!(
            "shard {k}/{n}: frames {}–{} of {frames}",
            range.start + 1,
            range.end
        );
    }
    eprintln!("rendering {count} frames ({width}×{height}) on {threads} threads…");
    let (png_tx, png_rx) = mpsc::sync_channel::<(usize, Vec<u8>, u32, bool)>(threads * 2);
    let png_rx = Mutex::new(png_rx);
    let (raw_tx, raw_rx) = mpsc::sync_channel::<(usize, Vec<u8>)>(threads * 2);
    std::thread::scope(|scope| {
        let (ref_tx, ref_rx) = mpsc::sync_channel::<(usize, crate::fractal::RefOrbit)>(threads * 2);
        for _ in 0..threads {
            let ref_tx = ref_tx.clone();
            let (jobs, next_job, saved, failed) = (&jobs, &next_job, &saved, &failed);
            scope.spawn(move || {
                loop {
                    let i = next_job.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() || failed.load(Ordering::Relaxed) {
                        break;
                    }
                    while stream
                        && i >= saved.load(Ordering::Relaxed) + window
                        && !failed.load(Ordering::Relaxed)
                    {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                    if ref_tx.send((i, jobs[i].compute())).is_err() {
                        break;
                    }
                }
            });
        }
        drop(ref_tx);

        if stream {
            let (saved, fail) = (&saved, &fail);
            scope.spawn(move || {
                let mut out = std::io::stdout().lock();
                let mut pending = BTreeMap::new();
                let mut next = 0;
                for (i, raw) in raw_rx.iter() {
                    pending.insert(i, raw);
                    while let Some(raw) = pending.remove(&next) {
                        if let Err(e) = out.write_all(&raw) {
                            fail(format!("writing to stdout failed: {e}"));
                            return;
                        }
                        next += 1;
                        saved.store(next, Ordering::Relaxed);
                        eprint!("\r[{next:>4}/{count}] streamed");
                    }
                }
                if let Err(e) = out.flush() {
                    fail(format!("writing to stdout failed: {e}"));
                }
            });
        } else {
            drop(raw_rx);
        }

        for _ in 0..threads {
            let raw_tx = raw_tx.clone();
            let (png_rx, out_dir, saved, failed, fail) =
                (&png_rx, &out_dir, &saved, &failed, &fail);
            scope.spawn(move || {
                loop {
                    // Hold the lock only for the receive, not the encode.
                    let Ok((i, padded, bpr, swap_rb)) = png_rx.lock().unwrap().recv() else {
                        break;
                    };
                    // After a failure, keep draining (without work) until the
                    // GPU stage hangs up, so it can't block on a full channel.
                    if failed.load(Ordering::Relaxed) {
                        continue;
                    }
                    if stream {
                        let raw = unpad_rgba(&padded, width, height, bpr, swap_rb);
                        // Only fails once the writer has failed and hung up.
                        let _ = raw_tx.send((i, raw));
                        continue;
                    }
                    let png =
                        encode_png(&padded, width, height, bpr, swap_rb, png::Compression::Fast);
                    let path = format!("{out_dir}/frame-{:05}.png", first as usize + i + 1);
                    if let Err(e) = std::fs::write(&path, &png) {
                        fail(format!("save failed: {e}"));
                        continue;
                    }
                    let done = saved.fetch_add(1, Ordering::Relaxed) + 1;
                    eprint!("\r[{done:>4}/{count}] saved");
                }
            });
        }

        // GPU stage, on this thread (it owns the app and the device). The
        // shader specialization (kind, Julia, DE, morph) can change between
        // frames during a kind morph; build each pipeline once.
        let mut pipelines = HashMap::new();
        for (i, points) in ref_rx.iter() {
            if failed.load(Ordering::Relaxed) {
                break;
            }
            apply_frame(&mut app, first + i as u32);
            app.finish_reference(jobs[i].clone(), points);

            let uniforms = app.make_uniforms(aspect, height as f64);
            let handles = pipelines
                .entry(PipelineKey::from_uniforms(&uniforms))
                .or_insert_with(|| renderer.export_handles(&device, &uniforms, auto_color));
            let mut er = ExportRender::new(
                &device,
                &queue,
                handles,
                width,
                height,
                uniforms,
                app.reference_points(),
                // Built here rather than in the orbit pool: it needs this
                // frame's uniforms, and costs ~1 ms per 100k points.
                &bla::for_uniforms(app.reference_points(), &uniforms, app.use_bla()),
                app.lights(),
            );
            if auto_color {
                app.set_color_fit(color_fit0);
                fit_auto_color(&mut app, &mut er, &device, &queue);
            }
            let padded = render_readback_blocking(&device, &queue, &er);
            if png_tx.send((i, padded, er.padded_bpr, er.swap_rb)).is_err() {
                break;
            }
        }
        // Dropping the channel ends lets the workers drain and exit.
        drop(raw_tx);
        drop(png_tx);
        drop(ref_rx);
    });
    eprintln!();

    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    let saved = saved.into_inner();
    if saved != count {
        return Err(format!("only {saved} of {count} frames were rendered"));
    }

    if stream {
        eprintln!("streamed {count} frames ({width}×{height})");
        return Ok(());
    }
    println!("saved {count} frames to {out_dir}/ ({width}×{height})");
    if targets.shard.is_some() {
        println!(
            "tip: once every shard is rendered into {out_dir}/, they form the full sequence; \
             this shard alone: ffmpeg -framerate {fps} -start_number {} -i {out_dir}/frame-%05d.png \
             -frames:v {count} -c:v libx264 -pix_fmt yuv420p out.mp4",
            range.start + 1
        );
    } else {
        println!(
            "tip: ffmpeg -framerate {fps} -i {out_dir}/frame-%05d.png -c:v libx264 -pix_fmt yuv420p out.mp4"
        );
    }
    Ok(())
}

/// Auto colour scale: fit `app`'s colour scale to `er`'s image (a prepass)
/// and upload the result. Returns the fitted `ci` range, or `None` (colours
/// untouched) if nothing escaped.
fn fit_auto_color(
    app: &mut FractalApp,
    er: &mut ExportRender,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Option<(f32, f32)> {
    let range = er.ci_range_blocking(device, queue)?;
    app.apply_ci_range(range);
    er.set_uniforms(
        queue,
        app.make_uniforms(er.width as f64 / er.height as f64, er.height as f64),
    );
    Some(range)
}

/// Parse `--to-view`/`--to-share` (at most one is used) into the end view
/// of an animation, or `None` if neither is set (the camera stays put). Only position/zoom/iterations are pulled from a
/// share fragment — the rest of its state (kind, colors, ...) is ignored, so
/// pasting a link from the app doesn't unexpectedly change the fractal kind
/// mid-animation.
fn parse_animation_target(
    to_view: Option<&str>,
    to_share: Option<&str>,
) -> Result<Option<(ViewState, Option<u32>)>, String> {
    if let Some(spec) = to_view {
        return parse_view_spec(spec)
            .map(Some)
            .ok_or_else(|| format!("invalid --to-view spec: {spec}"));
    }
    let Some(frag) = to_share else {
        return Ok(None);
    };
    let state =
        ShareState::decode(frag).ok_or_else(|| format!("invalid --to-share fragment: {frag}"))?;
    let bits = precision_for(state.half_height);
    let re =
        big_from_decimal_str(&state.center_re, bits).ok_or("invalid --to-share center (re)")?;
    let im =
        big_from_decimal_str(&state.center_im, bits).ok_or("invalid --to-share center (im)")?;
    Ok(Some((
        ViewState::with_center(re, im, state.half_height),
        Some(state.iterations),
    )))
}

/// Global frame indices of shard `shard` (1-based) out of `shards` equal
/// parts of a `frames`-frame animation. Consecutive shards tile `0..frames`
/// with no gap or overlap.
fn shard_range(frames: u32, shard: u32, shards: u32) -> std::ops::Range<u32> {
    let bound = |k: u32| (k as u64 * frames as u64 / shards as u64) as u32;
    bound(shard - 1)..bound(shard)
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

#[cfg(test)]
mod tests {
    use super::shard_range;

    #[test]
    fn shards_tile_all_frames() {
        for frames in [2, 3, 10, 97, 1000, u32::MAX] {
            for shards in [1, 2, 3, 7, 10] {
                if shards > frames {
                    continue;
                }
                let mut next = 0;
                for k in 1..=shards {
                    let r = shard_range(frames, k, shards);
                    assert_eq!(
                        r.start, next,
                        "gap/overlap at shard {k}/{shards} of {frames}"
                    );
                    assert!(!r.is_empty(), "empty shard {k}/{shards} of {frames}");
                    next = r.end;
                }
                assert_eq!(next, frames);
            }
        }
    }
}
