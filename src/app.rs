use std::sync::{Arc, Mutex};

use eframe::CreationContext;
use eframe::egui_wgpu;
#[cfg(target_arch = "wasm32")]
use eframe::egui_wgpu::wgpu;
use glam::Vec4;
use glam::Vec4Swizzles;

use crate::camera::Camera;
#[cfg(not(target_arch = "wasm32"))]
use crate::cli::Cli;
use crate::fractal::{
    BuddhabrotCallback, BuddhabrotRenderer, BuddhabrotUniforms, ExportRender, FractalCallback,
    FractalKind, FractalRenderer, MAX_REF_POINTS, ShareState, Uniforms, compute_reference,
    compute_set_reference,
};
use crate::lights::{Light, gpu_lights};
use crate::view::parse_half_height_spec;
use crate::view::parse_re_im_spec;
use crate::view::{
    Big, DEFAULT_HALF_HEIGHT, ViewState, big_from_decimal_str, big_from_f64, big_to_decimal_str,
    parse_view_spec, precision_for,
};
#[cfg(not(target_arch = "wasm32"))]
use clap::Parser;

const BAILOUT_SQ: f32 = 1.0e6;
/// Cap on exported image dimension (px), to stay within GPU texture limits.
const MAX_EXPORT_DIM: u32 = 8192 * 16;
/// While the user is actively panning/zooming, the fractal is rendered into a
/// cache texture downscaled by this factor per axis (and with AA forced off), so
/// each interacting frame is cheap; the linear blit upsamples it to the widget.
/// A full-resolution render replaces it once input settles. 2 → quarter the
/// pixels (~4× faster); raise for more speed at the cost of more blur in motion.
const INTERACT_DOWNSCALE: u32 = 2;
/// Seconds without pan/zoom input after which the view counts as settled and is
/// re-rendered at full resolution.
const INTERACT_SETTLE: f64 = 0.12;
/// Palette names; index maps to `palette_id` in the shader.
const PALETTE_NAMES: &[&str] = &["Amber", "Rainbow", "Ember", "Lime", "Grayscale"];
/// Shadow palette names; index maps to `palette_id` in the shader.
const SHADOW_PALETTE_NAMES: &[&str] = &["Grayscale", "Red & Blue", "Custom lights"];
/// Buddhabrot tonemap style names; index maps to `BuddhabrotUniforms::palette`.
const BUDDHA_PALETTE_NAMES: &[&str] = &["Nebula", "Yellow", "Grayscale"];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FractalMode {
    Mandelbrot,
    Julia,
    Buddhabrot,
}

type JuliaPreset = (&'static str, f64, f64, u32, Option<(f64, f64)>);

/// Nice-looking Julia constants offered as presets.
const JULIA_PRESETS: [&[JuliaPreset]; FractalKind::ComplexMultibrot as usize + 1] = [
    &[
        ("dendrite", -0.8, 0.156, 400, None),
        ("rabbit", -0.123, 0.745, 400, None),
        ("spiral", -0.4, 0.6, 400, None),
        ("san marco", -0.75, 0.0, 400, None),
        ("siegel", -0.391, -0.587, 400, None),
    ],
    &[("eyes", -0.241, 0.157, 1000, None)],
    &[("pools", -0.50381, 0.07750, 400, None)],
    &[],
    &[],
    &[],
    &[],
    &[
        ("archipelago 1", -0.415, -0.267, 500, Some((-0.556, 0.253))),
        ("archipelago 2", -0.556, 0.253, 500, Some((-0.415, -0.267))),
    ],
    &[],
    &[],
];

type SetPreset = (
    &'static str,
    &'static str,
    &'static str,
    f64,
    u32,
    Option<(f64, f64)>,
);

/// Curated beautiful locations offered as one-click presets.
/// Each is `(name, center_re, center_im, half_height, iterations)`; the centers
/// are decimals parsed at full precision so deep places stay sharp.
const SET_PRESETS: [&[SetPreset]; FractalKind::ComplexMultibrot as usize + 1] = [
    &[
        (
            "Seahorse Valley",
            "-0.743643887037158704752191506114774",
            "0.131825904205311970493132056385139",
            4.0e-6,
            1500,
            None,
        ),
        (
            "Elephant Valley",
            "0.2549870375144766",
            "0.0005679790528465",
            6.0e-5,
            2000,
            None,
        ),
        ("Scepter Valley", "-1.36012", "0.0406", 2.5e-4, 2000, None),
        ("Starburst", "-1.62917", "0.0203968", 1.5e-3, 1500, None),
        (
            "Deep Spiral",
            "-0.7436438870371587",
            "0.1318259042053",
            8.0e-8,
            2000,
            None,
        ),
    ],
    &[(
        "Ship",
        "-1.76485017213465",
        "-0.0317013204392752",
        5.3e-2,
        1500,
        None,
    )],
    &[],
    &[],
    &[],
    &[],
    &[],
    &[(
        "Galaxy",
        "-0.2165696026100408",
        "-0.0676553191878954",
        5e-1,
        1000,
        Some((-0.9, -0.49)),
    )],
    &[],
    &[],
];

/// Parameters a reference orbit was (or will be) computed for. Used to decide
/// when the current reference is stale enough to recompute.
struct RequestKey {
    center_re: Big,
    center_im: Big,
    half_height: f64,
    julia: bool,
    julia_c: (f64, f64),
    phoenix_p: (f64, f64),
    lambda_l: (f64, f64),
    iter: u32,
    kind: FractalKind,
    power: u32,
    complex_power: (f64, f64),
}

/// Shared state for an in-progress PNG export. The worker (a background thread
/// on native, an async task on web) writes `fraction`/`phase` as it goes and
/// sets `result` once when finished; the UI reads it each frame to draw a
/// progress bar and, on completion, to report the outcome.
struct ExportShared {
    fraction: f32,
    phase: &'static str,
    result: Option<Result<String, String>>,
}

/// Time-based animation of a few view/coloring parameters. Each toggle drives
/// continuous repaints while on; orbit-affecting ones (Julia c, Phoenix p, zoom)
/// recompute the reference each frame and render the cheap low-res pass so they
/// stay smooth.
#[derive(Clone)]
struct AnimState {
    /// Cycle the palette offset (colours flow through the fractal).
    color: bool,
    /// Palette cycles per second.
    color_speed: f32,

    /// Drift the Julia constant `c` around a circle to morph the Julia set.
    julia: bool,
    /// Revolutions per second.
    julia_speed: f32,
    /// Circle radius in the c-plane.
    julia_radius: f64,
    /// Circle center, captured when the animation is enabled.
    julia_base: (f64, f64),
    julia_angle: f64,

    /// Drift the Phoenix distortion `p` around a circle.
    phoenix: bool,
    phoenix_speed: f32,
    phoenix_radius: f64,
    phoenix_base: (f64, f64),
    phoenix_angle: f64,

    /// Drift the Lambda distortion `λ` around a circle.
    lambda: bool,
    lambda_speed: f32,
    lambda_radius: f64,
    lambda_base: (f64, f64),
    lambda_angle: f64,

    /// Continuously zoom toward the current center.
    zoom: bool,
    /// e-folds per second; positive zooms in, negative zooms out.
    zoom_speed: f32,

    /// Linear 2D <-> 3D transition progress in [0, 1], advanced at a constant
    /// rate; `camera_state` is its smoothstep-eased value.
    camera_progress: f32,
    /// Camera state in [0, 1]: 0 = top-down 2D view, 1 = full 3D camera.
    camera_state: f32,
}

impl Default for AnimState {
    fn default() -> Self {
        Self {
            color: false,
            color_speed: 0.15,
            julia: false,
            julia_speed: 0.05,
            julia_radius: 0.08,
            julia_base: (0.0, 0.0),
            julia_angle: 0.0,
            phoenix: false,
            phoenix_speed: 0.05,
            phoenix_radius: 0.08,
            phoenix_base: (0.0, 0.0),
            phoenix_angle: 0.0,
            lambda: false,
            lambda_speed: 0.05,
            lambda_radius: 0.08,
            lambda_base: (0.0, 0.0),
            lambda_angle: 0.0,
            zoom: false,
            zoom_speed: 0.5,
            camera_progress: 0.,
            camera_state: 0.,
        }
    }
}

/// Top-level egui application.
pub struct FractalApp {
    view: ViewState,
    mode: FractalMode,
    /// Iteration formula.
    kind: FractalKind,
    /// Exponent for the Multibrot kind.
    power: u32,
    /// Complex exponent for the Complex Multibrot kind (`z^power + c`).
    complex_power: (f64, f64),
    julia_c: (f64, f64),
    /// Distortion constant `p` for the Phoenix kind (`z^2 + c + p·z_{n-1}`).
    phoenix_p: (f64, f64),
    /// Distortion constant `l` for the Lambda kind (`l·z(1 - z_{n-1})`).
    lambda_l: (f64, f64),
    max_iterations: u32,
    /// When set, `max_iterations` tracks the zoom depth automatically (so deep
    /// zooms stay sharp without hand-tuning); the manual slider takes over when
    /// unset. Turned off when a preset or share link supplies an explicit count.
    auto_iterations: bool,
    color_scale: f32,
    color_offset: f32,
    palette: u32,
    shadow_palette: u32,
    /// Supersample each pixel 2×2 for smoother edges (costs ~4× fragment work).
    antialias: bool,
    /// Distance-estimation shading: darkens toward the set boundary using the
    /// orbit derivative, giving crisp filaments at deep zoom instead of speckle.
    de_coloring: bool,
    // Use shadow coloring
    // Use 3D raymarching rendering
    rendering_mode: u32,

    /// List of enabled lights in the world
    lights: Vec<Light>,

    /// Nested escape-iteration caps for the R/G/B histogram channels
    /// (Nebulabrot coloring); kept ordered r <= g <= b by the UI.
    buddha_r_cap: u32,
    buddha_g_cap: u32,
    buddha_b_cap: u32,
    /// Tonemap brightness multiplier.
    buddha_exposure: f32,
    /// Tonemap colour style (index into `BUDDHA_PALETTE_NAMES`).
    buddha_palette: u32,
    /// Keep dispatching new sample batches every frame (progressive
    /// accumulation). Turning it off freezes the current histogram.
    buddha_accumulate: bool,
    /// Whether the controls side panel is expanded. Collapsible so the fractal
    /// can take (nearly) the whole screen — important on a phone.
    controls_open: bool,
    /// Whether the app is in fullscreen (browser Fullscreen API on web, viewport
    /// fullscreen on native). Kept in sync with the real state each frame.
    fullscreen: bool,
    /// Whether the "Fractal Info" popup (formula/constants/zoom for the
    /// current view) is open.
    info_open: bool,
    /// Whether the Help window (about + mouse/touch controls) is open.
    help_open: bool,
    /// Time-based animation of colours / Julia c / Phoenix p / zoom.
    anim: AnimState,

    /// Smoothed frames-per-second, recomputed each ~0.5 s window. Only advances
    /// while the app is actually repainting (interaction / animation / export);
    /// idle frames aren't forced, so a frozen value means "nothing to render".
    fps: f32,
    /// Frames counted in the current FPS window, and its start time (`i.time`).
    fps_frames: u32,
    fps_window_start: f64,

    /// Reference orbit (`Z_n` as f32 pairs) for the current view.
    reference: Arc<Vec<[f32; 2]>>,
    /// Bumped whenever `reference` is replaced, so the GPU re-uploads it.
    generation: u64,
    /// Center + zoom the current `reference` was computed at (may differ
    /// slightly from the live view; the shader compensates via `dc_offset`).
    ref_center_re: Big,
    ref_center_im: Big,
    ref_half_height: f64,
    /// Parameters of the most recent reference request (drift baseline / dedupe).
    last_request: Option<RequestKey>,

    #[cfg(not(target_arch = "wasm32"))]
    worker: crate::worker::RefWorker,
    /// A reference computation is in flight (native async worker).
    pending: bool,

    /// PNG export resolution multiplier over the on-screen size.
    export_scale: f32,
    /// Last on-screen fractal size in physical pixels (for export sizing).
    last_size_px: egui::Vec2,
    /// egui time (seconds) of the most recent pan/zoom. While recent (within
    /// `INTERACT_SETTLE`) the fractal renders downscaled for smooth interaction.
    last_interact_time: f64,
    /// Set when the user requests a PNG export (handled after the panels draw).
    export_requested: bool,
    /// Progress/handle for an in-flight PNG export, if any.
    export: Option<Arc<Mutex<ExportShared>>>,
    /// Output path for `--export-path` (native CLI only); falls back to a
    /// timestamped name when unset.
    export_path: Option<String>,
    /// Short status line (saved path, "link copied", errors).
    status: Option<String>,

    /// Editable text buffers for the center coordinates (decimal, full
    /// precision). Kept in sync with the live view except while the field is
    /// focused, so the user's in-progress typing is not clobbered by pan/zoom.
    center_re_edit: String,
    center_im_edit: String,
    /// Editable magnification (×). Its display is lossy, so `zoom_edited` guards
    /// applying it: without that, clicking in and out would round-trip the value
    /// through the display format and drift the zoom.
    zoom_edit: String,
    zoom_edited: bool,

    /// The camera used to render 3D fractals
    camera: Camera,
    /// Screen dimension.
    screen_dim: [f32; 2],
}

/// Significant decimal digits to show for a center at the given precision (bits).
fn sig_digits_for(bits: usize) -> usize {
    ((bits as f64) * std::f64::consts::LOG10_2).ceil() as usize + 3
}

/// Format a magnification for the editable field (compact scientific).
fn format_zoom(m: f64) -> String {
    format!("{m:.4e}")
}

/// Precision (bits) to parse a typed center at: at least what the current zoom
/// needs, but enough to preserve every digit the user pasted, so a deep
/// coordinate entered while zoomed out is not truncated. Capped like `view`.
fn parse_bits_for(s: &str, min_bits: usize) -> usize {
    let digits = s.chars().filter(char::is_ascii_digit).count();
    let from_input = (digits as f64 * std::f64::consts::LOG2_10).ceil() as usize + 16;
    min_bits.max(from_input).min(2048)
}

impl FractalApp {
    pub fn new(cc: &CreationContext<'_>) -> Self {
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe must run with the wgpu backend");

        let renderer = FractalRenderer::new(&render_state.device, render_state.target_format);
        let buddhabrot_renderer =
            BuddhabrotRenderer::new(&render_state.device, render_state.target_format);
        {
            let mut guard = render_state.renderer.write();
            guard.callback_resources.insert(renderer);
            guard.callback_resources.insert(buddhabrot_renderer);
        }

        let mut app = Self::default_state();

        // On the web, restore a shared view from the URL fragment (#...).
        #[cfg(target_arch = "wasm32")]
        if let Some(frag) = web_location_hash() {
            if let Some(state) = ShareState::decode(&frag) {
                app.apply_share(&state);
            }
        }

        cc.egui_ctx.set_zoom_factor(1.1);

        // Debug/testing hooks, driven by CLI flags.
        #[cfg(not(target_arch = "wasm32"))]
        app.apply_cli(Cli::parse());

        app
    }

    /// Build the app's default state (no window, no GPU, no CLI applied yet).
    /// Shared by the windowed app (`new`, which then layers CLI/share-link
    /// overrides on top) and headless rendering.
    pub(crate) fn default_state() -> Self {
        let view = ViewState::default();
        let ref_center_re = view.center_re.clone();
        let ref_center_im = view.center_im.clone();
        let ref_half_height = view.half_height;
        let sig = sig_digits_for(view.precision_bits());
        let center_re_edit = big_to_decimal_str(&view.center_re, sig);
        let center_im_edit = big_to_decimal_str(&view.center_im, sig);
        let zoom_edit = format_zoom(view.zoom());

        Self {
            view,
            mode: FractalMode::Mandelbrot,
            kind: FractalKind::Mandelbrot,
            power: 3,
            complex_power: (2.0, 0.5),
            julia_c: (-0.8, 0.156),
            phoenix_p: (-0.5, 0.0),
            lambda_l: (-0.5, 0.0),
            max_iterations: 512,
            auto_iterations: true,
            color_scale: 0.15,
            color_offset: 0.0,
            palette: 0,
            shadow_palette: 0,
            antialias: false,
            de_coloring: false,
            rendering_mode: 0,
            lights: vec![Light::default()],
            buddha_r_cap: 50,
            buddha_g_cap: 500,
            buddha_b_cap: 2000,
            buddha_exposure: 1.0,
            buddha_palette: 0,
            buddha_accumulate: true,
            controls_open: true,
            fullscreen: false,
            info_open: false,
            help_open: false,
            anim: AnimState::default(),
            fps: 0.0,
            fps_frames: 0,
            fps_window_start: 0.0,
            reference: Arc::new(Vec::new()),
            generation: 0,
            ref_center_re,
            ref_center_im,
            ref_half_height,
            last_request: None,
            #[cfg(not(target_arch = "wasm32"))]
            worker: crate::worker::RefWorker::spawn(),
            pending: false,
            export_scale: 2.0,
            last_size_px: egui::vec2(1280.0, 720.0),
            last_interact_time: -1.0e9,
            export_requested: false,
            export: None,
            export_path: None,
            status: None,
            center_re_edit,
            center_im_edit,
            zoom_edit,
            zoom_edited: false,
            camera: Camera::new(),
            screen_dim: [0., 0.],
        }
    }

    /// Apply native CLI flags on top of the default state: fractal kind/mode,
    /// a restored share link or explicit view, coloring toggles, and export
    /// options. Shared by the windowed app and headless rendering.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn apply_cli(&mut self, cli: Cli) {
        if let Some(k) = cli.kind {
            self.kind = k.into();
            if let Some(p) = cli.power {
                self.power = p.clamp(2, 8);
            }
            if let Some(cp) = &cli.complex_power {
                let p: Vec<&str> = cp.split(',').collect();
                if let (Some(Ok(re)), Some(Ok(im))) = (
                    p.first().map(|s| s.trim().parse::<f64>()),
                    p.get(1).map(|s| s.trim().parse::<f64>()),
                ) {
                    self.complex_power = (re, im);
                }
            }
            self.view = Self::default_view_for(self.mode, self.kind);
        }
        if let Some(k) = cli.rendering_kind {
            use crate::cli::RenderingKindArg;

            match k {
                RenderingKindArg::Classic => self.rendering_mode = 0,
                RenderingKindArg::Shadow => self.rendering_mode = 1,
                RenderingKindArg::Dimension3 => self.rendering_mode = 2,
            }
        }
        if let Some(jc) = cli.julia {
            let p: Vec<&str> = jc.split(',').collect();
            if let (Some(Ok(re)), Some(Ok(im))) = (
                p.first().map(|s| s.trim().parse::<f64>()),
                p.get(1).map(|s| s.trim().parse::<f64>()),
            ) {
                self.mode = FractalMode::Julia;
                self.julia_c = (re, im);
                self.view = Self::default_view_for(FractalMode::Julia, self.kind);
            }
        }
        if let Some(pp) = &cli.phoenix_p {
            let p: Vec<&str> = pp.split(',').collect();
            if let (Some(Ok(re)), Some(Ok(im))) = (
                p.first().map(|s| s.trim().parse::<f64>()),
                p.get(1).map(|s| s.trim().parse::<f64>()),
            ) {
                self.phoenix_p = (re, im);
            }
        }
        if let Some(ll) = &cli.lambda_l {
            let p: Vec<&str> = ll.split(',').collect();
            if let (Some(Ok(re)), Some(Ok(im))) = (
                p.first().map(|s| s.trim().parse::<f64>()),
                p.get(1).map(|s| s.trim().parse::<f64>()),
            ) {
                self.lambda_l = (re, im);
            }
        }
        if let Some(frag) = cli.share
            && let Some(state) = ShareState::decode(&frag)
        {
            self.apply_share(&state);
        }
        if let Some(spec) = cli.view {
            self.apply_view_spec(&spec);
        }
        if let Some(iterations) = cli.iterations {
            self.auto_iterations = false;
            self.max_iterations = iterations;
        }
        if let Some(half_height) = cli.half_height {
            self.apply_half_height_spec(&half_height);
        }
        if let Some(position) = cli.position {
            self.apply_re_im_spec(&position);
        }
        if cli.de {
            self.de_coloring = true;
        }
        if cli.buddhabrot {
            self.mode = FractalMode::Buddhabrot;
        }
        if let Some(p) = cli.palette {
            self.buddha_palette = p.min(BUDDHA_PALETTE_NAMES.len() as u32 - 1);
            self.palette = p.min(PALETTE_NAMES.len() as u32 - 1);
        }
        self.export_path = cli.export_path;
    }

    /// Apply a view spec "re,im,half_height[,iterations]" (re/im are decimal,
    /// parsed at full precision). Used by the native debug env var.
    #[allow(dead_code)]
    pub fn apply_view_spec(&mut self, spec: &str) -> bool {
        let Some((view, iterations)) = parse_view_spec(spec) else {
            return false;
        };
        self.view = view;
        if let Some(v) = iterations {
            self.auto_iterations = false;
            self.max_iterations = v.clamp(32, MAX_REF_POINTS as u32 - 1);
        }
        true
    }

    /// Apply a half_height spec. Used by the native debug env var.
    #[allow(dead_code)]
    pub fn apply_half_height_spec(&mut self, spec: &str) -> bool {
        let Some(half_height) = parse_half_height_spec(spec) else {
            return false;
        };
        self.view.half_height = half_height;
        true
    }
    /// Apply a view spec "re,im" (re/im are decimal,
    /// parsed at full precision). Used by the native debug env var.
    #[allow(dead_code)]
    pub fn apply_re_im_spec(&mut self, spec: &str) -> bool {
        let Some((re, im)) = parse_re_im_spec(spec, self.view.precision_bits()) else {
            return false;
        };
        self.view.center_re = re;
        self.view.center_im = im;
        true
    }

    /// The current view (center + half-height). Used by headless animation
    /// to snapshot the start of a camera path.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn view_state(&self) -> &ViewState {
        &self.view
    }

    /// Jump straight to `view` for the next frame, keeping every other
    /// parameter (kind, colors, iteration count, ...) as-is. Used by
    /// headless animation to step through interpolated keyframes.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn set_view(&mut self, view: ViewState) {
        self.view = view;
    }

    /// Force `max_iterations` to auto-scale with zoom depth on every
    /// subsequent `compute_reference_blocking` call. Used by headless
    /// animation so iteration count keeps pace with the camera zooming in,
    /// the same way it does while dragging/zooming interactively.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn set_auto_iterations(&mut self, v: bool) {
        self.auto_iterations = v;
    }

    /// Set `max_iterations`.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn set_max_iterations(&mut self, i: u32) {
        self.auto_iterations = false;
        self.max_iterations = i;
    }

    /// Get `max_iterations`.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn max_iterations(&mut self) -> u32 {
        self.max_iterations
    }

    /// Jump to a preset Mandelbrot location: decimal center (parsed at the
    /// precision the zoom needs), half-height, and a fitting iteration count.
    fn go_to_place(&mut self, re: &str, im: &str, half_height: f64, iterations: u32) {
        let bits = precision_for(half_height);
        if let (Some(cre), Some(cim)) = (
            big_from_decimal_str(re, bits),
            big_from_decimal_str(im, bits),
        ) {
            self.mode = FractalMode::Mandelbrot;
            self.view = ViewState::with_center(cre, cim, half_height);
            // Presets carry a hand-tuned count; don't let the auto-scaler clobber it.
            self.auto_iterations = false;
            self.max_iterations = iterations.clamp(32, MAX_REF_POINTS as u32 - 1);
        }
    }

    /// Iteration count scaled to the current zoom depth, used while
    /// `auto_iterations` is on. Grows roughly linearly with zoom decades so deep
    /// zooms keep enough iterations to stay sharp instead of banding.
    fn auto_iteration_count(&self) -> u32 {
        let decades = self.view.magnification().log10().max(0.0);
        let iters = 400.0 + 900.0 * decades;
        (iters.round() as u32).clamp(200, MAX_REF_POINTS as u32 - 1)
    }

    /// Snapshot the current view as a shareable state.
    fn share_state(&self) -> ShareState {
        let sig_digits = sig_digits_for(self.view.precision_bits());
        ShareState {
            julia: matches!(self.mode, FractalMode::Julia),
            kind: self.kind,
            power: self.power,
            center_re: big_to_decimal_str(&self.view.center_re, sig_digits),
            center_im: big_to_decimal_str(&self.view.center_im, sig_digits),
            half_height: self.view.half_height,
            iterations: self.max_iterations,
            julia_c: self.julia_c,
            phoenix_p: self.phoenix_p,
            lambda_l: self.lambda_l,
            complex_power: self.complex_power,
            color_scale: self.color_scale,
            color_offset: self.color_offset,
            palette: self.palette,
            shadow_palette: self.shadow_palette,
        }
    }

    /// Restore a shared state into this app.
    fn apply_share(&mut self, s: &ShareState) {
        self.mode = if s.julia {
            FractalMode::Julia
        } else {
            FractalMode::Mandelbrot
        };
        self.kind = s.kind;
        self.power = s.power.clamp(2, 8);
        self.julia_c = s.julia_c;
        self.phoenix_p = s.phoenix_p;
        self.lambda_l = s.lambda_l;
        self.complex_power = s.complex_power;
        self.color_scale = s.color_scale;
        self.color_offset = s.color_offset;
        self.palette = (s.palette as usize).min(PALETTE_NAMES.len() - 1) as u32;
        self.shadow_palette =
            (s.shadow_palette as usize).min(SHADOW_PALETTE_NAMES.len() - 1) as u32;
        // The link carries an explicit iteration count; honor it rather than
        // letting the auto-scaler immediately overwrite it.
        self.auto_iterations = false;
        self.max_iterations = s.iterations.clamp(32, MAX_REF_POINTS as u32 - 1);
        let bits = precision_for(s.half_height);
        if let (Some(re), Some(im)) = (
            big_from_decimal_str(&s.center_re, bits),
            big_from_decimal_str(&s.center_im, bits),
        ) {
            self.view = ViewState::with_center(re, im, s.half_height);
        }
    }

    /// A full shareable URL. On web this is the page URL with a `#fragment`; on
    /// native (no page) it is just the fragment for pasting onto a deployment.
    fn share_url(&self) -> String {
        let fragment = self.share_state().encode();
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(w) = web_sys::window() {
                let loc = w.location();
                let origin = loc.origin().unwrap_or_default();
                let path = loc.pathname().unwrap_or_default();
                return format!("{origin}{path}#{fragment}");
            }
        }
        format!("#{fragment}")
    }

    /// Default view for a given set type and fractal kind. The Julia (dynamical)
    /// plane is centered on the origin for every kind; the parameter plane frames
    /// each kind's interesting region.
    fn default_view_for(mode: FractalMode, kind: FractalKind) -> ViewState {
        if mode == FractalMode::Julia {
            return ViewState::with_center(big_from_f64(0.0, 53), big_from_f64(0.0, 53), 1.5);
        }
        let (cr, ci, hh) = kind.default_set_view();
        ViewState::with_center(big_from_f64(cr, 53), big_from_f64(ci, 53), hh)
    }

    /// The request key for the current state. Its `iter` is the reference
    /// length to compute, which carries headroom over `max_iterations` (see
    /// [`reference_iterations`]).
    fn current_key(&self) -> RequestKey {
        RequestKey {
            center_re: self.view.center_re.clone(),
            center_im: self.view.center_im.clone(),
            half_height: self.view.half_height,
            julia: matches!(self.mode, FractalMode::Julia),
            julia_c: self.julia_c,
            phoenix_p: self.phoenix_p,
            lambda_l: self.lambda_l,
            iter: reference_iterations(self.max_iterations),
            kind: self.kind,
            power: self.power,
            complex_power: self.complex_power,
        }
    }

    /// Distance (complex units) the live view center has drifted from `key`.
    fn drift_from(&self, key: &RequestKey) -> f64 {
        let dre = (&self.view.center_re - &key.center_re).to_f64().value();
        let dim = (&self.view.center_im - &key.center_im).to_f64().value();
        (dre * dre + dim * dim).sqrt()
    }

    /// Whether the reference should be (re)computed: parameters changed, or the
    /// view drifted / zoomed far enough that the current reference no longer
    /// serves it well. Lambda in Set mode has a static fractal (doesn't depend
    /// on center), so we skip center drift checks but allow zoom precision updates.
    fn should_request(&self) -> bool {
        let Some(key) = &self.last_request else {
            return true;
        };
        if key.julia != matches!(self.mode, FractalMode::Julia)
            || key.julia_c != self.julia_c
            || key.phoenix_p != self.phoenix_p
            || key.lambda_l != self.lambda_l
            // The reference is computed with headroom, so it keeps serving
            // while auto-iterations creep up during a zoom (the shader clamps
            // to `max_iterations`); only recompute once it's too short, or
            // far longer than needed.
            || self.max_iterations > key.iter
            || self.max_iterations.saturating_mul(4) < key.iter
            || key.kind != self.kind
            || key.power != self.power
            || key.complex_power != self.complex_power
        {
            return true;
        }
        // Lambda in Set mode is a static fractal; don't trigger recompute on center drift.
        if self.kind == FractalKind::Lambda && matches!(self.mode, FractalMode::Mandelbrot) {
            // But still recompute on significant zoom changes for precision
            let ratio = self.view.half_height / key.half_height;
            return !(0.5..=2.0).contains(&ratio);
        }
        let ratio = self.view.half_height / key.half_height;
        self.drift_from(key) > 0.5 * self.view.half_height || !(0.5..=2.0).contains(&ratio)
    }

    /// Complex offset of the live view center from the reference center, in f32.
    fn dc_offset(&self) -> [f32; 2] {
        let dre = (&self.view.center_re - &self.ref_center_re)
            .to_f64()
            .value() as f32;
        let dim = (&self.view.center_im - &self.ref_center_im)
            .to_f64()
            .value() as f32;
        [dre, dim]
    }

    fn apply_reference(&mut self, points: Vec<[f32; 2]>, cre: Big, cim: Big, hh: f64) {
        self.reference = Arc::new(points);
        self.ref_center_re = cre;
        self.ref_center_im = cim;
        self.ref_half_height = hh;
        self.generation = self.generation.wrapping_add(1);
    }

    /// The current reference orbit, as uploaded to the GPU. Used by headless
    /// rendering to build its own `ExportRender` without going through
    /// `egui_wgpu`'s callback machinery.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn reference_points(&self) -> &[[f32; 2]] {
        &self.reference
    }

    /// The configured shadow-style lights, for headless export's `ExportRender`
    /// (which has no `FractalCallback` to source them from).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn lights(&self) -> &[Light] {
        &self.lights
    }

    /// Recompute the reference orbit when needed. Native: dispatch to a worker
    /// thread and pick up completed results. Web: compute inline.
    fn ensure_reference(&mut self) {
        if self.should_request() {
            let mut key = self.current_key();
            let precision = self.view.precision_bits();
            let max_iter = key.iter.min(MAX_REF_POINTS as u32 - 1);

            // Lambda in Set mode has a static fractal centered at origin.
            if key.kind == FractalKind::Lambda && !key.julia {
                key.center_re = big_from_f64(0.0, precision);
                key.center_im = big_from_f64(0.0, precision);
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                self.worker.request(crate::worker::RefRequest {
                    center_re: key.center_re.clone(),
                    center_im: key.center_im.clone(),
                    half_height: key.half_height,
                    julia: key.julia,
                    julia_c: key.julia_c,
                    max_iter,
                    precision,
                    kind: key.kind,
                    power: key.power,
                    phoenix_p: key.phoenix_p,
                    lambda_l: key.lambda_l,
                    complex_power: key.complex_power,
                });
                self.pending = true;
            }
            #[cfg(target_arch = "wasm32")]
            {
                let points = if key.julia {
                    let jr = big_from_f64(key.julia_c.0, precision);
                    let ji = big_from_f64(key.julia_c.1, precision);
                    compute_reference(
                        &key.center_re,
                        &key.center_im,
                        &jr,
                        &ji,
                        max_iter,
                        precision,
                        key.kind,
                        key.power,
                        key.phoenix_p,
                        key.lambda_l,
                        key.complex_power,
                    )
                } else {
                    compute_set_reference(
                        &key.center_re,
                        &key.center_im,
                        max_iter,
                        precision,
                        key.kind,
                        key.power,
                        key.phoenix_p,
                        key.lambda_l,
                        key.complex_power,
                    )
                };
                self.apply_reference(
                    points,
                    key.center_re.clone(),
                    key.center_im.clone(),
                    key.half_height,
                );
            }

            self.last_request = Some(key);
        }

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(res) = self.worker.try_take_latest() {
            self.apply_reference(res.points, res.center_re, res.center_im, res.half_height);
            self.pending = false;
        }
    }

    /// Compute the reference orbit for the current view synchronously, on the
    /// calling thread — unlike `ensure_reference`, which dispatches to the
    /// native worker (or, on wasm, computes inline but still runs once per
    /// frame poll). Used by headless rendering, which has no frame loop to
    /// poll a background result on and only ever needs one reference.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn compute_reference_blocking(&mut self) {
        if self.auto_iterations {
            self.max_iterations = self.auto_iteration_count();
        }
        let mut key = self.current_key();
        // One-shot render: no later frames for iteration headroom to serve.
        key.iter = self.max_iterations.min(MAX_REF_POINTS as u32 - 1);
        let precision = self.view.precision_bits();
        let max_iter = key.iter;

        // Lambda in Set mode has a static fractal centered at origin.
        if key.kind == FractalKind::Lambda && !key.julia {
            key.center_re = big_from_f64(0.0, precision);
            key.center_im = big_from_f64(0.0, precision);
        }

        let points = if key.julia {
            let jr = big_from_f64(key.julia_c.0, precision);
            let ji = big_from_f64(key.julia_c.1, precision);
            compute_reference(
                &key.center_re,
                &key.center_im,
                &jr,
                &ji,
                max_iter,
                precision,
                key.kind,
                key.power,
                key.phoenix_p,
                key.lambda_l,
                key.complex_power,
            )
        } else {
            compute_set_reference(
                &key.center_re,
                &key.center_im,
                max_iter,
                precision,
                key.kind,
                key.power,
                key.phoenix_p,
                key.lambda_l,
                key.complex_power,
            )
        };
        self.apply_reference(
            points,
            key.center_re.clone(),
            key.center_im.clone(),
            key.half_height,
        );
        self.last_request = Some(key);
    }

    pub(crate) fn make_uniforms(&self, aspect: f64) -> Uniforms {
        let (span_x, span_y) = self.view.span(aspect);
        Uniforms {
            span: [span_x as f32, span_y as f32],
            max_iter: self.max_iterations.min(MAX_REF_POINTS as u32 - 1),
            ref_len: self.reference.len() as u32,
            color_offset: self.color_offset,
            color_scale: self.color_scale,
            bailout_sq: BAILOUT_SQ,
            is_julia: matches!(self.mode, FractalMode::Julia) as u32,
            palette_id: self.palette,
            shadow_palette_id: self.shadow_palette,
            aa_level: if self.antialias { 2 } else { 1 },
            kind: self.kind as u32,
            power: self.power,
            dc_offset: self.dc_offset(),
            phoenix_p: [self.phoenix_p.0 as f32, self.phoenix_p.1 as f32],
            lambda_l: [self.lambda_l.0 as f32, self.lambda_l.1 as f32],
            complex_power: [self.complex_power.0 as f32, self.complex_power.1 as f32],
            de_coloring: (self.de_coloring | (self.rendering_mode > 0)) as u32,
            rendering_mode: if self.anim.camera_state > 0.0 {
                2
            } else {
                self.rendering_mode
            },
            camera_direction: self.camera.direction(self.anim.camera_state).to_array(),
            camera_inv_proj: self
                .camera
                .orthographic(self.anim.camera_state)
                .inverse()
                .to_cols_array(),
            screen_dim: self.screen_dim,
            light_count: gpu_lights(&self.lights).1,
            cm_coef: complex_binomials(self.complex_power),
            _pad: [0; _],
            _pad3: [0; _],
        }
    }

    /// Buddhabrot pass uniforms. Unlike `make_uniforms`, the view center is
    /// collapsed straight to f32 (no arbitrary-precision reference orbit) —
    /// Buddhabrot mode doesn't support deep zoom (see `fractal::buddhabrot`).
    fn make_buddhabrot_uniforms(&self, aspect: f64) -> BuddhabrotUniforms {
        let center = [
            self.view.center_re.to_f64().value() as f32,
            self.view.center_im.to_f64().value() as f32,
        ];
        BuddhabrotUniforms {
            center,
            half_height: self.view.half_height as f32,
            aspect: aspect as f32,
            phoenix_p: [self.phoenix_p.0 as f32, self.phoenix_p.1 as f32],
            lambda_l: [self.lambda_l.0 as f32, self.lambda_l.1 as f32],
            complex_power: [self.complex_power.0 as f32, self.complex_power.1 as f32],
            bailout_sq: BAILOUT_SQ,
            kind: self.kind as u32,
            power: self.power,
            r_cap: self.buddha_r_cap,
            g_cap: self.buddha_g_cap,
            b_cap: self.buddha_b_cap,
            seed: 0,                  // set by the callback's own dispatch counter
            samples_this_dispatch: 0, // set by the callback
            exposure: self.buddha_exposure,
            width: 0,           // set by the callback from size_px
            height: 0,          // set by the callback from size_px
            total_samples: 0.0, // tracked by the renderer across frames
            palette: self.buddha_palette,
            _pad0: 0,
        }
    }

    /// Render the current view to a PNG at `export_scale` × the on-screen size,
    /// then save it (native: file in cwd; web: browser download). Runs off the
    /// UI thread so a progress bar can animate; progress lands in `self.export`.
    fn do_export(&mut self, frame: &mut eframe::Frame) {
        if self.export.is_some() {
            return; // one export at a time
        }
        if self.mode == FractalMode::Buddhabrot {
            self.status = Some("PNG export isn't available in Buddhabrot mode yet".into());
            return;
        }
        let Some(rs) = frame.wgpu_render_state() else {
            self.status = Some("export unavailable (no wgpu backend)".into());
            return;
        };
        if self.reference.is_empty() {
            self.status = Some("still computing reference…".into());
            self.export_requested = true; // retry once the reference is ready
            return;
        }

        let scale = self.export_scale.max(1.0);
        let w = ((self.last_size_px.x * scale).round() as u32).clamp(16, MAX_EXPORT_DIM);
        let h = ((self.last_size_px.y * scale).round() as u32).clamp(16, MAX_EXPORT_DIM);
        let uniforms = self.make_uniforms(w as f64 / h as f64);

        let device = rs.device.clone();
        let queue = rs.queue.clone();
        let (pipeline, bind_group_layout, format) = {
            let guard = rs.renderer.read();
            let Some(renderer) = guard.callback_resources.get::<FractalRenderer>() else {
                self.status = Some("export unavailable".into());
                return;
            };
            renderer.export_handles(&device, &uniforms)
        };
        let reference = Arc::clone(&self.reference);
        let lights = self.lights.clone();

        let shared = Arc::new(Mutex::new(ExportShared {
            fraction: 0.0,
            phase: "Rendering",
            result: None,
        }));
        self.status = None;
        self.export = Some(Arc::clone(&shared));

        #[cfg(not(target_arch = "wasm32"))]
        {
            let name = self
                .export_path
                .clone()
                .unwrap_or_else(|| format!("fractal-{}.png", unix_timestamp()));
            std::thread::spawn(move || {
                let er = ExportRender::new(
                    &device,
                    &queue,
                    pipeline,
                    &bind_group_layout,
                    format,
                    w,
                    h,
                    uniforms,
                    reference.as_slice(),
                    &lights,
                );
                let sh = Arc::clone(&shared);
                let png =
                    crate::fractal::export_to_png_blocking(&device, &queue, &er, |phase, f| {
                        set_progress(&sh, phase, f)
                    });

                set_progress(&shared, "Saving", 0.98);
                let result = std::fs::write(&name, &png)
                    .map(|_| format!("saved {name} ({w}×{h})"))
                    .map_err(|e| format!("save failed: {e}"));
                finish_export(&shared, result);
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            // Progress budget: rendering fills [0, RENDER_END], encoding the rest.
            const RENDER_END: f32 = 0.6;
            wasm_bindgen_futures::spawn_local(async move {
                let er = ExportRender::new(
                    &device,
                    &queue,
                    pipeline,
                    &bind_group_layout,
                    format,
                    w,
                    h,
                    uniforms,
                    reference.as_slice(),
                    &lights,
                );

                // Render tile by tile, awaiting each submission so the browser
                // executes it and the UI can repaint between tiles.
                for t in 0..er.tiles {
                    er.render_tile(&device, &queue, t);
                    let (tx, rx) = futures_channel::oneshot::channel();
                    queue.on_submitted_work_done(move || {
                        let _ = tx.send(());
                    });
                    let _ = rx.await;
                    let done = (t + 1) as f32 / er.tiles as f32;
                    set_progress(&shared, "Rendering", RENDER_END * done);
                }
                er.copy_to_readback(&device, &queue);

                let (tx, rx) = futures_channel::oneshot::channel();
                er.readback()
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |res| {
                        let _ = tx.send(res);
                    });
                let _ = rx.await;

                set_progress(&shared, "Encoding", RENDER_END);
                let png = {
                    let data = er
                        .readback()
                        .slice(..)
                        .get_mapped_range()
                        .expect("map readback buffer");
                    let sh = Arc::clone(&shared);
                    crate::fractal::encode_png_with_progress(
                        &data,
                        er.width,
                        er.height,
                        er.padded_bpr,
                        er.swap_rb,
                        |f| set_progress(&sh, "Encoding", RENDER_END + (0.97 - RENDER_END) * f),
                    )
                };
                er.readback().unmap();

                set_progress(&shared, "Saving", 0.98);
                web_download_png(&png, "fractal.png");
                finish_export(&shared, Ok(format!("downloaded {w}×{h}")));
            });
        }
    }

    /// Pick up a finished export (setting the status line) and keep repainting
    /// while one is in flight so its progress bar animates.
    fn poll_export(&mut self, ctx: &egui::Context) {
        if let Some(shared) = &self.export {
            let done = shared.lock().unwrap().result.take();
            match done {
                Some(Ok(msg)) => {
                    self.status = Some(msg);
                    self.export = None;
                }
                Some(Err(e)) => {
                    self.status = Some(e);
                    self.export = None;
                }
                None => ctx.request_repaint(),
            }
        }
    }

    /// Floating top-left overlay with the panel toggle and fullscreen toggle.
    /// Always on top of the fractal, so both stay reachable when the controls
    /// panel is collapsed (the common case on a phone).
    fn overlay_buttons(&mut self, ui: &mut egui::Ui) {
        egui::Area::new(egui::Id::new("overlay_buttons"))
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(8.0, 8.0))
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style())
                    .shadow(egui::Shadow::NONE)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let panel_label = if self.controls_open { "Hide" } else { "Menu" };
                            if ui
                                .button(panel_label)
                                .on_hover_text("Show/hide the controls panel")
                                .clicked()
                            {
                                self.controls_open = !self.controls_open;
                            }
                            let fs_label = if self.fullscreen {
                                "Windowed"
                            } else {
                                "Fullscreen"
                            };
                            if ui
                                .button(fs_label)
                                .on_hover_text("Toggle fullscreen")
                                .clicked()
                            {
                                self.fullscreen = !self.fullscreen;
                                self.apply_fullscreen(ui.ctx());
                            }
                            if ui
                                .button("Help")
                                .on_hover_text("About this app, and mouse/touch controls")
                                .clicked()
                            {
                                self.help_open = !self.help_open;
                            }
                            // FPS readout. Monospace + fixed width so the number
                            // changing doesn't jitter the button row.
                            ui.add(
                                egui::Label::new(
                                    egui::RichText::new(format!("{:>3.0} FPS", self.fps))
                                        .monospace(),
                                )
                                .selectable(false),
                            )
                            .on_hover_text(
                                "Frames per second while rendering (interaction, \
                                 animation, export). Frozen when idle.",
                            );
                        });
                    });
            });
    }

    /// Floating bottom-left overlay: a single button that toggles the
    /// "Fractal Info" window. Kept separate from `overlay_buttons` (top-left)
    /// so it stays out of the way of the panel toggle / fullscreen controls,
    /// but is still reachable even when the controls panel is collapsed.
    fn info_button(&mut self, ui: &mut egui::Ui) {
        egui::Area::new(egui::Id::new("info_button"))
            .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(8.0, -8.0))
            .show(ui.ctx(), |ui| {
                egui::Frame::popup(ui.style())
                    .shadow(egui::Shadow::NONE)
                    .show(ui, |ui| {
                        if ui
                            .button("Fractal infos")
                            .on_hover_text("Show details about the current fractal")
                            .clicked()
                        {
                            self.info_open = !self.info_open;
                        }
                    });
            });
    }

    /// Window with details about what's currently on screen: formula, active
    /// per-kind constants, zoom depth, iteration count. Reads live state, so
    /// it stays correct as the user pans/zooms/switches kinds.
    fn info_window(&mut self, ctx: &egui::Context) {
        let mut open = self.info_open;
        egui::Window::new("Fractal Info")
            .id(egui::Id::new("info_window"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(8.0, -44.0))
            .show(ctx, |ui| {
                ui.label(egui::RichText::new(self.kind.label()).strong().heading());
                let mode_label = match self.mode {
                    FractalMode::Mandelbrot => {
                        "Mandelbrot mode — parameter space (c varies per pixel, z₀ = 0)"
                    }
                    FractalMode::Julia => {
                        "Julia mode — dynamical plane for a fixed c (z₀ varies per pixel)"
                    }
                    FractalMode::Buddhabrot => "Buddhabrot mode — orbit density (random c, z₀ = 0)",
                };
                ui.label(mode_label);
                ui.separator();

                ui.label(format!(
                    "formula: {}",
                    self.kind.formula(self.power, self.complex_power)
                ));
                if self.mode == FractalMode::Julia {
                    ui.label(format!("c = {:.6} {:+.6}i", self.julia_c.0, self.julia_c.1));
                }
                if self.kind == FractalKind::Phoenix {
                    ui.label(format!(
                        "p = {:.6} {:+.6}i",
                        self.phoenix_p.0, self.phoenix_p.1
                    ));
                }
                if self.kind == FractalKind::Lambda {
                    ui.label(format!(
                        "λ = {:.6} {:+.6}i",
                        self.lambda_l.0, self.lambda_l.1
                    ));
                }
                if self.kind == FractalKind::ComplexMultibrot {
                    ui.label(format!(
                        "power = {:.6} {:+.6}i",
                        self.complex_power.0, self.complex_power.1
                    ));
                }
                ui.separator();

                ui.label(self.kind.description());
            });
        self.info_open = open;
    }

    /// Help window: what the app does, plus a reference for mouse/touch and
    /// keyboard controls.
    fn help_window(&mut self, ctx: &egui::Context) {
        let mut open = self.help_open;
        egui::Window::new("Help")
            .id(egui::Id::new("help_window"))
            .open(&mut open)
            .collapsible(false)
            .default_width(360.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .max_height(480.0)
                    .show(ui, |ui| {
                        ui.heading("About");
                        ui.label(
                            "A deep-zoom fractal explorer. It renders the Mandelbrot set \
                             and several related fractals (Burning Ship, Tricorn, \
                             Multibrot, Complex Multibrot, Celtic, Perpendicular, Buffalo, \
                             Phoenix, Lambda).",
                        );
                        ui.add_space(4.0);
                        ui.label(
                            "Each fractals can be rendered in different modes: \n\
                             • Mandelbrot mode fixes z₀=0 and then for each pixel, set c as it's position \
                             in the complex plane. \n\
                             • Julia mode fixes c and instead varies the \
                             starting point z₀ across the plane. \n\
                             • Buddhabrot mode switches to a different, Monte-Carlo rendering of orbit density \
                             instead of the ordinary escape-time set.",
                        );
                        ui.separator();

                        ui.heading("Mouse & touch");
                        egui::Grid::new("help_mouse_grid")
                            .num_columns(2)
                            .spacing([12.0, 6.0])
                            .show(ui, |ui| {
                                ui.label("Drag");
                                ui.label("Pan the view");
                                ui.end_row();
                                ui.label("Scroll / trackpad");
                                ui.label("Zoom toward the cursor");
                                ui.end_row();
                                ui.label("Pinch (touch)");
                                ui.label("Zoom toward the gesture center");
                                ui.end_row();
                                ui.label("Two-finger drag (touch)");
                                ui.label("Pan the view");
                                ui.end_row();
                            });
                        ui.separator();

                        ui.heading("Keyboard");
                        egui::Grid::new("help_keyboard_grid")
                            .num_columns(2)
                            .spacing([12.0, 6.0])
                            .show(ui, |ui| {
                                ui.label("Arrow keys");
                                ui.label("Pan the view");
                                ui.end_row();
                                ui.label("Z / S");
                                ui.label("Zoom in / out toward the center");
                                ui.end_row();
                                ui.label("+ / -");
                                ui.label("Increase / decrease iterations");
                                ui.end_row();
                                ui.label("R");
                                ui.label("Reset to the default view");
                                ui.end_row();
                                ui.label("H");
                                ui.label("Toggle this Help window");
                                ui.end_row();
                                ui.label("I");
                                ui.label("Toggle the Info window");
                                ui.end_row();
                                ui.label("A");
                                ui.label("Toggle antialiasing (2×2)");
                                ui.end_row();
                            });
                        ui.separator();

                        ui.heading("Tips");
                        ui.label(
                            "• \"Copy link\" (in the panel) encodes the exact view so it \
                             can be reopened later or sent to someone else.",
                        );
                    });
            });
        self.help_open = open;
    }

    /// Push the desired fullscreen state to the platform.
    #[cfg(not(target_arch = "wasm32"))]
    fn apply_fullscreen(&mut self, ctx: &egui::Context) {
        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.fullscreen));
    }

    /// Push the desired fullscreen state to the browser. `request_fullscreen`
    /// must run inside a user gesture; the button click provides the transient
    /// activation that carries into this frame.
    #[cfg(target_arch = "wasm32")]
    fn apply_fullscreen(&mut self, _ctx: &egui::Context) {
        let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
            return;
        };
        if self.fullscreen {
            if let Some(el) = doc.document_element() {
                let _ = el.request_fullscreen();
            }
        } else {
            doc.exit_fullscreen();
        }
    }

    /// Refresh `self.fullscreen` from the real platform state, so the label is
    /// correct even when fullscreen is left by Esc/F11 or the browser UI.
    #[cfg(not(target_arch = "wasm32"))]
    fn sync_fullscreen(&mut self, ctx: &egui::Context) {
        if let Some(fs) = ctx.input(|i| i.viewport().fullscreen) {
            self.fullscreen = fs;
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn sync_fullscreen(&mut self, _ctx: &egui::Context) {
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            self.fullscreen = doc.fullscreen_element().is_some();
        }
    }

    /// Recompute the smoothed FPS. Counts frames over a ~0.5 s wall-clock window
    /// (using egui's monotonic `i.time`, which works on native and web) and
    /// divides once the window closes, so the readout is steady rather than
    /// jittering every frame. Only advances when egui repaints — i.e. while the
    /// app is doing work — so an idle app shows its last measured rate.
    fn update_fps(&mut self, ui: &egui::Ui) {
        let now = ui.input(|i| i.time);
        // Reset the window if time went backwards or hasn't started yet.
        if self.fps_window_start <= 0.0 || now < self.fps_window_start {
            self.fps_window_start = now;
            self.fps_frames = 0;
        }
        self.fps_frames += 1;
        let elapsed = now - self.fps_window_start;
        if elapsed >= 0.5 {
            self.fps = (self.fps_frames as f64 / elapsed) as f32;
            self.fps_frames = 0;
            self.fps_window_start = now;
        }
    }

    /// Advance any enabled animations by the frame's elapsed time, and request a
    /// repaint while active. Animations render at full resolution/AA (they do not
    /// trigger the interaction low-res pass).
    fn tick_animations(&mut self, ui: &egui::Ui) {
        // Julia c only matters in Julia mode; Phoenix p only for the Phoenix kind; Lambda λ only for Lambda kind.
        let julia_on = self.anim.julia && self.mode == FractalMode::Julia;
        let phoenix_on = self.anim.phoenix && self.kind == FractalKind::Phoenix;
        let lambda_on = self.anim.lambda && self.kind == FractalKind::Lambda;

        // Clamp dt so a stall (tab hidden, first frame) can't jump the animation.
        let dt = ui.input(|i| i.stable_dt as f64).clamp(0.0, 0.1);

        // Animate the 2D <-> 3D camera transition over a fixed duration with
        // smoothstep easing: it lands on exactly 0 or 1 (no asymptotic tail,
        // no snap), so the shader's mode switch (`camera_state > 0.0` in
        // `make_uniforms`) happens only once the camera is exactly top-down.
        const CAMERA_DURATION: f32 = 0.6; // seconds
        let target = if self.rendering_mode == 2 { 1.0 } else { 0.0 };
        let p = self.anim.camera_progress;
        if p != target {
            let step = dt as f32 / CAMERA_DURATION;
            self.anim.camera_progress = if target > p {
                (p + step).min(target)
            } else {
                (p - step).max(target)
            };
            ui.ctx().request_repaint();
        }
        let p = self.anim.camera_progress;
        self.anim.camera_state = p * p * (3.0 - 2.0 * p);

        if !(self.anim.color || self.anim.zoom || julia_on || phoenix_on || lambda_on) {
            return;
        }

        if self.anim.color {
            self.color_offset =
                (self.color_offset + self.anim.color_speed * dt as f32).rem_euclid(1.0);
        }
        if julia_on {
            self.anim.julia_angle += std::f64::consts::TAU * self.anim.julia_speed as f64 * dt;
            let (s, c) = self.anim.julia_angle.sin_cos();
            self.julia_c = (
                self.anim.julia_base.0 + self.anim.julia_radius * c,
                self.anim.julia_base.1 + self.anim.julia_radius * s,
            );
        }
        if phoenix_on {
            self.anim.phoenix_angle += std::f64::consts::TAU * self.anim.phoenix_speed as f64 * dt;
            let (s, c) = self.anim.phoenix_angle.sin_cos();
            self.phoenix_p = (
                self.anim.phoenix_base.0 + self.anim.phoenix_radius * c,
                self.anim.phoenix_base.1 + self.anim.phoenix_radius * s,
            );
        }
        let lambda_on = self.anim.lambda && self.kind == FractalKind::Lambda;
        if lambda_on {
            self.anim.lambda_angle += std::f64::consts::TAU * self.anim.lambda_speed as f64 * dt;
            let (s, c) = self.anim.lambda_angle.sin_cos();
            self.lambda_l = (
                self.anim.lambda_base.0 + self.anim.lambda_radius * c,
                self.anim.lambda_base.1 + self.anim.lambda_radius * s,
            );
        }
        if self.anim.zoom && self.anim.zoom_speed != 0.0 {
            let min_hh = DEFAULT_HALF_HEIGHT * 1.0e-26; // practical f32-perturbation depth
            let max_hh = DEFAULT_HALF_HEIGHT * 4.0;
            let factor = (-(self.anim.zoom_speed as f64) * dt).exp();
            let target = (self.view.half_height * factor).clamp(min_hh, max_hh);
            let f = target / self.view.half_height;
            if (f - 1.0).abs() > 1.0e-9 {
                self.view
                    .zoom_at_pixel(0.0, 0.0, self.last_size_px.y.max(1.0) as f64, f);
            }
        }

        ui.ctx().request_repaint();
    }

    fn controls_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Fractal Explorer");
        ui.separator();
        ui.add_space(4.);

        // Fractal formula. Switching kinds jumps to a sensible default view,
        // since interesting regions differ between fractals.
        let prev_kind = self.kind;
        egui::ComboBox::from_label("fractal")
            .selected_text(self.kind.label())
            .show_ui(ui, |ui| {
                for kind in FractalKind::ALL {
                    ui.selectable_value(&mut self.kind, kind, kind.label());
                }
            });
        if self.kind == FractalKind::Multibrot {
            ui.add(egui::Slider::new(&mut self.power, 2..=8).text("power"));
        }
        if self.kind == FractalKind::Phoenix {
            ui.horizontal(|ui| {
                ui.label("p =");
                ui.add(
                    egui::DragValue::new(&mut self.phoenix_p.0)
                        .speed(0.001)
                        .range(-2.0..=2.0),
                );
                ui.add(
                    egui::DragValue::new(&mut self.phoenix_p.1)
                        .speed(0.001)
                        .range(-2.0..=2.0),
                );
                ui.label("i");
            });
        }
        if self.kind == FractalKind::Lambda {
            ui.horizontal(|ui| {
                ui.label("λ =");
                ui.add(
                    egui::DragValue::new(&mut self.lambda_l.0)
                        .speed(0.001)
                        .range(-2.0..=2.0),
                );
                ui.add(
                    egui::DragValue::new(&mut self.lambda_l.1)
                        .speed(0.001)
                        .range(-2.0..=2.0),
                );
                ui.label("i");
            });
        }
        if self.kind == FractalKind::ComplexMultibrot {
            ui.horizontal(|ui| {
                ui.label("power =");
                ui.add(
                    egui::DragValue::new(&mut self.complex_power.0)
                        .speed(0.01)
                        .range(-8.0..=8.0),
                );
                ui.add(
                    egui::DragValue::new(&mut self.complex_power.1)
                        .speed(0.01)
                        .range(-8.0..=8.0),
                );
                ui.label("i");
            });
        }
        if self.kind != prev_kind {
            self.view = Self::default_view_for(self.mode, self.kind);
        }

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mode, FractalMode::Mandelbrot, "Set");
            ui.radio_value(&mut self.mode, FractalMode::Julia, "Julia");
            ui.radio_value(&mut self.mode, FractalMode::Buddhabrot, "Buddhabrot")
                .on_hover_text(
                    "Monte-Carlo density of escaping orbits instead of the ordinary \
                 escape-time set. Plain f32 view (no deep zoom); the image \
                 progressively sharpens while the view stays still.",
                );
        });

        if self.mode == FractalMode::Buddhabrot {
            self.buddhabrot_ui(ui);
            ui.add_space(4.);
            ui.separator();
            ui.add_space(4.);
            if ui.button("Reset view").clicked() {
                self.view = Self::default_view_for(self.mode, self.kind);
            }
            ui.add_space(8.0);
            ui.small("Drag to pan · scroll to zoom toward the cursor");
            return;
        }

        if self.mode == FractalMode::Julia && self.kind != FractalKind::Lambda {
            ui.horizontal(|ui| {
                ui.label("c =");
                ui.add(
                    egui::DragValue::new(&mut self.julia_c.0)
                        .speed(0.001)
                        .range(-2.0..=2.0),
                );
                ui.add(
                    egui::DragValue::new(&mut self.julia_c.1)
                        .speed(0.001)
                        .range(-2.0..=2.0),
                );
                ui.label("i");
            });

            if !JULIA_PRESETS[self.kind as usize].is_empty() {
                ui.label("places:");
                ui.horizontal_wrapped(|ui| {
                    for &(name, re, im, iterations, phoenix) in JULIA_PRESETS[self.kind as usize] {
                        if ui.small_button(name).clicked() {
                            self.julia_c = (re, im);
                            self.max_iterations = iterations.clamp(32, MAX_REF_POINTS as u32 - 1);

                            if let Some(phoenix) = phoenix {
                                self.phoenix_p = phoenix;
                            }
                        }
                    }
                });
            }
        }

        if self.mode == FractalMode::Mandelbrot && !SET_PRESETS[self.kind as usize].is_empty() {
            ui.label("places:");
            ui.horizontal_wrapped(|ui| {
                for &(name, re, im, half_height, iter, phoenix) in SET_PRESETS[self.kind as usize] {
                    if ui.small_button(name).clicked() {
                        self.go_to_place(re, im, half_height, iter);

                        if let Some(phoenix) = phoenix {
                            self.phoenix_p = phoenix;
                        }
                    }
                }
            });
        }

        ui.label("rendering:");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.rendering_mode, 0, "Classic");
            ui.radio_value(&mut self.rendering_mode, 1, "Shadow");
            ui.radio_value(&mut self.rendering_mode, 2, "3D");
        });

        ui.add_space(4.);
        ui.separator();
        ui.add_space(4.);
        ui.checkbox(&mut self.auto_iterations, "Auto iterations")
            .on_hover_text("Scale the iteration count with zoom depth so deep zooms stay sharp.");
        if self.auto_iterations {
            ui.label(format!("iterations: {} (auto)", self.max_iterations));
        } else {
            ui.add(
                egui::Slider::new(&mut self.max_iterations, 32..=100_000)
                    .text("iterations")
                    .logarithmic(true),
            );
        }
        ui.checkbox(&mut self.antialias, "Antialiasing (2×2)")
            .on_hover_text("Supersample each pixel for smoother edges (~4× slower).");
        if self.rendering_mode == 0 {
            ui.checkbox(&mut self.de_coloring, "Distance shading")
                .on_hover_text(
                    "Shade by distance to the set boundary (from the orbit derivative) \
                 for crisp filaments at deep zoom. Exact for the holomorphic kinds \
                 (Mandelbrot/Multibrot/Phoenix), approximate for the abs-based kinds \
                 (Burning Ship/Tricorn/Celtic/Perpendicular/Buffalo).",
                );
        }

        ui.add_space(4.);
        ui.separator();
        ui.add_space(4.);

        if self.rendering_mode == 0 {
            ui.add(
                egui::Slider::new(&mut self.color_scale, 0.01..=1.0)
                    .text("color scale")
                    .logarithmic(true),
            );
            ui.add(egui::Slider::new(&mut self.color_offset, 0.0..=1.0).text("color offset"));
            egui::ComboBox::from_label("palette")
                .selected_text(PALETTE_NAMES[self.palette as usize])
                .show_ui(ui, |ui| {
                    for (i, name) in PALETTE_NAMES.iter().enumerate() {
                        ui.selectable_value(&mut self.palette, i as u32, *name);
                    }
                });
        } else {
            egui::ComboBox::from_label("palette")
                .selected_text(SHADOW_PALETTE_NAMES[self.shadow_palette as usize])
                .show_ui(ui, |ui| {
                    for (i, name) in SHADOW_PALETTE_NAMES.iter().enumerate() {
                        ui.selectable_value(&mut self.shadow_palette, i as u32, *name);
                    }
                });
        }
        if self.rendering_mode > 0 && self.shadow_palette as usize == SHADOW_PALETTE_NAMES.len() - 1
        {
            ui.horizontal(|ui| {
                ui.label("lights:");
                if ui.button("+").clicked() {
                    self.lights.push(Light::default());
                }
            });
            egui::Grid::new("lights")
                .striped(true)
                .num_columns(1)
                .show(ui, |ui| {
                    self.lights.retain_mut(|light| {
                        let delete = !light.widget(ui);
                        ui.end_row();
                        delete
                    });
                });
        }
        ui.add_space(4.);
        ui.separator();
        ui.add_space(4.);

        ui.collapsing("Animation", |ui| {
            if self.rendering_mode == 0 {
                ui.checkbox(&mut self.anim.color, "Cycle colours")
                    .on_hover_text("Scroll the palette offset over time.");
                if self.anim.color {
                    ui.add(
                        egui::Slider::new(&mut self.anim.color_speed, 0.01..=2.0)
                            .text("cycles/s")
                            .logarithmic(true),
                    );
                }
            }

            ui.checkbox(&mut self.anim.zoom, "Auto-zoom")
                .on_hover_text("Continuously zoom toward the current center.");
            if self.anim.zoom {
                ui.add(
                    egui::Slider::new(&mut self.anim.zoom_speed, -2.0..=2.0).text("rate (+ = in)"),
                );
            }

            // Julia c only affects Julia mode; Phoenix p only the Phoenix kind.
            if self.mode == FractalMode::Julia {
                if ui.checkbox(&mut self.anim.julia, "Morph c").changed() && self.anim.julia {
                    self.anim.julia_base = self.julia_c; // orbit around the current c
                    self.anim.julia_angle = 0.0;
                }
                if self.anim.julia {
                    ui.add(
                        egui::Slider::new(&mut self.anim.julia_speed, 0.005..=0.5)
                            .text("c rev/s")
                            .logarithmic(true),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.anim.julia_radius, 0.005..=0.5)
                            .text("c radius")
                            .logarithmic(true),
                    );
                }
            }
            if self.kind == FractalKind::Phoenix {
                if ui.checkbox(&mut self.anim.phoenix, "Morph p").changed() && self.anim.phoenix {
                    self.anim.phoenix_base = self.phoenix_p;
                    self.anim.phoenix_angle = 0.0;
                }
                if self.anim.phoenix {
                    ui.add(
                        egui::Slider::new(&mut self.anim.phoenix_speed, 0.005..=0.5)
                            .text("p rev/s")
                            .logarithmic(true),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.anim.phoenix_radius, 0.005..=0.5)
                            .text("p radius")
                            .logarithmic(true),
                    );
                }
            }
            if self.kind == FractalKind::Lambda {
                if ui.checkbox(&mut self.anim.lambda, "Morph λ").changed() && self.anim.lambda {
                    self.anim.lambda_base = self.lambda_l;
                    self.anim.lambda_angle = 0.0;
                }
                if self.anim.lambda {
                    ui.add(
                        egui::Slider::new(&mut self.anim.lambda_speed, 0.005..=0.5)
                            .text("λ rev/s")
                            .logarithmic(true),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.anim.lambda_radius, 0.005..=0.5)
                            .text("λ radius")
                            .logarithmic(true),
                    );
                }
            }
        });

        ui.add_space(4.);
        ui.separator();
        ui.add_space(4.);
        // Editable center coordinates. Shown at full precision; parsed
        // losslessly on commit (Enter or focus loss). While a field is focused
        // we leave the user's text alone; otherwise we refresh it from the live
        // view, which panning and zooming keep changing.
        let bits = self.view.precision_bits();
        let sig = sig_digits_for(bits);

        ui.label("center re:");
        let re_resp = ui.add(
            egui::TextEdit::singleline(&mut self.center_re_edit)
                .desired_width(f32::INFINITY)
                .font(egui::TextStyle::Monospace),
        );
        if re_resp.lost_focus()
            && let Some(v) = big_from_decimal_str(
                &self.center_re_edit,
                parse_bits_for(&self.center_re_edit, bits),
            )
        {
            self.view.center_re = v;
            self.view.sync_precision();
        }
        if !re_resp.has_focus() {
            self.center_re_edit = big_to_decimal_str(&self.view.center_re, sig);
        }

        ui.label("center im:");
        let im_resp = ui.add(
            egui::TextEdit::singleline(&mut self.center_im_edit)
                .desired_width(f32::INFINITY)
                .font(egui::TextStyle::Monospace),
        );
        if im_resp.lost_focus()
            && let Some(v) = big_from_decimal_str(
                &self.center_im_edit,
                parse_bits_for(&self.center_im_edit, bits),
            )
        {
            self.view.center_im = v;
            self.view.sync_precision();
        }
        if !im_resp.has_focus() {
            self.center_im_edit = big_to_decimal_str(&self.view.center_im, sig);
        }

        ui.label("zoom:");
        let zoom_resp = ui.add(
            egui::TextEdit::singleline(&mut self.zoom_edit)
                .desired_width(f32::INFINITY)
                .font(egui::TextStyle::Monospace),
        );
        if zoom_resp.changed() {
            self.zoom_edited = true;
        }
        if zoom_resp.lost_focus() {
            if self.zoom_edited
                && let Ok(hh) = self.zoom_edit.trim().parse::<f64>()
                && hh > 0.0
                && hh.is_finite()
            {
                self.view.half_height = hh;
                self.view.sync_precision();
            }
            self.zoom_edited = false;
        }
        if !zoom_resp.has_focus() {
            self.zoom_edit = format_zoom(self.view.zoom());
        }
        ui.label(format!("reference: {} pts", self.reference.len()));
        ui.label(format!("precision: {} bits", self.view.precision_bits()));
        if self.pending {
            ui.colored_label(egui::Color32::LIGHT_YELLOW, "computing reference…");
        }

        ui.add_space(4.);
        ui.separator();
        ui.add_space(4.);
        let exporting = self.export.is_some();
        ui.horizontal(|ui| {
            if ui.button("Copy link").clicked() {
                let url = self.share_url();
                ui.ctx().copy_text(url);
                self.status = Some("link copied".into());
            }
            if ui
                .add_enabled(!exporting, egui::Button::new("Export PNG"))
                .clicked()
            {
                self.export_requested = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("export scale");
            ui.add(
                egui::DragValue::new(&mut self.export_scale)
                    .range(1.0..=16.0)
                    .speed(0.25)
                    .custom_formatter(|x, _| format!("x{:.1}", x)),
            );
            ui.label(format!(
                "= {}×{}",
                (self.last_size_px.x * self.export_scale) as u32,
                (self.last_size_px.y * self.export_scale) as u32,
            ));
        });
        if let Some(shared) = &self.export {
            let (fraction, phase) = {
                let s = shared.lock().unwrap();
                (s.fraction, s.phase)
            };
            ui.add(
                egui::ProgressBar::new(fraction)
                    .animate(true)
                    .text(format!("{phase} {:.0}%", fraction * 100.0)),
            );
        } else if let Some(status) = &self.status {
            ui.small(status);
        }

        ui.add_space(4.);
        ui.separator();
        ui.add_space(4.);
        if ui.button("Reset view").clicked() {
            self.view = Self::default_view_for(self.mode, self.kind);
        }
        ui.add_space(8.0);
        ui.small("Drag to pan · scroll to zoom toward the cursor");
    }

    /// Controls for Buddhabrot mode: nested iteration caps (Nebulabrot R/G/B
    /// coloring), exposure, and the progressive-accumulation toggle.
    fn buddhabrot_ui(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.add(
            egui::Slider::new(&mut self.buddha_r_cap, 5..=5_000)
                .text("red cap")
                .logarithmic(true),
        );
        ui.add(
            egui::Slider::new(&mut self.buddha_g_cap, 5..=20_000)
                .text("green cap")
                .logarithmic(true),
        );
        ui.add(
            egui::Slider::new(&mut self.buddha_b_cap, 5..=50_000)
                .text("blue cap")
                .logarithmic(true),
        );
        ui.add(
            egui::Slider::new(&mut self.buddha_exposure, 0.02..=50.0)
                .text("exposure")
                .logarithmic(true),
        );
        egui::ComboBox::from_label("colors")
            .selected_text(BUDDHA_PALETTE_NAMES[self.buddha_palette as usize])
            .show_ui(ui, |ui| {
                for (i, name) in BUDDHA_PALETTE_NAMES.iter().enumerate() {
                    ui.selectable_value(&mut self.buddha_palette, i as u32, *name);
                }
            });
        ui.checkbox(&mut self.buddha_accumulate, "Keep sampling")
            .on_hover_text("Dispatch a fresh batch of random samples every frame.");
        if self.view.magnification() > 1.0e5 {
            ui.colored_label(
                egui::Color32::LIGHT_YELLOW,
                "deep zoom isn't supported here (f32 precision only)",
            );
        }
        ui.small("PNG export isn't available in Buddhabrot mode yet.");
    }

    fn fractal_ui(&mut self, ui: &mut egui::Ui) {
        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
        if rect.width() < 1.0 || rect.height() < 1.0 {
            return;
        }
        let height_px = rect.height() as f64;
        let aspect = (rect.width() / rect.height()) as f64;
        self.last_size_px = rect.size();

        // Tracks whether the view actually moved this frame, so progressive
        // rendering can drop to a cheap low-res pass only while interacting.
        let mut interacted = false;

        // Advance time-based animations (colours / Julia c / Phoenix p / zoom).
        // These render at full resolution/AA — only real pan/zoom drops to the
        // cheap low-res pass, so `interacted` is left untouched here.
        self.tick_animations(ui);

        // Touch: pinch to zoom (toward the gesture center) and two-finger pan.
        // Takes precedence over single-finger drag while two fingers are down.
        // In 3D mode the same gestures orbit/dolly the raymarch camera
        // instead of panning/zooming the 2D fractal view.
        const ROT_SENS: f32 = 0.002; // radians per dragged pixel
        let multi_touch = ui.input(|i| i.multi_touch());
        if self.rendering_mode == 2 {
            if let Some(mt) = multi_touch {
                let t = mt.translation_delta;
                if t.x != 0.0 || t.y != 0.0 {
                    self.camera.rotate(t.x * ROT_SENS, -t.y * ROT_SENS);
                    interacted = true;
                }
                if mt.zoom_delta != 1.0 {
                    let ndc = (mt.center_pos.to_vec2() / rect.size()) * 2.;
                    let camera_ndc_pos = self.camera.orthographic(self.anim.camera_state).inverse()
                        * Vec4::new(ndc.x, ndc.y, 0., 1.);
                    let view_direction = self.camera.direction(self.anim.camera_state);

                    let z_move = camera_ndc_pos.z / view_direction.z;

                    let ndc_pos = camera_ndc_pos.xyz() + view_direction * -z_move;

                    let pos = egui::Vec2::new(ndc_pos.x / self.camera.aspect_ratio, ndc_pos.y)
                        * rect.size()
                        - rect.center().to_vec2();

                    self.view.zoom_at_pixel(
                        pos.x as f64,
                        pos.y as f64,
                        height_px,
                        1. / (mt.zoom_delta as f64),
                    );
                    interacted = true;
                }
                ui.ctx().request_repaint();
            } else if response.dragged() {
                let d = response.drag_delta();
                if d.x != 0.0 || d.y != 0.0 {
                    self.camera.rotate(d.x * ROT_SENS, -d.y * ROT_SENS);
                }
            }
        } else if let Some(mt) = multi_touch {
            let t = mt.translation_delta;
            if t.x != 0.0 || t.y != 0.0 {
                self.view.pan_pixels(t.x as f64, t.y as f64, height_px);
                interacted = true;
            }
            if mt.zoom_delta != 1.0 {
                let off = mt.center_pos - rect.center();
                // zoom_delta > 1 = fingers spreading = zoom in (smaller span).
                let factor = 1.0 / mt.zoom_delta as f64;
                self.view
                    .zoom_at_pixel(off.x as f64, off.y as f64, height_px, factor);
                interacted = true;
            }
            ui.ctx().request_repaint();
        } else if response.dragged() {
            // Single-finger / mouse drag pans.
            let d = response.drag_delta();
            if d.x != 0.0 || d.y != 0.0 {
                self.view.pan_pixels(d.x as f64, d.y as f64, height_px);
                interacted = true;
            }
        }

        // Mouse wheel / trackpad: zoom toward the cursor (2D), or dolly the
        // camera's ortho volume (3D).
        let (scroll_y, hover) = ui.input(|i| (i.smooth_scroll_delta.y, i.pointer.hover_pos()));
        if scroll_y != 0.0
            && let Some(pos) = hover
            && rect.contains(pos)
        {
            let factor = (-scroll_y as f64 * 0.0015).exp();
            let off = pos - rect.center();
            if self.rendering_mode == 2 {
                let ndc = (off / rect.size()) * 2.;
                let camera_ndc_pos = self.camera.orthographic(self.anim.camera_state).inverse()
                    * Vec4::new(ndc.x, ndc.y, 0., 1.);
                let view_direction = self.camera.direction(self.anim.camera_state);

                let z_move = camera_ndc_pos.z / view_direction.z;

                let ndc_pos = camera_ndc_pos.xyz() + view_direction * -z_move;

                let pos = egui::Vec2::new(ndc_pos.x / self.camera.aspect_ratio, ndc_pos.y)
                    * rect.size()
                    - rect.center().to_vec2();

                self.view
                    .zoom_at_pixel(pos.x as f64, pos.y as f64, height_px, factor);
            } else {
                self.view
                    .zoom_at_pixel(off.x as f64, off.y as f64, height_px, factor);
            }
            interacted = true;
            ui.ctx().request_repaint();
        }

        // Keyboard: arrows pan, z/s zoom in/out, +/- adjust iterations, R
        // resets the view, H/I toggle the Help/Info windows. In 3D mode,
        // ZQSD move the camera (forward/left/back/right), space/ctrl move it
        // up/down, and the arrow keys look around instead of panning.
        // Skipped while a text field (e.g. the center/zoom edit boxes) has
        // focus.
        if !ui.ctx().egui_wants_keyboard_input() {
            let dt = ui.input(|i| i.stable_dt as f64).clamp(0.0, 0.1);

            let not_modifier_ctrl = ui.input(|i| !i.modifiers.ctrl) || self.rendering_mode != 2;
            if self.rendering_mode == 2 {
                let (look_l, look_r, look_u, look_d) = ui.input(|i| {
                    (
                        i.key_down(egui::Key::ArrowLeft) && i.modifiers.ctrl,
                        i.key_down(egui::Key::ArrowRight) && i.modifiers.ctrl,
                        i.key_down(egui::Key::ArrowUp) && i.modifiers.ctrl,
                        i.key_down(egui::Key::ArrowDown) && i.modifiers.ctrl,
                    )
                });

                // Units/sec move speed and radians/sec look speed.
                const LOOK_SPEED: f32 = 0.5;

                let mut dyaw = 0.0f32;
                let mut dpitch = 0.0f32;
                if look_r {
                    dyaw += LOOK_SPEED * dt as f32;
                }
                if look_l {
                    dyaw -= LOOK_SPEED * dt as f32;
                }
                if look_u {
                    dpitch += LOOK_SPEED * dt as f32;
                }
                if look_d {
                    dpitch -= LOOK_SPEED * dt as f32;
                }
                if dyaw != 0.0 || dpitch != 0.0 {
                    self.camera.rotate(dyaw, dpitch);
                }

                if look_l || look_r || look_u || look_d {
                    ui.ctx().request_repaint();
                }
            }

            let (left, right, up, down, zoom_in, zoom_out) = ui.input(|i| {
                (
                    i.key_down(egui::Key::ArrowLeft) && not_modifier_ctrl,
                    i.key_down(egui::Key::ArrowRight) && not_modifier_ctrl,
                    i.key_down(egui::Key::ArrowUp) && not_modifier_ctrl,
                    i.key_down(egui::Key::ArrowDown) && not_modifier_ctrl,
                    i.key_down(egui::Key::Z),
                    i.key_down(egui::Key::S),
                )
            });

            // Pixels/sec pan speed — matches a brisk mouse drag regardless of
            // frame rate. See `pan_pixels`'s screen-space (+x right, +y down)
            // convention: Right/Down pan the *camera* right/down, which is
            // the opposite delta sign from a drag that would show the same
            // content (a drag grabs the canvas; these keys move the camera).
            const PAN_SPEED_PX: f64 = 700.0;
            let mut dx = 0.0;
            let mut dy = 0.0;
            if left {
                dx += PAN_SPEED_PX * dt;
            }
            if right {
                dx -= PAN_SPEED_PX * dt;
            }
            if down {
                dy -= PAN_SPEED_PX * dt;
            }
            if up {
                dy += PAN_SPEED_PX * dt;
            }
            if self.rendering_mode == 2 {
                let cos = self.camera.yaw.cos() as f64;
                let sin = self.camera.yaw.sin() as f64;
                (dx, dy) = (dx * cos + sin * dy, -dx * sin + cos * dy);
            }
            if dx != 0.0 || dy != 0.0 {
                self.view.pan_pixels(dx, dy, height_px);
                interacted = true;
            }

            // e-folds/sec, same scale as the auto-zoom animation.
            const ZOOM_SPEED: f64 = 1.0;
            if zoom_in != zoom_out {
                let rate = if zoom_in { ZOOM_SPEED } else { -ZOOM_SPEED };
                let factor = (-rate * dt).exp();
                self.view.zoom_at_pixel(0.0, 0.0, height_px, factor);
                interacted = true;
            }
            if left || right || up || down || zoom_in || zoom_out {
                ui.ctx().request_repaint();
            }

            if ui.input(|i| i.key_pressed(egui::Key::R)) {
                self.view = Self::default_view_for(self.mode, self.kind);
                self.camera = Camera::new();
                interacted = true;
            }
            if ui.input(|i| i.key_pressed(egui::Key::H)) {
                self.help_open = !self.help_open;
            }
            if ui.input(|i| i.key_pressed(egui::Key::I)) {
                self.info_open = !self.info_open;
            }
            if ui.input(|i| i.key_pressed(egui::Key::A)) {
                self.antialias = !self.antialias;
            }
            if ui.input(|i| i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals)) {
                self.auto_iterations = false;
                self.max_iterations = ((self.max_iterations as f64 * 1.25).round() as u32)
                    .clamp(32, MAX_REF_POINTS as u32 - 1);
            }
            if ui.input(|i| i.key_pressed(egui::Key::Minus)) {
                self.auto_iterations = false;
                self.max_iterations = ((self.max_iterations as f64 / 1.25).round() as u32)
                    .clamp(32, MAX_REF_POINTS as u32 - 1);
            }
        }

        if self.mode == FractalMode::Buddhabrot {
            // No reference orbit / perturbation machinery: iterate directly in
            // f32 from the live view. Progressive accumulation means this
            // needs its own continuous repaint, separate from the escape-time
            // interaction-driven one above.
            let ppp = ui.ctx().pixels_per_point();
            let size_px = [
                ((rect.width() * ppp).round() as u32).max(1),
                ((rect.height() * ppp).round() as u32).max(1),
            ];
            let uniforms = self.make_buddhabrot_uniforms(aspect);
            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect,
                BuddhabrotCallback {
                    uniforms,
                    accumulate: self.buddha_accumulate,
                    size_px,
                },
            ));
            if self.buddha_accumulate {
                ui.ctx().request_repaint();
            }
            return;
        }

        // Keep the iteration count matched to the zoom depth while auto is on.
        if self.auto_iterations {
            self.max_iterations = self.auto_iteration_count();
        }

        self.ensure_reference();

        // Poll the worker roughly every 30 ms while a reference is computing,
        // instead of spinning a full-speed repaint. Once ready, changed inputs
        // (or the initial draw) drive repaints on their own.
        let poll = std::time::Duration::from_millis(30);
        if self.reference.is_empty() {
            // Nothing to draw until the first reference orbit is ready.
            ui.ctx().request_repaint_after(poll);
            return;
        }
        if self.pending {
            ui.ctx().request_repaint_after(poll);
        }

        // Progressive rendering: while the user is actively panning/zooming (an
        // interaction within the last `INTERACT_SETTLE` seconds), render at a
        // fraction of the resolution with AA off so each frame is cheap, then let
        // it snap to full resolution once input settles. `i.time` is monotonic on
        // both native and web (avoids `Instant`, which isn't available on wasm).
        let now = ui.input(|i| i.time);
        if interacted {
            self.last_interact_time = now;
        }
        let interacting = now - self.last_interact_time < INTERACT_SETTLE;
        if interacting {
            // Ensure a frame fires once the settle window elapses, so the view
            // is re-rendered at full resolution even if no further input arrives.
            ui.ctx()
                .request_repaint_after(std::time::Duration::from_secs_f64(INTERACT_SETTLE));
        }

        // Cache-texture resolution: the widget size in physical pixels, divided
        // down while interacting (the linear blit upsamples it to the widget).
        let ppp = ui.ctx().pixels_per_point();
        let downscale = if interacting { INTERACT_DOWNSCALE } else { 1 };
        let mut size_px = [
            (((rect.width() * ppp).round() as u32) / downscale).max(1),
            (((rect.height() * ppp).round() as u32) / downscale).max(1),
        ];

        if self.rendering_mode == 2 {
            size_px = [size_px[0] * 2, size_px[1] * 2];
        }

        self.screen_dim = [rect.width(), rect.height()];
        self.camera.set_aspect_ratio(aspect as f32);
        let mut uniforms = self.make_uniforms(aspect);
        if interacting {
            uniforms.aa_level = 1; // supersampling is wasted on the low-res pass
        }
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            FractalCallback {
                uniforms,
                lights: gpu_lights(&self.lights).0,
                reference: Arc::clone(&self.reference),
                generation: self.generation,
                size_px,
            },
        ));
    }
}

impl eframe::App for FractalApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.poll_export(ui.ctx());
        self.update_fps(ui);
        // Track the real fullscreen state (e.g. the user pressing Esc/F11 or the
        // browser leaving fullscreen) so the toggle button label stays correct.
        self.sync_fullscreen(ui.ctx());

        // Cap the panel width so it never swallows a narrow (phone) screen, and
        // make it collapsible + scrollable so every parameter stays reachable.
        let panel_max = (ui.available_width() * 0.6).clamp(160.0, 340.0);
        let mut open = self.controls_open;
        egui::Panel::right("controls")
            .resizable(true)
            .default_size(panel_max.min(280.0))
            .max_size(panel_max)
            .show_collapsible(ui, &mut open, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.controls_ui(ui));
            });
        self.controls_open = open;

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.fractal_ui(ui));

        // Floating overlay, always reachable (even when the panel is collapsed):
        // toggle the panel and toggle fullscreen. Essential on a phone.
        self.overlay_buttons(ui);
        self.info_button(ui);
        self.info_window(ui.ctx());
        self.help_window(ui.ctx());

        if std::mem::take(&mut self.export_requested) {
            self.do_export(frame);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(&mut *self)
    }
}

/// Reference-orbit length to request for `max_iterations`: 1.5× headroom
/// (capped at the GPU buffer size). Auto-iterations grows with every zoom
/// frame, and without headroom each tiny increase re-ran the whole
/// high-precision orbit (plus a re-upload) on every frame of a zoom.
fn reference_iterations(max_iterations: u32) -> u32 {
    let cap = MAX_REF_POINTS as u32 - 1;
    (max_iterations.saturating_add(max_iterations / 2)).min(cap)
}

/// Complex binomial coefficients `C(p, k)` for k = 1..16, packed two per row
/// (odd k in `[0..2]`, even k in `[2..4]`) for `Uniforms::cm_coef`: the
/// Complex Multibrot delta series' coefficients, which only depend on the
/// power, so the shader doesn't rebuild them (with a complex division per
/// term) on every iteration of every pixel. Built up in f64 via
/// `C(p,k) = C(p,k-1) * (p - (k-1)) / k`.
fn complex_binomials(p: (f64, f64)) -> [[f32; 4]; 8] {
    let mut out = [[0.0f32; 4]; 8];
    let (mut cr, mut ci) = (1.0f64, 0.0f64); // C(p, 0)
    for k in 1..=16usize {
        // (cr + i ci) * ((p.0 - (k-1)) + i p.1) / k
        let (ar, ai) = (p.0 - (k - 1) as f64, p.1);
        let kf = k as f64;
        (cr, ci) = ((cr * ar - ci * ai) / kf, (cr * ai + ci * ar) / kf);
        let row = &mut out[(k - 1) / 2];
        let col = if k % 2 == 1 { 0 } else { 2 };
        row[col] = cr as f32;
        row[col + 1] = ci as f32;
    }
    out
}

/// Update an export's progress (phase label + fraction).
fn set_progress(shared: &Arc<Mutex<ExportShared>>, phase: &'static str, fraction: f32) {
    let mut s = shared.lock().unwrap();
    s.phase = phase;
    s.fraction = fraction;
}

/// Mark an export finished with its outcome.
fn finish_export(shared: &Arc<Mutex<ExportShared>>, result: Result<String, String>) {
    let mut s = shared.lock().unwrap();
    s.phase = "Done";
    s.fraction = 1.0;
    s.result = Some(result);
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(target_arch = "wasm32")]
fn web_location_hash() -> Option<String> {
    let hash = web_sys::window()?.location().hash().ok()?;
    if hash.trim_start_matches('#').is_empty() {
        None
    } else {
        Some(hash)
    }
}

#[cfg(target_arch = "wasm32")]
fn web_download_png(bytes: &[u8], filename: &str) {
    use wasm_bindgen::JsCast as _;

    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let array = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::new();
    parts.push(&array);
    let options = web_sys::BlobPropertyBag::new();
    options.set_type("image/png");
    let Ok(blob) = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &options) else {
        return;
    };
    let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) else {
        return;
    };
    if let Some(anchor) = document
        .create_element("a")
        .ok()
        .and_then(|el| el.dyn_into::<web_sys::HtmlAnchorElement>().ok())
    {
        anchor.set_href(&url);
        anchor.set_download(filename);
        anchor.click();
    }
    let _ = web_sys::Url::revoke_object_url(&url);
}
