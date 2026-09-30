use std::sync::{Arc, Mutex};

#[cfg(feature = "gui")]
use eframe::CreationContext;
#[cfg(feature = "gui")]
use eframe::egui_wgpu;
use glam::Vec4;
use glam::Vec4Swizzles;

use crate::camera::Camera;
#[cfg(not(target_arch = "wasm32"))]
use crate::cli::Cli;
use crate::fractal::bla::{self, BlaTable};
use crate::fractal::{
    BuddhabrotCallback, BuddhabrotRenderer, BuddhabrotUniforms, ExportRender, FractalCallback,
    FractalKind, FractalRenderer, MAX_REF_POINTS, RefOrbit, ShareState, Uniforms,
    compute_reference, compute_set_reference,
};
use crate::lights::{Light, gpu_lights};
use crate::view::parse_half_height_spec;
use crate::view::parse_re_im_spec;
use crate::view::{
    Big, DEFAULT_HALF_HEIGHT, MAX_PRECISION_BITS, Scale, ViewState, big_from_decimal_str,
    big_from_f64, big_to_decimal_str, deep_scale_exp, interpolate_view, needs_deep,
    parse_view_spec, precision_for,
};
#[cfg(not(target_arch = "wasm32"))]
use clap::Parser;

const BAILOUT_SQ: f32 = 1.0e6;

/// Pixel bailout |z|^2 for `kind`. For Multibrot z^p, one step from |z| = R
/// (with the reference within 2R, which rebasing guarantees) must stay a
/// finite f32, |z|^2 included: (2R)^(2p) <= 2^126, i.e. R^2 <= 2^(126/p - 2).
/// Otherwise inf - inf turns into NaN, which never compares above the bailout
/// and paints exterior pixels as interior. Unchanged for p <= 5; still well
/// above the escape radius (<= 2) at the maximum power.
fn bailout_sq(kind: FractalKind, power: u32) -> f32 {
    if kind == FractalKind::Multibrot {
        BAILOUT_SQ.min((126.0 / power.max(2) as f32 - 2.0).exp2())
    } else {
        BAILOUT_SQ
    }
}
/// Cap on exported image dimension (px), to stay within GPU texture limits.
const MAX_EXPORT_DIM: u32 = 8192 * 16;
/// Hard ceiling on the iteration count: the longest reference orbit the GPU
/// buffer holds (past it, the shader would read pixels as escaped).
const MAX_ITERATIONS: u32 = MAX_REF_POINTS as u32 - 1;

/// Binades of dc range a BLA table may overcover before it's rebuilt for a
/// zoomed-in view (it stays valid, but its radii shrink with dc_max).
const BLA_DC_SLACK: i32 = 2;

/// What the current BLA table was built for (see `FractalApp::bla_table`).
#[derive(Clone, Copy, PartialEq)]
struct BlaKey {
    generation: u64,
    kind: u32,
    on: bool,
    julia: bool,
    dc_log2: i32,
}

/// While the user is actively panning/zooming, the fractal is rendered into a
/// cache texture downscaled by this factor per axis (and with AA forced off), so
/// each interacting frame is cheap; the linear blit upsamples it to the widget.
/// A full-resolution render replaces it once input settles. 2 → quarter the
/// pixels (~4× faster); raise for more speed at the cost of more blur in motion.
/// Default of the Advanced "low resolution scale" setting (a power of 2).
const INTERACT_DOWNSCALE: u32 = 2;
/// Largest selectable interaction downscale, as log2 (16×).
const MAX_INTERACT_DOWNSCALE_LOG2: u32 = 4;
/// Seconds without pan/zoom input after which the view counts as settled and is
/// re-rendered at full resolution.
const INTERACT_SETTLE: f64 = 0.12;
/// Palette names; index maps to `palette_id` in the shader.
const PALETTE_NAMES: &[&str] = &["Amber", "Rainbow", "Ember", "Lime", "Grayscale"];
/// Shadow palette names; index maps to `shadow_palette_id` in the shader.
/// Append new entries: share links store the index.
const SHADOW_PALETTE_NAMES: &[&str] = &["Grayscale", "Red & Blue", "Custom lights", "Classic"];
/// Shadow palette lit by the user's `lights` list.
const SHADOW_PALETTE_CUSTOM_LIGHTS: u32 = 2;
/// Shadow palette that paints the classic escape-time palette, lit.
const SHADOW_PALETTE_CLASSIC: u32 = 3;
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

/// A reference-orbit computation detached from the app (see
/// `FractalApp::reference_job`), so it can run on any thread.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub(crate) struct RefJob {
    key: RequestKey,
    precision: usize,
    /// The frame's iteration count (auto-iterations resolved).
    max_iterations: u32,
}

#[cfg(not(target_arch = "wasm32"))]
impl RefJob {
    /// Orbit steps this job computes.
    pub(crate) fn steps(&self) -> u32 {
        self.key.iter
    }

    /// Working precision, in bits.
    pub(crate) fn precision(&self) -> usize {
        self.precision
    }

    /// This job with the orbit cut after `len` steps. `max_iterations` is
    /// unchanged: the shader treats the exhausted reference as an escape, so
    /// the render only differs if some pixel gets that far (headless probes
    /// for that before paying for the whole orbit).
    pub(crate) fn prefix(&self, len: u32) -> RefJob {
        let mut job = self.clone();
        job.key.iter = job.key.iter.min(len);
        job
    }

    /// Whether `points`, computed for this job, is the whole orbit `full`
    /// steps would give: not cut, or the reference escaped before the cut.
    pub(crate) fn is_whole(&self, full: u32, points: &RefOrbit) -> bool {
        self.key.iter >= full || points.len() <= self.key.iter as usize
    }

    /// Iterate the reference orbit at full precision (the expensive part).
    pub(crate) fn compute(&self) -> RefOrbit {
        let key = &self.key;
        let precision = self.precision;
        let morph = key.morph.map(|(k, w)| (k, w as f64));
        if key.julia {
            let jr = big_from_f64(key.julia_c.0, precision);
            let ji = big_from_f64(key.julia_c.1, precision);
            compute_reference(
                &key.center_re,
                &key.center_im,
                &jr,
                &ji,
                key.iter,
                precision,
                key.kind,
                key.power,
                key.phoenix_p,
                key.lambda_l,
                key.complex_power,
                morph,
            )
        } else {
            compute_set_reference(
                &key.center_re,
                &key.center_im,
                key.iter,
                precision,
                key.kind,
                key.power,
                key.phoenix_p,
                key.lambda_l,
                key.complex_power,
                morph,
            )
        }
    }
}

/// Parameters a reference orbit was (or will be) computed for. Used to decide
/// when the current reference is stale enough to recompute.
#[derive(Clone)]
struct RequestKey {
    center_re: Big,
    center_im: Big,
    half_height: Scale,
    julia: bool,
    julia_c: (f64, f64),
    phoenix_p: (f64, f64),
    lambda_l: (f64, f64),
    iter: u32,
    kind: FractalKind,
    power: u32,
    complex_power: (f64, f64),
    /// Kind-switch morph `(from_kind, weight)`, if one is running.
    morph: Option<(FractalKind, f32)>,
}

/// An in-progress kind-switch animation: the iteration formula is blended per
/// step from `from` to the current kind, `(1 - w)·f_kind + w·f_from`, while the
/// camera glides from `from_view` to the new kind's default view.
struct KindMorph {
    from: FractalKind,
    /// Linear progress in [0, 1]; eased with smoothstep.
    progress: f32,
    from_view: ViewState,
    to_view: ViewState,
    /// Whether the morph still drives the camera. Cleared as soon as the user
    /// pans/zooms, so they can take over mid-morph.
    camera: bool,
}

impl KindMorph {
    /// Smoothstep-eased progress.
    fn eased(&self) -> f32 {
        let p = self.progress.clamp(0.0, 1.0);
        p * p * (3.0 - 2.0 * p)
    }

    /// Weight of the old kind's formula: 1 at the start, 0 at the end.
    fn weight(&self) -> f32 {
        1.0 - self.eased()
    }
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

/// Drift of a complex constant around a circle in its plane (Julia `c`,
/// Phoenix `p`, Lambda `λ`).
#[derive(Clone)]
struct ConstOrbit {
    on: bool,
    /// Revolutions per second.
    speed: f32,
    /// Circle radius.
    radius: f64,
    /// Circle center, captured when the animation is enabled.
    base: (f64, f64),
    angle: f64,
}

impl Default for ConstOrbit {
    fn default() -> Self {
        Self {
            on: false,
            speed: 0.05,
            radius: 0.08,
            base: (0.0, 0.0),
            angle: 0.0,
        }
    }
}

impl ConstOrbit {
    /// Start orbiting around `current`.
    fn enable(&mut self, current: (f64, f64)) {
        self.base = current;
        self.angle = 0.0;
    }

    /// Advance by `dt` seconds and return the new value.
    fn step(&mut self, dt: f64) -> (f64, f64) {
        self.angle += std::f64::consts::TAU * self.speed as f64 * dt;
        let (s, c) = self.angle.sin_cos();
        (self.base.0 + self.radius * c, self.base.1 + self.radius * s)
    }

    /// Checkbox + speed/radius sliders; (re)centers the orbit on `current`
    /// when switched on.
    #[cfg(feature = "gui")]
    fn ui(&mut self, ui: &mut egui::Ui, name: &str, current: (f64, f64)) {
        if ui.checkbox(&mut self.on, format!("Morph {name}")).changed() && self.on {
            self.enable(current);
        }
        if self.on {
            ui.add(
                egui::Slider::new(&mut self.speed, 0.005..=0.5)
                    .text(format!("{name} rev/s"))
                    .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(&mut self.radius, 0.005..=0.5)
                    .text(format!("{name} radius"))
                    .logarithmic(true),
            );
        }
    }
}

/// Sine oscillation of one real parameter around a base value (used for each
/// component of the Complex Multibrot exponent, independently).
#[derive(Clone)]
struct AxisOsc {
    on: bool,
    /// Oscillation center, captured when the animation is enabled.
    base: f64,
    amplitude: f64,
    /// Oscillations per second.
    speed: f32,
    phase: f64,
}

impl Default for AxisOsc {
    fn default() -> Self {
        Self {
            on: false,
            base: 0.0,
            amplitude: 0.5,
            speed: 0.05,
            phase: 0.0,
        }
    }
}

impl AxisOsc {
    fn enable(&mut self, current: f64) {
        self.base = current;
        self.phase = 0.0;
    }

    fn step(&mut self, dt: f64) -> f64 {
        self.phase += std::f64::consts::TAU * self.speed as f64 * dt;
        self.base + self.amplitude * self.phase.sin()
    }

    #[cfg(feature = "gui")]
    fn ui(&mut self, ui: &mut egui::Ui, name: &str, current: f64) {
        if ui
            .checkbox(&mut self.on, format!("Animate {name}"))
            .changed()
            && self.on
        {
            self.enable(current);
        }
        if self.on {
            ui.add(
                egui::Slider::new(&mut self.amplitude, 0.01..=4.0)
                    .text(format!("{name} amplitude"))
                    .logarithmic(true),
            );
            ui.add(
                egui::Slider::new(&mut self.speed, 0.005..=0.5)
                    .text(format!("{name} Hz"))
                    .logarithmic(true),
            );
        }
    }
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
    julia: ConstOrbit,
    /// Drift the Phoenix distortion `p` around a circle.
    phoenix: ConstOrbit,
    /// Drift the Lambda distortion `λ` around a circle.
    lambda: ConstOrbit,
    /// Oscillate the Complex Multibrot exponent's real part.
    cpow_re: AxisOsc,
    /// Oscillate the Complex Multibrot exponent's imaginary part.
    cpow_im: AxisOsc,

    /// Continuously zoom toward the current center.
    zoom: bool,
    /// e-folds per second; positive zooms in, negative zooms out.
    zoom_speed: f32,

    /// Morph the iteration formula (and camera) when switching fractal kinds,
    /// instead of cutting straight to the new kind.
    kind_morph: bool,
    /// Kind-switch morph duration, in seconds.
    kind_morph_duration: f32,

    /// Step through every fractal kind in turn (each switch morphs if
    /// `kind_morph` is on).
    kind_cycle: bool,
    /// Seconds to rest on each kind before switching to the next.
    kind_cycle_hold: f32,
    /// Seconds spent on the current kind since the last cycle step.
    kind_cycle_timer: f32,

    /// Orbit the 3D camera: spin the yaw and bob the pitch.
    cam_orbit: bool,
    /// Yaw rate, degrees per second.
    cam_yaw_speed: f32,
    /// Pitch bob amplitude, degrees (0 = constant pitch).
    cam_pitch_amp: f32,
    /// Pitch bob frequency, Hz.
    cam_pitch_speed: f32,
    /// Pitch the bob oscillates around, captured when the orbit is enabled.
    cam_pitch_base: f32,
    cam_pitch_phase: f32,

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
            julia: ConstOrbit::default(),
            phoenix: ConstOrbit::default(),
            lambda: ConstOrbit::default(),
            cpow_re: AxisOsc::default(),
            cpow_im: AxisOsc::default(),
            zoom: false,
            zoom_speed: 0.5,
            kind_morph: true,
            kind_morph_duration: 1.5,
            kind_cycle: false,
            kind_cycle_hold: 3.0,
            kind_cycle_timer: 0.0,
            cam_orbit: false,
            cam_yaw_speed: 15.0,
            cam_pitch_amp: 0.0,
            cam_pitch_speed: 0.05,
            cam_pitch_base: 0.0,
            cam_pitch_phase: 0.0,
            camera_progress: 0.,
            camera_state: 0.,
        }
    }
}

/// Parse a "re,im" pair of plain `f64`s (per-kind constants on the CLI).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn parse_complex_pair(spec: &str) -> Option<(f64, f64)> {
    let (re, im) = spec.split_once(',')?;
    Some((re.trim().parse().ok()?, im.trim().parse().ok()?))
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
    /// When set, `color_scale` stretches one palette cycle across the `ci`
    /// range on screen (read back from the GPU, `FractalRenderer::take_ci_range`)
    /// and the palette starts at its low end, `ci_lo`.
    auto_color_scale: bool,
    /// Lowest on-screen `ci` from the last auto fit; 0 while auto is off.
    /// `color_offset` is relative to it (see `effective_color_offset`).
    ci_lo: f32,
    palette: u32,
    shadow_palette: u32,
    /// Supersample each pixel 2×2 for smoother edges (costs ~4× fragment work).
    antialias: bool,
    /// Skip perturbation steps with the BLA table (`fractal::bla`).
    use_bla: bool,
    /// Distance-estimation shading: darkens toward the set boundary using the
    /// orbit derivative, giving crisp filaments at deep zoom instead of speckle.
    de_coloring: bool,
    // Use shadow coloring
    // Use 3D raymarching rendering
    rendering_mode: u32,
    /// Per-axis scale of the 3D view's height-field texture relative to the
    /// widget: higher shows sharper, more distant terrain but costs GPU time
    /// and memory.
    render_scale_3d: f32,

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
    /// Kind-switch morph in progress, if any.
    morph: Option<KindMorph>,

    /// Smoothed frames-per-second, recomputed each ~0.5 s window. Only advances
    /// while the app is actually repainting (interaction / animation / export);
    /// idle frames aren't forced, so a frozen value means "nothing to render".
    fps: f32,
    /// Frames counted in the current FPS window, and its start time (`i.time`).
    fps_frames: u32,
    fps_window_start: f64,

    /// Reference orbit (`Z_n` as f32 pairs) for the current view.
    reference: Arc<RefOrbit>,
    /// Bumped whenever `reference` is replaced, so the GPU re-uploads it.
    generation: u64,
    /// BLA table for `reference` (see `bla_table`), what it was built for,
    /// and its own generation for the GPU upload.
    bla: Arc<BlaTable>,
    bla_key: Option<BlaKey>,
    bla_generation: u64,
    /// Center + zoom the current `reference` was computed at (may differ
    /// slightly from the live view; the shader compensates via `dc_offset`).
    ref_center_re: Big,
    ref_center_im: Big,
    ref_half_height: Scale,
    /// Kind and kind-switch morph the current `reference` was computed with.
    /// The shader iterates with these (not the live kind/morph) so its delta
    /// formula always matches the orbit, even while the worker lags a frame
    /// behind — otherwise a kind switch flashes the new kind, unblended, for
    /// the frame(s) before the morphed reference arrives. `None` until the
    /// first reference lands.
    ref_kind: Option<FractalKind>,
    ref_morph: Option<(FractalKind, f32)>,
    /// Parameters of the most recent reference request (drift baseline / dedupe).
    last_request: Option<RequestKey>,

    #[cfg(not(target_arch = "wasm32"))]
    worker: crate::worker::RefWorker,
    /// A reference computation is in flight (native async worker).
    pending: bool,

    /// PNG export resolution multiplier over the on-screen size.
    export_scale: f32,
    /// Last on-screen fractal size in physical pixels (for export sizing).
    #[cfg(feature = "gui")]
    last_size_px: egui::Vec2,
    /// egui time (seconds) of the most recent pan/zoom. While recent (within
    /// `INTERACT_SETTLE`) the fractal renders downscaled for smooth interaction.
    last_interact_time: f64,
    /// log2 of the per-axis downscale applied while panning/zooming
    /// (`1 << interact_downscale_log2`); higher = faster but blurrier in motion.
    interact_downscale_log2: u32,
    /// Set when the user requests a PNG export (handled after the panels draw).
    export_requested: bool,
    /// Progress/handle for an in-flight PNG export, if any.
    export: Option<Arc<Mutex<ExportShared>>>,
    /// Output path for `--export-path` (native CLI only); falls back to a
    /// timestamped name when unset.
    #[cfg(not(target_arch = "wasm32"))]
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
fn format_zoom(m: Scale) -> String {
    format!("{m:.4}")
}

/// Precision (bits) to parse a typed center at: at least what the current zoom
/// needs, but enough to preserve every digit the user pasted, so a deep
/// coordinate entered while zoomed out is not truncated. Capped like `view`.
fn parse_bits_for(s: &str, min_bits: usize) -> usize {
    let digits = s.chars().filter(char::is_ascii_digit).count();
    let from_input = (digits as f64 * std::f64::consts::LOG2_10).ceil() as usize + 16;
    min_bits.max(from_input).min(MAX_PRECISION_BITS)
}

impl FractalApp {
    #[cfg(feature = "gui")]
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
            auto_color_scale: false,
            ci_lo: 0.0,
            palette: 0,
            shadow_palette: 0,
            antialias: false,
            use_bla: true,
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
            morph: None,
            fps: 0.0,
            fps_frames: 0,
            fps_window_start: 0.0,
            reference: Arc::new(RefOrbit::default()),
            generation: 0,
            bla: Arc::new(BlaTable::empty()),
            bla_key: None,
            bla_generation: 0,
            ref_center_re,
            ref_center_im,
            ref_half_height,
            ref_kind: None,
            ref_morph: None,
            last_request: None,
            #[cfg(not(target_arch = "wasm32"))]
            worker: crate::worker::RefWorker::spawn(),
            pending: false,
            export_scale: 2.0,
            render_scale_3d: 2.0,
            #[cfg(feature = "gui")]
            last_size_px: egui::vec2(1280.0, 720.0),
            last_interact_time: -1.0e9,
            interact_downscale_log2: INTERACT_DOWNSCALE.trailing_zeros(),
            export_requested: false,
            export: None,
            #[cfg(not(target_arch = "wasm32"))]
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
                self.power = p.clamp(2, 20);
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
            // Start fully in 3D rather than transitioning in from top-down
            // (headless renders a single frame, with no transition to run).
            let p = if self.rendering_mode == 2 { 1.0 } else { 0.0 };
            self.anim.camera_progress = p;
            self.anim.camera_state = p;
        }
        if cli.yaw.is_some() || cli.pitch.is_some() {
            let yaw = cli.yaw.map_or(self.camera.yaw, f32::to_radians);
            let pitch = cli.pitch.map_or(self.camera.pitch, f32::to_radians);
            self.camera.set_angles(yaw, pitch);
            self.camera.rotate(0.0, 0.0); // wrap yaw
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
        // After --share so it can override the link's exponent.
        if let Some(cp) = cli.complex_power.as_deref().and_then(parse_complex_pair) {
            self.complex_power = cp;
        }
        if let Some(spec) = cli.view {
            self.apply_view_spec(&spec);
        }
        if let Some(iterations) = cli.iterations {
            self.auto_iterations = false;
            self.max_iterations = iterations.clamp(32, MAX_ITERATIONS);
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
        if cli.antialias {
            self.antialias = true;
        }
        if cli.no_bla {
            self.use_bla = false;
        }
        if cli.buddhabrot {
            self.mode = FractalMode::Buddhabrot;
        }
        if let Some(p) = cli.palette {
            self.buddha_palette = p.min(BUDDHA_PALETTE_NAMES.len() as u32 - 1);
            self.palette = p.min(PALETTE_NAMES.len() as u32 - 1);
        }
        // After --share, which turns it off.
        if cli.auto_color_scale {
            self.auto_color_scale = true;
        }
        if let Some(scale) = cli.color_scale {
            self.color_scale = scale.clamp(1e-4, 1.0);
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
            self.max_iterations = v.clamp(32, MAX_ITERATIONS);
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
    /// subsequent `reference_job` call. Used by headless
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
        self.max_iterations = i.clamp(32, MAX_ITERATIONS);
    }

    /// Per-kind constants `(julia_c, phoenix_p, lambda_l, complex_power)`.
    /// Used by headless animation to snapshot their start values.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn constants(&self) -> [(f64, f64); 4] {
        [
            self.julia_c,
            self.phoenix_p,
            self.lambda_l,
            self.complex_power,
        ]
    }

    /// Set the per-kind constants, in the order `constants` returns them.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn set_constants(&mut self, [c, p, l, cp]: [(f64, f64); 4]) {
        self.julia_c = c;
        self.phoenix_p = p;
        self.lambda_l = l;
        self.complex_power = cp;
    }

    /// 3D camera `(yaw, pitch)`, radians.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn camera_angles(&self) -> (f32, f32) {
        (self.camera.yaw, self.camera.pitch)
    }

    /// Set the 3D camera angles (radians; yaw unwrapped, see
    /// `Camera::set_angles`).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn set_camera_angles(&mut self, yaw: f32, pitch: f32) {
        self.camera.set_angles(yaw, pitch);
    }

    /// Size the 3D camera and raymarcher for a `width`×`height` render with
    /// no window (they normally follow the widget rect each frame).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn set_output_size(&mut self, width: u32, height: u32) {
        self.screen_dim = [width as f32, height as f32];
        self.camera.set_aspect_ratio(width as f32 / height as f32);
    }

    /// The current fractal kind.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn kind(&self) -> FractalKind {
        self.kind
    }

    /// Render a fraction `t` in [0, 1] of the way through a kind morph from
    /// `from` to `to`: the per-step formula blend, without touching the
    /// camera. `t >= 1` (or `from == to`) is plain `to`.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn set_kind_morph(&mut self, from: FractalKind, to: FractalKind, t: f64) {
        self.kind = to;
        self.morph = (from != to && t < 1.0).then(|| {
            // `KindMorph` eases its progress with smoothstep; invert that so
            // the blend follows `t` (already eased or not by the caller).
            let e = t.clamp(0.0, 1.0);
            let progress = 0.5 - ((1.0 - 2.0 * e).asin() / 3.0).sin();
            KindMorph {
                from,
                progress: progress as f32,
                from_view: self.view.clone(),
                to_view: self.view.clone(),
                camera: false,
            }
        });
    }

    /// Get `max_iterations`.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn max_iterations(&mut self) -> u32 {
        self.max_iterations
    }

    /// Jump to a preset Mandelbrot location: decimal center (parsed at the
    /// precision the zoom needs), half-height, and a fitting iteration count.
    fn go_to_place(&mut self, re: &str, im: &str, half_height: f64, iterations: u32) {
        let half_height = Scale::from_f64(half_height);
        let bits = precision_for(half_height);
        if let (Some(cre), Some(cim)) = (
            big_from_decimal_str(re, bits),
            big_from_decimal_str(im, bits),
        ) {
            self.mode = FractalMode::Mandelbrot;
            self.view = ViewState::with_center(cre, cim, half_height);
            self.morph = None;
            // Presets carry a hand-tuned count; don't let the auto-scaler clobber it.
            self.auto_iterations = false;
            self.max_iterations = iterations.clamp(32, MAX_ITERATIONS);
        }
    }

    /// Iteration count scaled to the current zoom depth, used while
    /// `auto_iterations` is on. Grows roughly linearly with zoom decades so deep
    /// zooms keep enough iterations to stay sharp instead of banding.
    fn auto_iteration_count(&self) -> u32 {
        let decades = self.view.magnification_log10().max(0.0);
        let iters = 400.0 + 900.0 * decades;
        (iters.round() as u32).clamp(200, MAX_ITERATIONS)
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
            color_offset: self.effective_color_offset(),
            palette: self.palette,
            shadow_palette: self.shadow_palette,
        }
    }

    /// Restore a shared state into this app.
    fn apply_share(&mut self, s: &ShareState) {
        self.morph = None;
        self.mode = if s.julia {
            FractalMode::Julia
        } else {
            FractalMode::Mandelbrot
        };
        self.kind = s.kind;
        self.power = s.power.clamp(2, 20);
        self.julia_c = s.julia_c;
        self.phoenix_p = s.phoenix_p;
        self.lambda_l = s.lambda_l;
        self.complex_power = s.complex_power;
        self.color_scale = s.color_scale;
        self.color_offset = s.color_offset;
        // Like the iteration count: the link's explicit scale wins.
        self.auto_color_scale = false;
        self.ci_lo = 0.0;
        self.palette = (s.palette as usize).min(PALETTE_NAMES.len() - 1) as u32;
        self.shadow_palette =
            (s.shadow_palette as usize).min(SHADOW_PALETTE_NAMES.len() - 1) as u32;
        // The link carries an explicit iteration count; honor it rather than
        // letting the auto-scaler immediately overwrite it.
        self.auto_iterations = false;
        self.max_iterations = s.iterations.clamp(32, MAX_ITERATIONS);
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
            return ViewState::with_center(
                big_from_f64(0.0, 53),
                big_from_f64(0.0, 53),
                Scale::from_f64(1.5),
            );
        }
        let (cr, ci, hh) = kind.default_set_view();
        ViewState::with_center(
            big_from_f64(cr, 53),
            big_from_f64(ci, 53),
            Scale::from_f64(hh),
        )
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
            morph: self.morph.as_ref().map(|m| (m.from, m.weight())),
        }
    }

    /// Distance (complex units) the live view center has drifted from `key`.
    /// Distance of the live center from `key`'s, in units of the live
    /// half-height (measured at that scale, so it works past f64's range).
    fn drift_from(&self, key: &RequestKey) -> f64 {
        let hh = self.view.half_height;
        let k = -hh.exponent() as isize;
        let dre = ((&self.view.center_re - &key.center_re) << k).to_f64();
        let dim = ((&self.view.center_im - &key.center_im) << k).to_f64();
        (dre * dre + dim * dim).sqrt() / hh.scaled_f64(-hh.exponent())
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
            || self.max_iterations.min(MAX_ITERATIONS) > key.iter
            || self.max_iterations.saturating_mul(4) < key.iter
            || key.kind != self.kind
            || key.power != self.power
            || key.complex_power != self.complex_power
            || key.morph != self.morph.as_ref().map(|m| (m.from, m.weight()))
        {
            return true;
        }
        // Lambda in Set mode is a static fractal; don't trigger recompute on
        // center drift. (Not while morphing: the other kind's formula does
        // depend on the center.)
        if self.kind == FractalKind::Lambda
            && matches!(self.mode, FractalMode::Mandelbrot)
            && self.morph.is_none()
        {
            // But still recompute on significant zoom changes for precision
            let ratio = self.view.half_height.ratio(key.half_height);
            return !(0.5..=2.0).contains(&ratio);
        }
        let ratio = self.view.half_height.ratio(key.half_height);
        self.drift_from(key) > 0.5 || !(0.5..=2.0).contains(&ratio)
    }

    /// Complex offset of the live view center from the reference center, in
    /// units of `2^scale_exp` (shifted exactly in `Big`, so it doesn't
    /// underflow f64 at deep zooms).
    fn dc_offset(&self, scale_exp: i32) -> (f64, f64) {
        let k = -scale_exp as isize;
        let dre = ((&self.view.center_re - &self.ref_center_re) << k).to_f64();
        let dim = ((&self.view.center_im - &self.ref_center_im) << k).to_f64();
        (dre, dim)
    }

    /// Binary exponent of the deep (rescaled) view scale, or 0 for the plain
    /// f32 path (see `Uniforms::scale_exp`). Deep when a pixel of a render
    /// `height_px` tall is too small for f32 (`needs_deep`), or when the
    /// reference orbit holds points only the deep pipeline can read.
    fn scale_exp(&self, height_px: f64) -> i32 {
        if needs_deep(self.view.half_height, height_px) || self.reference.has_scaled() {
            deep_scale_exp(self.view.half_height)
        } else {
            0
        }
    }

    fn apply_reference(
        &mut self,
        points: RefOrbit,
        cre: Big,
        cim: Big,
        hh: Scale,
        kind: FractalKind,
        morph: Option<(FractalKind, f32)>,
    ) {
        self.reference = Arc::new(points);
        self.ref_kind = Some(kind);
        self.ref_morph = morph;
        self.ref_center_re = cre;
        self.ref_center_im = cim;
        self.ref_half_height = hh;
        self.generation = self.generation.wrapping_add(1);
    }

    /// The current reference orbit, as uploaded to the GPU. Used by headless
    /// rendering to build its own `ExportRender` without going through
    /// `egui_wgpu`'s callback machinery.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn reference_points(&self) -> &RefOrbit {
        &self.reference
    }

    /// Whether BLA is on (headless builds its own tables per frame).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn use_bla(&self) -> bool {
        self.use_bla
    }

    /// The BLA table for rendering `u` from the current reference, rebuilt
    /// when the reference changed or the view's dc range left what the table
    /// covers. A table built for a larger dc range is still valid, just
    /// jumps less, so zooming in only rebuilds every `BLA_DC_SLACK` binades.
    fn bla_table(&mut self, u: &Uniforms) -> Arc<BlaTable> {
        let want = BlaKey {
            generation: self.generation,
            kind: u.kind,
            on: self.use_bla && bla::applies(u),
            julia: u.is_julia != 0,
            dc_log2: bla::dc_max_log2(u),
        };
        let fresh = self.bla_key.as_ref().is_some_and(|k| {
            k.generation == want.generation
                && k.kind == want.kind
                && k.on == want.on
                && k.julia == want.julia
                && (!want.on
                    || want.julia
                    || (want.dc_log2 <= k.dc_log2 && want.dc_log2 + BLA_DC_SLACK >= k.dc_log2))
        });
        if !fresh {
            self.bla = Arc::new(bla::for_uniforms(&self.reference, u, self.use_bla));
            self.bla_key = Some(want);
            self.bla_generation = self.bla_generation.wrapping_add(1);
        }
        Arc::clone(&self.bla)
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
            if key.kind == FractalKind::Lambda && !key.julia && key.morph.is_none() {
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
                    morph: key.morph,
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
                        key.morph.map(|(k, w)| (k, w as f64)),
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
                        key.morph.map(|(k, w)| (k, w as f64)),
                    )
                };
                self.apply_reference(
                    points,
                    key.center_re.clone(),
                    key.center_im.clone(),
                    key.half_height,
                    key.kind,
                    key.morph,
                );
            }

            self.last_request = Some(key);
        }

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(res) = self.worker.try_take_latest() {
            self.apply_reference(
                res.points,
                res.center_re,
                res.center_im,
                res.half_height,
                res.kind,
                res.morph,
            );
            self.pending = false;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn auto_iterations(&self) -> bool {
        self.auto_iterations
    }

    /// Snapshot everything the reference orbit for the current view depends
    /// on, as a self-contained job that can be computed on another thread
    /// (headless animation computes many frames' orbits in parallel). Also
    /// applies auto-iterations.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn reference_job(&mut self) -> RefJob {
        if self.auto_iterations {
            self.max_iterations = self.auto_iteration_count();
        }
        let mut key = self.current_key();
        // One-shot render: no later frames for iteration headroom to serve.
        key.iter = self.max_iterations.min(MAX_REF_POINTS as u32 - 1);
        let precision = self.view.precision_bits();

        // Lambda in Set mode has a static fractal centered at origin.
        if key.kind == FractalKind::Lambda && !key.julia && key.morph.is_none() {
            key.center_re = big_from_f64(0.0, precision);
            key.center_im = big_from_f64(0.0, precision);
        }
        RefJob {
            key,
            precision,
            max_iterations: self.max_iterations,
        }
    }

    /// Install the orbit computed for `job` (from `reference_job`) as the
    /// current reference, along with the iteration count it was made for.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn finish_reference(&mut self, job: RefJob, points: RefOrbit) {
        self.max_iterations = job.max_iterations;
        let key = job.key;
        self.apply_reference(
            points,
            key.center_re.clone(),
            key.center_im.clone(),
            key.half_height,
            key.kind,
            key.morph,
        );
        self.last_request = Some(key);
    }

    /// The palette offset the shaders get: `color_offset`, shifted so the
    /// palette starts at the auto fit's lowest on-screen `ci`.
    fn effective_color_offset(&self) -> f32 {
        (self.color_offset - self.ci_lo * self.color_scale).rem_euclid(1.0)
    }

    /// Pick up the GPU's latest on-screen `ci` range and refit the colour
    /// scale to it (auto colour scale).
    #[cfg(feature = "gui")]
    fn poll_ci_stats(&mut self, ctx: &egui::Context, frame: &eframe::Frame) {
        let Some(rs) = frame.wgpu_render_state() else {
            return;
        };
        let mut guard = rs.renderer.write();
        let Some(renderer) = guard.callback_resources.get_mut::<FractalRenderer>() else {
            return;
        };
        let range = renderer.take_ci_range(&rs.device);
        // Also one frame after a readback lands: the view may have changed
        // while it was in flight, and that frame histograms it again.
        if renderer.ci_stats_pending() || range.is_some() {
            ctx.request_repaint();
        }
        if self.auto_color_scale
            && let Some(range) = range
        {
            self.apply_ci_range(range);
        }
    }

    /// Fit one palette cycle across the `ci` range `[lo, hi]` (auto colour
    /// scale).
    pub(crate) fn apply_ci_range(&mut self, (lo, hi): (f32, f32)) {
        self.color_scale = (1.0 / (hi - lo).max(1e-3)).clamp(1e-4, 1.0);
        self.ci_lo = lo;
    }

    /// The colour scale and the auto fit's `ci_lo`, to restore with
    /// [`Self::set_color_fit`].
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn color_fit(&self) -> (f32, f32) {
        (self.color_scale, self.ci_lo)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn set_color_fit(&mut self, (scale, ci_lo): (f32, f32)) {
        self.color_scale = scale;
        self.ci_lo = ci_lo;
    }

    /// Whether the auto colour scale applies to the current render.
    pub(crate) fn auto_color_active(&self) -> bool {
        self.auto_color_scale && self.uses_classic_palette()
    }

    /// Whether the classic escape-time palette (and its scale / offset /
    /// palette controls) is in use: always in classic mode, and in shadow/3D
    /// modes under the "Classic" shading palette.
    fn uses_classic_palette(&self) -> bool {
        self.rendering_mode == 0 || self.shadow_palette == SHADOW_PALETTE_CLASSIC
    }

    /// The rendering mode the shaders should use this frame: 3D for as long
    /// as the 2D <-> 3D camera transition is in flight (the raymarcher, and
    /// the DE heights it reads, stay on until the camera is back top-down),
    /// otherwise the selected mode.
    fn effective_rendering_mode(&self) -> u32 {
        if self.anim.camera_state > 0.0 {
            2
        } else {
            self.rendering_mode
        }
    }

    /// 3D-mode zoom toward the screen point `off` (points from the widget
    /// center): unproject it through the camera onto the z = 0 fractal plane,
    /// then zoom the 2D view about the matching fractal-texture pixel.
    #[cfg(feature = "gui")]
    fn zoom_3d_at(&mut self, off: egui::Vec2, rect: egui::Rect, height_px: f64, factor: f64) {
        let ndc = (off / rect.size()) * 2.;
        let camera_ndc_pos = self
            .camera
            .orthographic(self.anim.camera_state, self.render_scale_3d)
            .inverse()
            * Vec4::new(ndc.x, ndc.y, 0., 1.);
        let view_direction = self.camera.direction(self.anim.camera_state);

        let z_move = camera_ndc_pos.z / view_direction.z;

        let ndc_pos = camera_ndc_pos.xyz() + view_direction * -z_move;

        let pos = egui::Vec2::new(ndc_pos.x / self.camera.aspect_ratio, ndc_pos.y) * rect.size()
            - rect.center().to_vec2();

        self.view
            .zoom_at_pixel(pos.x as f64, pos.y as f64, height_px, factor);
    }

    /// `height_px` is the full-resolution render height (it decides whether
    /// the view needs the deep pipeline); pass the same value during
    /// interaction's downscaled pass, so the pipeline doesn't flip.
    pub(crate) fn make_uniforms(&self, aspect: f64, height_px: f64) -> Uniforms {
        let (span_x, span_y) = self.view.span(aspect);
        let mode = self.effective_rendering_mode();
        // Deep views upload the geometry pre-multiplied by 2^-E (exact).
        let scale_exp = self.scale_exp(height_px);
        let (dc_re, dc_im) = self.dc_offset(scale_exp);
        Uniforms {
            span: [
                span_x.scaled_f64(-scale_exp) as f32,
                span_y.scaled_f64(-scale_exp) as f32,
            ],
            max_iter: self.max_iterations.min(MAX_REF_POINTS as u32 - 1),
            ref_len: self.reference.len() as u32,
            color_offset: self.effective_color_offset(),
            color_scale: self.color_scale,
            bailout_sq: bailout_sq(self.ref_kind.unwrap_or(self.kind), self.power),
            is_julia: matches!(self.mode, FractalMode::Julia) as u32,
            palette_id: self.palette,
            shadow_palette_id: self.shadow_palette,
            aa_level: if self.antialias { 2 } else { 1 },
            kind: self.ref_kind.unwrap_or(self.kind) as u32,
            power: self.power,
            morph_from: self.ref_morph.map_or(0, |(k, _)| k as u32),
            dc_offset: [dc_re as f32, dc_im as f32],
            phoenix_p: [self.phoenix_p.0 as f32, self.phoenix_p.1 as f32],
            lambda_l: [self.lambda_l.0 as f32, self.lambda_l.1 as f32],
            complex_power: [self.complex_power.0 as f32, self.complex_power.1 as f32],
            de_coloring: (self.de_coloring || mode > 0) as u32,
            rendering_mode: mode,
            camera_direction: self.camera.direction(self.anim.camera_state).to_array(),
            morph_w: self.ref_morph.map_or(0.0, |(_, w)| w),
            camera_inv_proj: self
                .camera
                .orthographic(self.anim.camera_state, self.render_scale_3d)
                .inverse()
                .to_cols_array(),
            screen_dim: self.screen_dim,
            light_count: gpu_lights(&self.lights).1,
            cm_coef: complex_binomials(self.complex_power),
            scale_exp,
        }
    }

    /// Buddhabrot pass uniforms. Unlike `make_uniforms`, the view center is
    /// collapsed straight to f32 (no arbitrary-precision reference orbit) —
    /// Buddhabrot mode doesn't support deep zoom (see `fractal::buddhabrot`).
    fn make_buddhabrot_uniforms(&self, aspect: f64) -> BuddhabrotUniforms {
        let center = [
            self.view.center_re.to_f64() as f32,
            self.view.center_im.to_f64() as f32,
        ];
        BuddhabrotUniforms {
            center,
            half_height: self.view.half_height.to_f64() as f32,
            aspect: aspect as f32,
            phoenix_p: [self.phoenix_p.0 as f32, self.phoenix_p.1 as f32],
            lambda_l: [self.lambda_l.0 as f32, self.lambda_l.1 as f32],
            complex_power: [self.complex_power.0 as f32, self.complex_power.1 as f32],
            bailout_sq: bailout_sq(self.kind, self.power),
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
    #[cfg(feature = "gui")]
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
        let uniforms = self.make_uniforms(w as f64 / h as f64, h as f64);

        let device = rs.device.clone();
        let queue = rs.queue.clone();
        let handles = {
            let guard = rs.renderer.read();
            let Some(renderer) = guard.callback_resources.get::<FractalRenderer>() else {
                self.status = Some("export unavailable".into());
                return;
            };
            renderer.export_handles(&device, &uniforms, false)
        };
        let reference = Arc::clone(&self.reference);
        let use_bla = self.use_bla;
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
                let bla = bla::for_uniforms(&reference, &uniforms, use_bla);
                let er = ExportRender::new(
                    &device, &queue, &handles, w, h, uniforms, &reference, &bla, &lights,
                );
                let sh = Arc::clone(&shared);
                let png = crate::fractal::export_to_png_blocking(
                    &device,
                    &queue,
                    &er,
                    png::Compression::Fast,
                    |phase, f| set_progress(&sh, phase, f),
                );

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
                let bla = bla::for_uniforms(&reference, &uniforms, use_bla);
                let er = ExportRender::new(
                    &device, &queue, &handles, w, h, uniforms, &reference, &bla, &lights,
                );

                // Render band by band, awaiting each submission so the browser
                // executes it and the UI can repaint between bands. (No
                // `Instant` here to time them, so bands keep the safe height.)
                let rows = er.band_sizer().rows();
                for y0 in (0..er.height).step_by(rows as usize) {
                    er.render_band(&device, &queue, y0, y0 + rows);
                    let (tx, rx) = futures_channel::oneshot::channel();
                    queue.on_submitted_work_done(move || {
                        let _ = tx.send(());
                    });
                    let _ = rx.await;
                    let done = (y0 + rows).min(er.height) as f32 / er.height as f32;
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
                        png::Compression::Fast,
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
    #[cfg(feature = "gui")]
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
    #[cfg(feature = "gui")]
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
    #[cfg(feature = "gui")]
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
    #[cfg(feature = "gui")]
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
    #[cfg(feature = "gui")]
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
    #[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
    fn apply_fullscreen(&mut self, ctx: &egui::Context) {
        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.fullscreen));
    }

    /// Push the desired fullscreen state to the browser. `request_fullscreen`
    /// must run inside a user gesture; the button click provides the transient
    /// activation that carries into this frame.
    #[cfg(all(feature = "gui", target_arch = "wasm32"))]
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
    #[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
    fn sync_fullscreen(&mut self, ctx: &egui::Context) {
        if let Some(fs) = ctx.input(|i| i.viewport().fullscreen) {
            self.fullscreen = fs;
        }
    }

    #[cfg(all(feature = "gui", target_arch = "wasm32"))]
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
    #[cfg(feature = "gui")]
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
    #[cfg(feature = "gui")]
    fn tick_animations(&mut self, ui: &egui::Ui) {
        // Julia c only matters in Julia mode; Phoenix p only for the Phoenix kind; Lambda λ only for Lambda kind.
        let julia_on = self.anim.julia.on && self.mode == FractalMode::Julia;
        let phoenix_on = self.anim.phoenix.on && self.kind == FractalKind::Phoenix;
        let lambda_on = self.anim.lambda.on && self.kind == FractalKind::Lambda;
        let cmulti = self.kind == FractalKind::ComplexMultibrot;
        let cpow_on = cmulti && (self.anim.cpow_re.on || self.anim.cpow_im.on);
        let cam_on = self.anim.cam_orbit && self.rendering_mode == 2;
        let cycle_on = self.anim.kind_cycle && self.mode != FractalMode::Buddhabrot;

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

        // Kind-switch morph: advance the per-iteration formula blend, and glide
        // the camera to the new kind's default view unless the user took over.
        if let Some(m) = &mut self.morph {
            m.progress += dt as f32 / self.anim.kind_morph_duration.max(0.05);
            if m.camera {
                self.view = interpolate_view(&m.from_view, &m.to_view, m.eased() as f64);
            }
            if m.progress >= 1.0 {
                if m.camera {
                    self.view = m.to_view.clone();
                }
                self.morph = None;
            }
            ui.ctx().request_repaint();
        }

        if !(self.anim.color
            || self.anim.zoom
            || julia_on
            || phoenix_on
            || lambda_on
            || cpow_on
            || cam_on
            || cycle_on)
        {
            return;
        }

        if self.anim.color {
            self.color_offset =
                (self.color_offset + self.anim.color_speed * dt as f32).rem_euclid(1.0);
        }
        if julia_on {
            self.julia_c = self.anim.julia.step(dt);
        }
        if phoenix_on {
            self.phoenix_p = self.anim.phoenix.step(dt);
        }
        if lambda_on {
            self.lambda_l = self.anim.lambda.step(dt);
        }
        if cmulti && self.anim.cpow_re.on {
            self.complex_power.0 = self.anim.cpow_re.step(dt).clamp(-8.0, 8.0);
        }
        if cmulti && self.anim.cpow_im.on {
            self.complex_power.1 = self.anim.cpow_im.step(dt).clamp(-8.0, 8.0);
        }
        if cam_on {
            let dyaw = self.anim.cam_yaw_speed.to_radians() * dt as f32;
            self.anim.cam_pitch_phase +=
                std::f32::consts::TAU * self.anim.cam_pitch_speed * dt as f32;
            // With no bob, leave pitch alone so it stays draggable mid-orbit.
            let dpitch = if self.anim.cam_pitch_amp > 0.0 {
                self.anim.cam_pitch_base
                    + self.anim.cam_pitch_amp.to_radians() * self.anim.cam_pitch_phase.sin()
                    - self.camera.pitch
            } else {
                0.0
            };
            // `rotate` wraps yaw and clamps pitch.
            self.camera.rotate(dyaw, dpitch);
        }
        if cycle_on && self.morph.is_none() {
            self.anim.kind_cycle_timer += dt as f32;
            if self.anim.kind_cycle_timer >= self.anim.kind_cycle_hold {
                self.anim.kind_cycle_timer = 0.0;
                let prev = self.kind;
                let i = FractalKind::ALL
                    .iter()
                    .position(|&k| k == prev)
                    .unwrap_or(0);
                self.kind = FractalKind::ALL[(i + 1) % FractalKind::ALL.len()];
                self.switch_kind(prev);
            }
        }
        if self.anim.zoom && self.anim.zoom_speed != 0.0 {
            let max_hh = Scale::from_f64(DEFAULT_HALF_HEIGHT * 4.0);
            let factor = (-(self.anim.zoom_speed as f64) * dt).exp();
            let target = self
                .view
                .half_height
                .mul_f64(factor)
                .clamp(Scale::MIN, max_hh);
            let f = target.ratio(self.view.half_height);
            if (f - 1.0).abs() > 1.0e-9 {
                self.view
                    .zoom_at_pixel(0.0, 0.0, self.last_size_px.y.max(1.0) as f64, f);
            }
        }

        ui.ctx().request_repaint();
    }

    /// React to `self.kind` having just changed from `prev`: jump (or, with
    /// kind morphing on, glide) to the new kind's default view.
    fn switch_kind(&mut self, prev: FractalKind) {
        let to_view = Self::default_view_for(self.mode, self.kind);
        // Buddhabrot has its own pipeline without the blended formula, so
        // it keeps the instant switch.
        self.morph =
            (self.anim.kind_morph && self.mode != FractalMode::Buddhabrot).then(|| KindMorph {
                from: prev,
                progress: 0.0,
                from_view: self.view.clone(),
                to_view: to_view.clone(),
                camera: true,
            });
        if self.morph.is_none() {
            self.view = to_view;
        }
    }

    #[cfg(feature = "gui")]
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
            ui.add(egui::Slider::new(&mut self.power, 2..=20).text("power"));
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
            self.switch_kind(prev_kind);
        }

        let prev_mode = self.mode;
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
        if self.mode != prev_mode {
            self.morph = None;
        }

        if self.mode == FractalMode::Buddhabrot {
            self.buddhabrot_ui(ui);
            ui.add_space(4.);
            ui.separator();
            ui.add_space(4.);
            if ui.button("Reset view").clicked() {
                self.view = Self::default_view_for(self.mode, self.kind);
                self.morph = None;
            }
            ui.add_space(8.0);
            ui.small("Drag to pan · scroll to zoom toward the cursor");
            return;
        }

        if self.mode == FractalMode::Julia {
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
                            self.max_iterations = iterations.clamp(32, MAX_ITERATIONS);

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
        if self.rendering_mode == 2 {
            let old_scale = self.render_scale_3d;
            ui.add(egui::Slider::new(&mut self.render_scale_3d, 1.0..=4.0).text("3D render scale"))
                .on_hover_text(
                    "Resolution multiplier of the 3D height field. Higher shows more \
                 distant detail but costs GPU time and memory.",
                );
            // The 3D camera zooms in by the render scale (`Camera::orthographic`):
            // zoom the view out by the same ratio so the fractal keeps its
            // on-screen size and the extra texels become surrounding terrain.
            if self.render_scale_3d != old_scale {
                let factor = (self.render_scale_3d / old_scale) as f64;
                self.view
                    .zoom_at_pixel(0.0, 0.0, self.last_size_px.y.max(1.0) as f64, factor);
            }
        }

        ui.add_space(4.);
        ui.separator();
        ui.add_space(4.);
        ui.checkbox(&mut self.auto_iterations, "Auto iterations")
            .on_hover_text("Scale the iteration count with zoom depth so deep zooms stay sharp.");
        if self.auto_iterations {
            ui.label(format!("iterations: {} (auto)", self.max_iterations));
        } else {
            // Dragging stays within the slider's range, but a typed value
            // isn't clamped to it: only to what the reference buffer can hold.
            ui.add(
                egui::Slider::new(&mut self.max_iterations, 32..=100_000)
                    .text("iterations")
                    .logarithmic(true)
                    .clamping(egui::SliderClamping::Never),
            );
            self.max_iterations = self.max_iterations.clamp(32, MAX_ITERATIONS);
        }
        ui.checkbox(&mut self.antialias, "Antialiasing (2×2)")
            .on_hover_text("Supersample each pixel for smoother edges (~4× slower).");
        ui.checkbox(&mut self.use_bla, "Skip iterations (BLA)")
            .on_hover_text(
                "Bivariate linear approximation: jump over runs of iterations where \
             every pixel follows the reference linearly. Much faster at deep zoom.",
            );
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

        if self.rendering_mode != 0 {
            egui::ComboBox::from_label("shading")
                .selected_text(SHADOW_PALETTE_NAMES[self.shadow_palette as usize])
                .show_ui(ui, |ui| {
                    for (i, name) in SHADOW_PALETTE_NAMES.iter().enumerate() {
                        ui.selectable_value(&mut self.shadow_palette, i as u32, *name);
                    }
                });
        }
        if self.uses_classic_palette() {
            let was_auto = self.auto_color_scale;
            ui.checkbox(&mut self.auto_color_scale, "Auto color scale")
                .on_hover_text(
                    "Stretch one palette cycle across the escape-time range visible on screen.",
                );
            if was_auto && !self.auto_color_scale {
                // Fold the fit's start into the offset so nothing jumps.
                self.color_offset = self.effective_color_offset();
                self.ci_lo = 0.0;
            }
            if self.auto_color_scale {
                ui.label(format!("color scale: {:.4} (auto)", self.color_scale));
            } else {
                ui.add(
                    egui::Slider::new(&mut self.color_scale, 0.0001..=1.0)
                        .text("color scale")
                        .logarithmic(true),
                );
            }
            ui.add(egui::Slider::new(&mut self.color_offset, 0.0..=1.0).text("color offset"));
            egui::ComboBox::from_label("palette")
                .selected_text(PALETTE_NAMES[self.palette as usize])
                .show_ui(ui, |ui| {
                    for (i, name) in PALETTE_NAMES.iter().enumerate() {
                        ui.selectable_value(&mut self.palette, i as u32, *name);
                    }
                });
        }
        if self.rendering_mode != 0 && self.shadow_palette == SHADOW_PALETTE_CUSTOM_LIGHTS {
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

        ui.collapsing("Advanced", |ui| {
            ui.add(
                egui::Slider::new(
                    &mut self.interact_downscale_log2,
                    0..=MAX_INTERACT_DOWNSCALE_LOG2,
                )
                .text("low resolution scale")
                .custom_formatter(|v, _| format!("1/{}", 1u32 << v as u32))
                .custom_parser(|s| {
                    let s = s.trim();
                    let n: u32 = s.strip_prefix("1/").unwrap_or(s).trim().parse().ok()?;
                    n.is_power_of_two().then(|| n.trailing_zeros() as f64)
                }),
            )
            .on_hover_text(
                "Resolution divisor (per axis) while panning/zooming. Higher gives \
                 more FPS at deep zoom but a blurrier image in motion; full \
                 resolution returns once input settles.",
            );
        });

        ui.collapsing("Animation", |ui| {
            if self.uses_classic_palette() {
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

            ui.checkbox(&mut self.anim.kind_morph, "Morph kind switch")
                .on_hover_text(
                    "When picking another fractal, blend the old and new formulas \
                     at every iteration step and glide to the new default view.",
                );
            if self.anim.kind_morph {
                ui.add(
                    egui::Slider::new(&mut self.anim.kind_morph_duration, 0.2..=10.0)
                        .text("morph s")
                        .logarithmic(true),
                );
            }

            if self.mode != FractalMode::Buddhabrot {
                if ui
                    .checkbox(&mut self.anim.kind_cycle, "Cycle kinds")
                    .on_hover_text("Step through every fractal kind in turn.")
                    .changed()
                {
                    self.anim.kind_cycle_timer = 0.0;
                }
                if self.anim.kind_cycle {
                    ui.add(
                        egui::Slider::new(&mut self.anim.kind_cycle_hold, 0.5..=30.0)
                            .text("hold s")
                            .logarithmic(true),
                    );
                }
            }

            if self.rendering_mode == 2 {
                if ui
                    .checkbox(&mut self.anim.cam_orbit, "Orbit camera")
                    .on_hover_text(
                        "Spin the 3D camera around the view, optionally bobbing its pitch.",
                    )
                    .changed()
                    && self.anim.cam_orbit
                {
                    self.anim.cam_pitch_base = self.camera.pitch;
                    self.anim.cam_pitch_phase = 0.0;
                }
                if self.anim.cam_orbit {
                    ui.add(
                        egui::Slider::new(&mut self.anim.cam_yaw_speed, -90.0..=90.0)
                            .text("yaw °/s"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.anim.cam_pitch_amp, 0.0..=30.0)
                            .text("pitch bob °"),
                    );
                    if self.anim.cam_pitch_amp > 0.0 {
                        ui.add(
                            egui::Slider::new(&mut self.anim.cam_pitch_speed, 0.005..=0.5)
                                .text("bob Hz")
                                .logarithmic(true),
                        );
                    }
                }
            }

            // Julia c only affects Julia mode; Phoenix p / λ / complex power
            // only their own kinds.
            if self.mode == FractalMode::Julia {
                self.anim.julia.ui(ui, "c", self.julia_c);
            }
            if self.kind == FractalKind::Phoenix {
                self.anim.phoenix.ui(ui, "p", self.phoenix_p);
            }
            if self.kind == FractalKind::Lambda {
                self.anim.lambda.ui(ui, "λ", self.lambda_l);
            }
            if self.kind == FractalKind::ComplexMultibrot {
                self.anim.cpow_re.ui(ui, "Re(power)", self.complex_power.0);
                self.anim.cpow_im.ui(ui, "Im(power)", self.complex_power.1);
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
                && let Ok(hh) = self.zoom_edit.parse::<Scale>()
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
            self.morph = None;
        }
        ui.add_space(8.0);
        ui.small("Drag to pan · scroll to zoom toward the cursor");
    }

    /// Controls for Buddhabrot mode: nested iteration caps (Nebulabrot R/G/B
    /// coloring), exposure, and the progressive-accumulation toggle.
    #[cfg(feature = "gui")]
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
        if self.view.magnification_log10() > 5.0 {
            ui.colored_label(
                egui::Color32::LIGHT_YELLOW,
                "deep zoom isn't supported here (f32 precision only)",
            );
        }
        ui.small("PNG export isn't available in Buddhabrot mode yet.");
    }

    #[cfg(feature = "gui")]
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
                // Orbiting only moves the camera, which the colourise pass
                // handles alone, so (like mouse-drag orbiting) it doesn't count
                // as interaction: that would drop to the low-res pass and
                // re-iterate the fractal twice.
                if t.x != 0.0 || t.y != 0.0 {
                    self.camera.rotate(-t.x * ROT_SENS, -t.y * ROT_SENS);
                }
                if mt.zoom_delta != 1.0 {
                    let off = mt.center_pos - rect.center();
                    self.zoom_3d_at(off, rect, height_px, 1. / (mt.zoom_delta as f64));
                    interacted = true;
                }
                ui.ctx().request_repaint();
            } else if response.dragged() {
                let d = response.drag_delta();
                if d.x != 0.0 || d.y != 0.0 {
                    self.camera.rotate(-d.x * ROT_SENS, -d.y * ROT_SENS);
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
                self.zoom_3d_at(off, rect, height_px, factor);
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
                self.morph = None;
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
                self.max_iterations =
                    ((self.max_iterations as f64 * 1.25).round() as u32).clamp(32, MAX_ITERATIONS);
            }
            if ui.input(|i| i.key_pressed(egui::Key::Minus)) {
                self.auto_iterations = false;
                self.max_iterations =
                    ((self.max_iterations as f64 / 1.25).round() as u32).clamp(32, MAX_ITERATIONS);
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
            // The user is steering the camera: stop the kind-switch morph from
            // overriding it (the formula blend itself carries on).
            if let Some(m) = &mut self.morph {
                m.camera = false;
            }
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
        let downscale = if interacting {
            1 << self.interact_downscale_log2
        } else {
            1
        };
        let mut size_px = [
            (((rect.width() * ppp).round() as u32) / downscale).max(1),
            (((rect.height() * ppp).round() as u32) / downscale).max(1),
        ];

        if self.effective_rendering_mode() == 2 {
            let s = self.render_scale_3d;
            size_px = size_px.map(|v| ((v as f32 * s).round() as u32).max(1));
        }

        self.screen_dim = [rect.width(), rect.height()];
        self.camera.set_aspect_ratio(aspect as f32);
        // Full-resolution height (3D included), not the interaction-downscaled one.
        let mut full_height = (rect.height() * ppp).round() as f64;
        if self.effective_rendering_mode() == 2 {
            full_height *= self.render_scale_3d as f64;
        }
        let mut uniforms = self.make_uniforms(aspect, full_height);
        // Supersampling is wasted on the low-res pass, and on a kind-switch
        // morph (every frame re-iterates, and the blend moves on next frame).
        // `ref_morph` too: the last morphed reference outlives `morph` by a
        // frame or so, until the worker delivers the plain one.
        if interacting || self.morph.is_some() || self.ref_morph.is_some() {
            uniforms.aa_level = 1;
        }
        let bla = self.bla_table(&uniforms);
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            FractalCallback {
                uniforms,
                lights: gpu_lights(&self.lights).0,
                reference: Arc::clone(&self.reference),
                generation: self.generation,
                bla,
                bla_generation: self.bla_generation,
                size_px,
                auto_color: self.auto_color_active(),
            },
        ));
    }
}

#[cfg(feature = "gui")]
impl eframe::App for FractalApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.poll_export(ui.ctx());
        self.poll_ci_stats(ui.ctx(), frame);
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
