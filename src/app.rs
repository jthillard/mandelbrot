use std::sync::{Arc, Mutex};

use eframe::CreationContext;
use eframe::egui_wgpu;
use eframe::egui_wgpu::wgpu;

use crate::fractal::{
    ExportRender, FractalCallback, FractalKind, FractalRenderer, MAX_REF_POINTS, ShareState,
    Uniforms,
};
#[cfg(target_arch = "wasm32")]
use crate::fractal::{compute_reference, compute_set_reference};
use crate::view::{
    Big, DEFAULT_HALF_HEIGHT, ViewState, big_from_decimal_str, big_from_f64, big_to_decimal_str,
    precision_for,
};

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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FractalMode {
    Mandelbrot,
    Julia,
}

/// Selectable fractal formulas, with UI labels.
const KINDS: &[(FractalKind, &str)] = &[
    (FractalKind::Mandelbrot, "Mandelbrot"),
    (FractalKind::BurningShip, "Burning Ship"),
    (FractalKind::Tricorn, "Tricorn"),
    (FractalKind::Multibrot, "Multibrot"),
    (FractalKind::Celtic, "Celtic"),
    (FractalKind::Perpendicular, "Perpendicular"),
    (FractalKind::Buffalo, "Buffalo"),
    (FractalKind::Phoenix, "Phoenix"),
];

/// UI label for a fractal kind.
fn kind_label(kind: FractalKind) -> &'static str {
    KINDS
        .iter()
        .find(|(k, _)| *k == kind)
        .map(|(_, name)| *name)
        .unwrap_or("Mandelbrot")
}

/// Nice-looking Julia constants offered as presets.
const JULIA_PRESETS: &[(&str, f64, f64)] = &[
    ("dendrite", -0.8, 0.156),
    ("rabbit", -0.123, 0.745),
    ("spiral", -0.4, 0.6),
    ("san marco", -0.75, 0.0),
    ("siegel", -0.391, -0.587),
];

/// Curated beautiful Mandelbrot locations offered as one-click presets.
/// Each is `(name, center_re, center_im, half_height, iterations)`; the centers
/// are decimals parsed at full precision so deep places stay sharp.
const MANDEL_PLACES: &[(&str, &str, &str, f64, u32)] = &[
    (
        "Seahorse Valley",
        "-0.743643887037158704752191506114774",
        "0.131825904205311970493132056385139",
        4.0e-6,
        1500,
    ),
    (
        "Elephant Valley",
        "0.2549870375144766",
        "0.0005679790528465",
        6.0e-5,
        2000,
    ),
    ("Scepter Valley", "-1.360022", "0.041", 3.0e-4, 2000),
    ("Starburst", "-1.62917", "0.0203968", 1.5e-3, 1500),
    (
        "Deep Spiral",
        "-0.7436438870371587",
        "0.1318259042053",
        8.0e-8,
        3000,
    ),
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
    iter: u32,
    kind: FractalKind,
    power: u32,
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

/// Top-level egui application.
pub struct FractalApp {
    view: ViewState,
    mode: FractalMode,
    /// Iteration formula.
    kind: FractalKind,
    /// Exponent for the Multibrot kind.
    power: u32,
    julia_c: (f64, f64),
    /// Distortion constant `p` for the Phoenix kind (`z^2 + c + p·z_{n-1}`).
    phoenix_p: (f64, f64),
    max_iterations: u32,
    /// When set, `max_iterations` tracks the zoom depth automatically (so deep
    /// zooms stay sharp without hand-tuning); the manual slider takes over when
    /// unset. Turned off when a preset or share link supplies an explicit count.
    auto_iterations: bool,
    color_scale: f32,
    color_offset: f32,
    palette: u32,
    /// Supersample each pixel 2×2 for smoother edges (costs ~4× fragment work).
    antialias: bool,
    /// Distance-estimation shading: darkens toward the set boundary using the
    /// orbit derivative, giving crisp filaments at deep zoom instead of speckle.
    de_coloring: bool,

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
}

/// Significant decimal digits to show for a center at the given precision (bits).
fn sig_digits_for(bits: usize) -> usize {
    ((bits as f64) * std::f64::consts::LOG10_2).ceil() as usize + 3
}

/// Format a magnification for the editable field (compact scientific).
fn format_magnification(m: f64) -> String {
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
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(renderer);

        let view = ViewState::default();
        let ref_center_re = view.center_re.clone();
        let ref_center_im = view.center_im.clone();
        let ref_half_height = view.half_height;
        let sig = sig_digits_for(view.precision_bits());
        let center_re_edit = big_to_decimal_str(&view.center_re, sig);
        let center_im_edit = big_to_decimal_str(&view.center_im, sig);
        let zoom_edit = format_magnification(view.magnification());

        let mut app = Self {
            view,
            mode: FractalMode::Mandelbrot,
            kind: FractalKind::Mandelbrot,
            power: 3,
            julia_c: (-0.8, 0.156),
            phoenix_p: (-0.5, 0.0),
            max_iterations: 512,
            auto_iterations: true,
            color_scale: 0.15,
            color_offset: 0.0,
            palette: 0,
            antialias: false,
            de_coloring: false,
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
            status: None,
            center_re_edit,
            center_im_edit,
            zoom_edit,
            zoom_edited: false,
        };

        // On the web, restore a shared view from the URL fragment (#...).
        #[cfg(target_arch = "wasm32")]
        if let Some(frag) = web_location_hash() {
            if let Some(state) = ShareState::decode(&frag) {
                app.apply_share(&state);
            }
        }

        // Debug/testing hooks.
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(k) = std::env::var("MANDEL_KIND") {
                app.kind = match k.trim().to_ascii_lowercase().as_str() {
                    "burningship" | "burning_ship" | "ship" => FractalKind::BurningShip,
                    "tricorn" | "mandelbar" => FractalKind::Tricorn,
                    "multibrot" | "multi" => FractalKind::Multibrot,
                    "celtic" => FractalKind::Celtic,
                    "perpendicular" | "perp" => FractalKind::Perpendicular,
                    "buffalo" => FractalKind::Buffalo,
                    "phoenix" => FractalKind::Phoenix,
                    _ => FractalKind::Mandelbrot,
                };
                if let Ok(p) = std::env::var("MANDEL_POWER")
                    && let Ok(p) = p.trim().parse::<u32>()
                {
                    app.power = p.clamp(2, 8);
                }
                app.view = Self::default_view_for(app.mode, app.kind);
            }
            if let Ok(jc) = std::env::var("MANDEL_JULIA") {
                let p: Vec<&str> = jc.split(',').collect();
                if let (Some(Ok(re)), Some(Ok(im))) = (
                    p.first().map(|s| s.trim().parse::<f64>()),
                    p.get(1).map(|s| s.trim().parse::<f64>()),
                ) {
                    app.mode = FractalMode::Julia;
                    app.julia_c = (re, im);
                    app.view = Self::default_view_for(FractalMode::Julia, app.kind);
                }
            }
            if let Ok(frag) = std::env::var("MANDEL_SHARE")
                && let Some(state) = ShareState::decode(&frag)
            {
                app.apply_share(&state);
            }
            if let Ok(spec) = std::env::var("MANDEL_VIEW") {
                app.apply_view_spec(&spec);
            }
            if std::env::var("MANDEL_DE").is_ok() {
                app.de_coloring = true;
            }
            if std::env::var("MANDEL_EXPORT").is_ok() {
                app.export_requested = true;
            }
        }

        app
    }

    /// Apply a view spec "re,im,half_height[,iterations]" (re/im are decimal,
    /// parsed at full precision). Used by the native debug env var.
    #[allow(dead_code)]
    pub fn apply_view_spec(&mut self, spec: &str) -> bool {
        let parts: Vec<&str> = spec.split(',').collect();
        if parts.len() < 3 {
            return false;
        }
        let Ok(half_height) = parts[2].trim().parse::<f64>() else {
            return false;
        };
        if !(half_height > 0.0 && half_height.is_finite()) {
            return false;
        }
        let bits = precision_for(half_height);
        let (Some(re), Some(im)) = (
            big_from_decimal_str(parts[0], bits),
            big_from_decimal_str(parts[1], bits),
        ) else {
            return false;
        };
        self.view = ViewState::with_center(re, im, half_height);
        if let Some(it) = parts.get(3)
            && let Ok(v) = it.trim().parse::<u32>()
        {
            self.max_iterations = v.clamp(32, MAX_REF_POINTS as u32 - 1);
        }
        true
    }

    /// Jump to a preset Mandelbrot location: decimal center (parsed at the
    /// precision the zoom needs), half-height, and a fitting iteration count.
    fn go_to_place(&mut self, re: &str, im: &str, half_height: f64, iterations: u32) {
        let bits = precision_for(half_height);
        if let (Some(cre), Some(cim)) =
            (big_from_decimal_str(re, bits), big_from_decimal_str(im, bits))
        {
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
            color_scale: self.color_scale,
            color_offset: self.color_offset,
            palette: self.palette,
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
        self.color_scale = s.color_scale;
        self.color_offset = s.color_offset;
        self.palette = (s.palette as usize).min(PALETTE_NAMES.len() - 1) as u32;
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
        let (cr, ci, hh) = match kind {
            FractalKind::Mandelbrot => (-0.5, 0.0, 1.25),
            FractalKind::BurningShip => (-0.5, -0.5, 1.3),
            FractalKind::Tricorn => (-0.25, 0.0, 1.6),
            FractalKind::Multibrot => (0.0, 0.0, 1.5),
            FractalKind::Celtic => (-0.5, 0.0, 1.6),
            FractalKind::Perpendicular => (-0.5, 0.0, 1.5),
            FractalKind::Buffalo => (-0.5, -0.5, 1.5),
            FractalKind::Phoenix => (0.0, 0.0, 1.6),
        };
        ViewState::with_center(big_from_f64(cr, 53), big_from_f64(ci, 53), hh)
    }

    fn current_key(&self) -> RequestKey {
        RequestKey {
            center_re: self.view.center_re.clone(),
            center_im: self.view.center_im.clone(),
            half_height: self.view.half_height,
            julia: matches!(self.mode, FractalMode::Julia),
            julia_c: self.julia_c,
            phoenix_p: self.phoenix_p,
            iter: self.max_iterations,
            kind: self.kind,
            power: self.power,
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
    /// serves it well.
    fn should_request(&self) -> bool {
        let Some(key) = &self.last_request else {
            return true;
        };
        if key.julia != matches!(self.mode, FractalMode::Julia)
            || key.julia_c != self.julia_c
            || key.phoenix_p != self.phoenix_p
            || key.iter != self.max_iterations
            || key.kind != self.kind
            || key.power != self.power
        {
            return true;
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

    /// Recompute the reference orbit when needed. Native: dispatch to a worker
    /// thread and pick up completed results. Web: compute inline.
    fn ensure_reference(&mut self) {
        if self.should_request() {
            let key = self.current_key();
            let precision = self.view.precision_bits();
            let max_iter = key.iter.min(MAX_REF_POINTS as u32 - 1);

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

    fn make_uniforms(&self, aspect: f64) -> Uniforms {
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
            aa_level: if self.antialias { 2 } else { 1 },
            kind: self.kind.shader_id(),
            power: self.power,
            dc_offset: self.dc_offset(),
            phoenix_p: [self.phoenix_p.0 as f32, self.phoenix_p.1 as f32],
            de_coloring: self.de_coloring as u32,
            _pad: [0, 0, 0],
        }
    }

    /// Render the current view to a PNG at `export_scale` × the on-screen size,
    /// then save it (native: file in cwd; web: browser download). Runs off the
    /// UI thread so a progress bar can animate; progress lands in `self.export`.
    fn do_export(&mut self, frame: &mut eframe::Frame) {
        if self.export.is_some() {
            return; // one export at a time
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
            renderer.export_handles()
        };
        let reference = Arc::clone(&self.reference);

        let shared = Arc::new(Mutex::new(ExportShared {
            fraction: 0.0,
            phase: "Rendering",
            result: None,
        }));
        self.status = None;
        self.export = Some(Arc::clone(&shared));

        // Progress budget: rendering fills [0, RENDER_END], encoding the rest.
        const RENDER_END: f32 = 0.6;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let name = std::env::var("MANDEL_EXPORT_PATH")
                .unwrap_or_else(|_| format!("fractal-{}.png", unix_timestamp()));
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
                );

                // Render the image tile by tile, waiting for each so progress
                // reflects real GPU work.
                for t in 0..er.tiles {
                    er.render_tile(&device, &queue, t);
                    let _ = device.poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    });
                    let done = (t + 1) as f32 / er.tiles as f32;
                    set_progress(&shared, "Rendering", RENDER_END * done);
                }
                er.copy_to_readback(&device, &queue);

                // Wait for the copy, then read the mapped bytes.
                let (tx, rx) = std::sync::mpsc::channel();
                er.readback().slice(..).map_async(wgpu::MapMode::Read, move |res| {
                    let _ = tx.send(res);
                });
                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });
                let _ = rx.recv();

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
                let result = std::fs::write(&name, &png)
                    .map(|_| format!("saved {name} ({w}×{h})"))
                    .map_err(|e| format!("save failed: {e}"));
                finish_export(&shared, result);
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
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
                er.readback().slice(..).map_async(wgpu::MapMode::Read, move |res| {
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

    fn controls_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Fractal Explorer");
        ui.separator();

        // Fractal formula. Switching kinds jumps to a sensible default view,
        // since interesting regions differ between fractals.
        let prev_kind = self.kind;
        egui::ComboBox::from_label("fractal")
            .selected_text(kind_label(self.kind))
            .show_ui(ui, |ui| {
                for &(kind, name) in KINDS {
                    ui.selectable_value(&mut self.kind, kind, name);
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
        if self.kind != prev_kind {
            self.view = Self::default_view_for(self.mode, self.kind);
        }

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mode, FractalMode::Mandelbrot, "Set");
            ui.radio_value(&mut self.mode, FractalMode::Julia, "Julia");
        });

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
            ui.horizontal_wrapped(|ui| {
                for &(name, re, im) in JULIA_PRESETS {
                    if ui.small_button(name).clicked() {
                        self.julia_c = (re, im);
                    }
                }
            });
        }

        if self.mode == FractalMode::Mandelbrot && self.kind == FractalKind::Mandelbrot {
            ui.label("places:");
            ui.horizontal_wrapped(|ui| {
                for &(name, re, im, half_height, iter) in MANDEL_PLACES {
                    if ui.small_button(name).clicked() {
                        self.go_to_place(re, im, half_height, iter);
                    }
                }
            });
        }

        ui.separator();
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
        ui.checkbox(&mut self.antialias, "Antialiasing (2×2)")
            .on_hover_text("Supersample each pixel for smoother edges (~4× slower).");
        ui.checkbox(&mut self.de_coloring, "Distance shading")
            .on_hover_text(
                "Shade by distance to the set boundary (from the orbit derivative) \
                 for crisp filaments at deep zoom. Exact for the holomorphic kinds \
                 (Mandelbrot/Multibrot/Phoenix), approximate for the abs-based kinds \
                 (Burning Ship/Tricorn/Celtic/Perpendicular/Buffalo).",
            );

        ui.separator();
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
            && let Some(v) =
                big_from_decimal_str(&self.center_re_edit, parse_bits_for(&self.center_re_edit, bits))
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
            && let Some(v) =
                big_from_decimal_str(&self.center_im_edit, parse_bits_for(&self.center_im_edit, bits))
        {
            self.view.center_im = v;
            self.view.sync_precision();
        }
        if !im_resp.has_focus() {
            self.center_im_edit = big_to_decimal_str(&self.view.center_im, sig);
        }

        ui.label("magnification (×):");
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
                && let Ok(m) = self.zoom_edit.trim().parse::<f64>()
            {
                let hh = DEFAULT_HALF_HEIGHT / m;
                if m > 0.0 && hh > 0.0 && hh.is_finite() {
                    self.view.half_height = hh;
                    self.view.sync_precision();
                }
            }
            self.zoom_edited = false;
        }
        if !zoom_resp.has_focus() {
            self.zoom_edit = format_magnification(self.view.magnification());
        }
        ui.label(format!("reference: {} pts", self.reference.len()));
        ui.label(format!("precision: {} bits", self.view.precision_bits()));
        if self.pending {
            ui.colored_label(egui::Color32::LIGHT_YELLOW, "computing reference…");
        }

        ui.separator();
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
                    .speed(0.5),
            );
            ui.label(format!(
                "→ {}×{}",
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

        ui.separator();
        if ui.button("Reset view").clicked() {
            self.view = Self::default_view_for(self.mode, self.kind);
        }
        ui.add_space(8.0);
        ui.small("Drag to pan · scroll to zoom toward the cursor");
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

        // Touch: pinch to zoom (toward the gesture center) and two-finger pan.
        // Takes precedence over single-finger drag while two fingers are down.
        let multi_touch = ui.input(|i| i.multi_touch());
        if let Some(mt) = multi_touch {
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

        // Mouse wheel / trackpad: zoom toward the cursor.
        let (scroll_y, hover) = ui.input(|i| (i.smooth_scroll_delta.y, i.pointer.hover_pos()));
        if scroll_y != 0.0
            && let Some(pos) = hover
            && rect.contains(pos)
        {
            let off = pos - rect.center();
            let factor = (-scroll_y as f64 * 0.0015).exp();
            self.view
                .zoom_at_pixel(off.x as f64, off.y as f64, height_px, factor);
            interacted = true;
            ui.ctx().request_repaint();
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
        let size_px = [
            (((rect.width() * ppp).round() as u32) / downscale).max(1),
            (((rect.height() * ppp).round() as u32) / downscale).max(1),
        ];

        let mut uniforms = self.make_uniforms(aspect);
        if interacting {
            uniforms.aa_level = 1; // supersampling is wasted on the low-res pass
        }
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            FractalCallback {
                uniforms,
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

        egui::Panel::right("controls")
            .default_size(280.0)
            .show(ui, |ui| self.controls_ui(ui));

        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.fractal_ui(ui));

        if std::mem::take(&mut self.export_requested) {
            self.do_export(frame);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(&mut *self)
    }
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
fn unix_timestamp() -> u64 {
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
