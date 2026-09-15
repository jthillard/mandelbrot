use std::sync::Arc;

use eframe::CreationContext;
use eframe::egui_wgpu;
use eframe::egui_wgpu::wgpu;

use crate::fractal::{FractalCallback, FractalRenderer, MAX_REF_POINTS, ShareState, Uniforms};
#[cfg(target_arch = "wasm32")]
use crate::fractal::{compute_mandelbrot_reference, compute_reference};
use crate::view::{
    Big, ViewState, big_from_decimal_str, big_from_f64, big_to_decimal_str, precision_for,
};

const BAILOUT_SQ: f32 = 1.0e6;
/// Cap on exported image dimension (px), to stay within GPU texture limits.
const MAX_EXPORT_DIM: u32 = 8192 * 16;
/// Palette names; index maps to `palette_id` in the shader.
const PALETTE_NAMES: &[&str] = &["Rainbow", "Amber", "Ember", "Lime", "Grayscale"];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FractalMode {
    Mandelbrot,
    Julia,
}

/// Nice-looking Julia constants offered as presets.
const JULIA_PRESETS: &[(&str, f64, f64)] = &[
    ("dendrite", -0.8, 0.156),
    ("rabbit", -0.123, 0.745),
    ("spiral", -0.4, 0.6),
    ("san marco", -0.75, 0.0),
    ("siegel", -0.391, -0.587),
];

/// Parameters a reference orbit was (or will be) computed for. Used to decide
/// when the current reference is stale enough to recompute.
struct RequestKey {
    center_re: Big,
    center_im: Big,
    half_height: f64,
    julia: bool,
    julia_c: (f64, f64),
    iter: u32,
}

/// Top-level egui application.
pub struct FractalApp {
    view: ViewState,
    mode: FractalMode,
    julia_c: (f64, f64),
    max_iterations: u32,
    color_scale: f32,
    color_offset: f32,
    palette: u32,
    /// Supersample each pixel 2×2 for smoother edges (costs ~4× fragment work).
    antialias: bool,

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
    /// Set when the user requests a PNG export (handled after the panels draw).
    export_requested: bool,
    /// Short status line (saved path, "link copied", errors).
    status: Option<String>,
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

        let mut app = Self {
            view,
            mode: FractalMode::Mandelbrot,
            julia_c: (-0.8, 0.156),
            max_iterations: 512,
            color_scale: 0.15,
            color_offset: 0.0,
            palette: 0,
            antialias: false,
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
            export_requested: false,
            status: None,
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
            if let Ok(jc) = std::env::var("MANDEL_JULIA") {
                let p: Vec<&str> = jc.split(',').collect();
                if let (Some(Ok(re)), Some(Ok(im))) = (
                    p.first().map(|s| s.trim().parse::<f64>()),
                    p.get(1).map(|s| s.trim().parse::<f64>()),
                ) {
                    app.mode = FractalMode::Julia;
                    app.julia_c = (re, im);
                    app.view = Self::default_view_for(FractalMode::Julia);
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

    /// Snapshot the current view as a shareable state.
    fn share_state(&self) -> ShareState {
        let bits = self.view.precision_bits();
        let sig_digits = ((bits as f64) * std::f64::consts::LOG10_2).ceil() as usize + 3;
        ShareState {
            julia: matches!(self.mode, FractalMode::Julia),
            center_re: big_to_decimal_str(&self.view.center_re, sig_digits),
            center_im: big_to_decimal_str(&self.view.center_im, sig_digits),
            half_height: self.view.half_height,
            iterations: self.max_iterations,
            julia_c: self.julia_c,
            color_scale: self.color_scale,
            color_offset: self.color_offset,
        }
    }

    /// Restore a shared state into this app.
    fn apply_share(&mut self, s: &ShareState) {
        self.mode = if s.julia {
            FractalMode::Julia
        } else {
            FractalMode::Mandelbrot
        };
        self.julia_c = s.julia_c;
        self.color_scale = s.color_scale;
        self.color_offset = s.color_offset;
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

    /// Default view for a given fractal mode.
    fn default_view_for(mode: FractalMode) -> ViewState {
        match mode {
            FractalMode::Mandelbrot => ViewState::default(),
            FractalMode::Julia => {
                ViewState::with_center(big_from_f64(0.0, 53), big_from_f64(0.0, 53), 1.5)
            }
        }
    }

    fn current_key(&self) -> RequestKey {
        RequestKey {
            center_re: self.view.center_re.clone(),
            center_im: self.view.center_im.clone(),
            half_height: self.view.half_height,
            julia: matches!(self.mode, FractalMode::Julia),
            julia_c: self.julia_c,
            iter: self.max_iterations,
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
            || key.iter != self.max_iterations
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
                    )
                } else {
                    compute_mandelbrot_reference(
                        &key.center_re,
                        &key.center_im,
                        max_iter,
                        precision,
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
            dc_offset: self.dc_offset(),
        }
    }

    /// Render the current view to a PNG at `export_scale` × the on-screen size,
    /// then save it (native: file in cwd; web: browser download).
    fn do_export(&mut self, frame: &mut eframe::Frame) {
        let Some(rs) = frame.wgpu_render_state() else {
            self.status = Some("export unavailable (no wgpu backend)".into());
            return;
        };
        if self.reference.is_empty() {
            self.status = Some("still computing reference…".into());
            return;
        }

        let scale = self.export_scale.max(1.0);
        let w = ((self.last_size_px.x * scale).round() as u32).clamp(16, MAX_EXPORT_DIM);
        let h = ((self.last_size_px.y * scale).round() as u32).clamp(16, MAX_EXPORT_DIM);
        let uniforms = self.make_uniforms(w as f64 / h as f64);

        let device = rs.device.clone();
        let queue = rs.queue.clone();
        let (buffer, padded_bpr, swap) = {
            let guard = rs.renderer.read();
            let Some(renderer) = guard.callback_resources.get::<FractalRenderer>() else {
                self.status = Some("export unavailable".into());
                return;
            };
            renderer.upload_reference(&queue, &self.reference);
            let (buffer, bpr) = renderer.render_to_readback(&device, &queue, w, h, uniforms);
            (buffer, bpr, renderer.needs_rb_swap())
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            let png = pollster::block_on(read_and_encode(&device, buffer, w, h, padded_bpr, swap));
            let name = std::env::var("MANDEL_EXPORT_PATH")
                .unwrap_or_else(|_| format!("fractal-{}.png", unix_timestamp()));
            match std::fs::write(&name, &png) {
                Ok(_) => self.status = Some(format!("saved {name} ({w}×{h})")),
                Err(e) => self.status = Some(format!("save failed: {e}")),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.status = Some(format!("exporting {w}×{h}…"));
            wasm_bindgen_futures::spawn_local(async move {
                let png = read_and_encode(&device, buffer, w, h, padded_bpr, swap).await;
                web_download_png(&png, "fractal.png");
            });
        }
    }

    fn controls_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Fractal Explorer");
        ui.separator();

        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mode, FractalMode::Mandelbrot, "Mandelbrot");
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

        ui.separator();
        ui.add(
            egui::Slider::new(&mut self.max_iterations, 32..=100_000)
                .text("iterations")
                .logarithmic(true),
        );
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

        ui.separator();
        let (cre, cim) = self.view.center_f64();
        ui.label(format!("center re:\n  {cre:+.17}"));
        ui.label(format!("center im:\n  {cim:+.17}"));
        ui.label(format!("magnification: {:.3e}×", self.view.magnification()));
        ui.label(format!("reference: {} pts", self.reference.len()));
        ui.label(format!("precision: {} bits", self.view.precision_bits()));
        if self.pending {
            ui.colored_label(egui::Color32::LIGHT_YELLOW, "computing reference…");
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("Copy link").clicked() {
                let url = self.share_url();
                ui.ctx().copy_text(url);
                self.status = Some("link copied".into());
            }
            if ui.button("Export PNG").clicked() {
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
        if let Some(status) = &self.status {
            ui.small(status);
        }

        ui.separator();
        if ui.button("Reset view").clicked() {
            self.view = Self::default_view_for(self.mode);
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

        // Pan by dragging.
        if response.dragged() {
            let d = response.drag_delta();
            if d.x != 0.0 || d.y != 0.0 {
                self.view.pan_pixels(d.x as f64, d.y as f64, height_px);
            }
        }

        // Zoom toward the cursor on scroll.
        let (scroll_y, hover) = ui.input(|i| (i.smooth_scroll_delta.y, i.pointer.hover_pos()));
        if scroll_y != 0.0
            && let Some(pos) = hover
            && rect.contains(pos)
        {
            let off = pos - rect.center();
            let factor = (-scroll_y as f64 * 0.0015).exp();
            self.view
                .zoom_at_pixel(off.x as f64, off.y as f64, height_px, factor);
            ui.ctx().request_repaint();
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

        // Cache-texture resolution: the widget size in physical pixels.
        let ppp = ui.ctx().pixels_per_point();
        let size_px = [
            ((rect.width() * ppp).round() as u32).max(1),
            ((rect.height() * ppp).round() as u32).max(1),
        ];

        let uniforms = self.make_uniforms(aspect);
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

/// Map the readback buffer, unpad it, and encode a PNG. Awaited on both targets
/// (blocked on via pollster natively; spawned on the web).
async fn read_and_encode(
    device: &wgpu::Device,
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bpr: u32,
    swap_rb: bool,
) -> Vec<u8> {
    let (tx, rx) = futures_channel::oneshot::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });

    #[cfg(not(target_arch = "wasm32"))]
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    #[cfg(target_arch = "wasm32")]
    let _ = device;

    let _ = rx.await;

    let png = {
        let data = buffer
            .slice(..)
            .get_mapped_range()
            .expect("map readback buffer");
        crate::fractal::encode_png(&data, width, height, padded_bpr, swap_rb)
    };
    buffer.unmap();
    png
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
