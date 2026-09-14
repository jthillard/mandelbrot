// Fractal Explorer — Rust + wgpu + egui + WGSL deep-zoom Mandelbrot.
//
// A single binary drives both native and web (WASM/WebGPU) builds; the two
// `main` functions below are selected by target. Trunk builds the wasm32 target
// and calls the wasm `main`, which boots eframe onto the page's <canvas>.

mod app;
mod fractal;
mod view;

#[cfg(not(target_arch = "wasm32"))]
mod worker;

use app::FractalApp;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([640.0, 480.0])
            .with_title("Fractal Explorer"),
        ..Default::default()
    };

    eframe::run_native(
        "Fractal Explorer",
        native_options,
        Box::new(|cc| Ok(Box::new(FractalApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast as _;

    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("no window")
            .document()
            .expect("no document");
        let canvas = document
            .get_element_by_id("the_canvas_id")
            .expect("missing element with id `the_canvas_id`")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("`the_canvas_id` is not a <canvas>");

        let result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(FractalApp::new(cc)))),
            )
            .await;

        // Remove the "Loading…" splash regardless of success/failure.
        if let Some(loading) = document.get_element_by_id("loading_text") {
            match result {
                Ok(_) => loading.remove(),
                Err(e) => {
                    loading.set_inner_html(
                        "<p>The app has crashed. See the developer console for details.</p>",
                    );
                    log::error!("failed to start eframe: {e:?}");
                }
            }
        }
    });
}
