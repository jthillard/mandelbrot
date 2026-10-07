// Without these, rust fails to infer Send/Sync trait impls
// Probably caused by the new trait solver
#![recursion_limit = "256"]
// Without the `gui` feature, UI-only state and helpers go unused.
#![cfg_attr(not(feature = "gui"), allow(dead_code, unused_imports))]

// Fractal Explorer — Rust + wgpu + egui + WGSL deep-zoom Mandelbrot.
//
// A single binary drives both native and web (WASM/WebGPU) builds; the two
// `main` functions below are selected by target. Trunk builds the wasm32 target
// and calls the wasm `main`, which boots eframe onto the page's <canvas>.

mod app;
mod bignum;
mod camera;
mod fractal;
mod lights;
mod view;

#[cfg(not(target_arch = "wasm32"))]
mod cli;
#[cfg(not(target_arch = "wasm32"))]
mod headless;
#[cfg(not(target_arch = "wasm32"))]
mod worker;

#[cfg(all(target_arch = "wasm32", not(feature = "gui")))]
compile_error!("the web build needs the `gui` feature");

#[cfg(feature = "gui")]
use app::FractalApp;

#[cfg(all(feature = "gui", not(target_arch = "wasm32")))]
type MainResult = eframe::Result;
#[cfg(all(not(feature = "gui"), not(target_arch = "wasm32")))]
type MainResult = Result<(), String>;

/// wgpu configuration for eframe. The fractal fragment shader reads the
/// reference orbit from **storage buffers** where the device allows them in
/// the fragment stage (`GpuPath` in renderer.rs). eframe's default requests
/// WebGL2-downlevel limits when a GL adapter is picked (storage buffers = 0),
/// so we:
///   * request the adapter's real limits (which include storage buffers), and
///   * on the web, use WebGPU only, failing cleanly on browsers without it,
///     unless the `webgl` feature adds WebGL2 as a fallback (egui-wgpu
///     tries WebGPU first).
#[cfg(feature = "gui")]
fn wgpu_options() -> eframe::egui_wgpu::WgpuConfiguration {
    use eframe::egui_wgpu::{WgpuSetup, wgpu};

    let mut options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let WgpuSetup::CreateNew(setup) = &mut options.wgpu_setup {
        setup.device_descriptor =
            std::sync::Arc::new(|adapter: &wgpu::Adapter| wgpu::DeviceDescriptor {
                label: Some("fractal wgpu device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            });
        #[cfg(all(target_arch = "wasm32", not(feature = "webgl")))]
        {
            setup.instance_descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
        }
        #[cfg(all(target_arch = "wasm32", feature = "webgl"))]
        {
            setup.instance_descriptor.backends = if web_forced_webgl() {
                wgpu::Backends::GL
            } else {
                wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL
            };
        }
    }
    options
}

/// The page asks for WebGL2 even where WebGPU works (`?backend=webgl`, set
/// by the "Use WebGPU" toggle in the app's Advanced section).
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
fn web_forced_webgl() -> bool {
    web_sys::window()
        .and_then(|w| w.location().search().ok())
        .and_then(|q| web_sys::UrlSearchParams::new_with_str(&q).ok())
        .and_then(|p| p.get("backend"))
        .is_some_and(|b| b == "webgl")
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> MainResult {
    use clap::Parser as _;

    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let cli = cli::Cli::parse();
    if cli.headless {
        return match headless::run(cli) {
            Ok(()) => Ok(()),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        };
    }

    #[cfg(not(feature = "gui"))]
    {
        eprintln!("error: built without the \"gui\" feature; only --headless is supported");
        std::process::exit(1);
    }

    #[cfg(feature = "gui")]
    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: wgpu_options(),
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([640.0, 480.0])
            .with_title("Fractal Explorer"),
        ..Default::default()
    };

    #[cfg(feature = "gui")]
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

    let web_options = eframe::WebOptions {
        wgpu_options: wgpu_options(),
        ..Default::default()
    };

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
                    loading.set_inner_html(&format!("<p>The app has crashed.</br>{e:?}</p>"));
                    log::error!("failed to start eframe: {e:?}");
                }
            }
        }
    });
}
