// Native command-line arguments. Currently mirrors the old `MANDEL_*` debug
// env vars one-for-one; this is the foundation a future headless (no-window,
// render-to-file) mode will build on.

use clap::{Parser, ValueEnum};

use crate::fractal::FractalKind;

#[derive(Parser, Debug, Default)]
#[command(name = "mandelbrot", about = "Deep-zoom fractal explorer", version)]
pub struct Cli {
    /// Fractal formula to render.
    #[arg(long, value_enum)]
    pub kind: Option<KindArg>,

    /// Exponent for the Multibrot kind (z -> z^power + c), clamped to [2, 8].
    #[arg(long)]
    pub power: Option<u32>,

    /// Complex exponent for the Complex Multibrot kind (z -> z^power + c).
    #[arg(long, value_name = "RE,IM")]
    pub complex_power: Option<String>,

    /// Start in Julia mode with this seed constant.
    #[arg(long, value_name = "RE,IM")]
    pub julia: Option<String>,

    /// Restore a view from a share-link fragment (the part after '#').
    #[arg(long, value_name = "FRAGMENT")]
    pub share: Option<String>,

    /// Jump to a view on startup.
    #[arg(long, value_name = "RE,IM,HALF_HEIGHT[,ITERATIONS]")]
    pub view: Option<String>,

    /// Enable distance-estimation shading.
    #[arg(long)]
    pub de: bool,

    /// Switch to the Buddhabrot renderer.
    #[arg(long)]
    pub buddhabrot: bool,

    /// Buddhabrot tonemap palette index.
    #[arg(long, value_name = "INDEX")]
    pub buddha_palette: Option<u32>,

    /// Render a PNG export on startup.
    #[arg(long)]
    pub export: bool,

    /// Output path for --export/--headless (default: fractal-<timestamp>.png).
    #[arg(long, value_name = "PATH")]
    pub export_path: Option<String>,

    /// Run without opening a window: render the current view to a PNG and
    /// exit. Combine with --kind/--julia/--share/--view etc. to pick what to
    /// render. Not yet supported with --buddhabrot.
    #[arg(long)]
    pub headless: bool,

    /// Output image width in pixels (--headless only).
    #[arg(long, value_name = "PX", default_value_t = 1920)]
    pub width: u32,

    /// Output image height in pixels (--headless only).
    #[arg(long, value_name = "PX", default_value_t = 1080)]
    pub height: u32,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum KindArg {
    Mandelbrot,
    #[value(alias = "ship")]
    BurningShip,
    #[value(alias = "mandelbar")]
    Tricorn,
    #[value(alias = "multi")]
    Multibrot,
    Celtic,
    #[value(alias = "perp")]
    Perpendicular,
    Buffalo,
    Phoenix,
    Lambda,
    #[value(alias = "cmulti")]
    ComplexMultibrot,
}

impl From<KindArg> for FractalKind {
    fn from(k: KindArg) -> Self {
        match k {
            KindArg::Mandelbrot => FractalKind::Mandelbrot,
            KindArg::BurningShip => FractalKind::BurningShip,
            KindArg::Tricorn => FractalKind::Tricorn,
            KindArg::Multibrot => FractalKind::Multibrot,
            KindArg::Celtic => FractalKind::Celtic,
            KindArg::Perpendicular => FractalKind::Perpendicular,
            KindArg::Buffalo => FractalKind::Buffalo,
            KindArg::Phoenix => FractalKind::Phoenix,
            KindArg::Lambda => FractalKind::Lambda,
            KindArg::ComplexMultibrot => FractalKind::ComplexMultibrot,
        }
    }
}
