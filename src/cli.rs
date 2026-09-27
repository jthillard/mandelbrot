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

    /// Start in Julia mode with this seed constant.
    #[arg(long, value_name = "RE,IM")]
    pub julia: Option<String>,

    /// Switch to the Buddhabrot renderer.
    #[arg(long)]
    pub buddhabrot: bool,

    /// Rendering mode to use.
    #[arg(long)]
    pub rendering_kind: Option<RenderingKindArg>,

    /// Exponent for the Multibrot kind (z -> z^power + c), clamped to [2, 20].
    #[arg(long)]
    pub power: Option<u32>,

    /// Complex exponent for the Complex Multibrot kind (z -> z^power + c).
    #[arg(long, value_name = "RE,IM")]
    pub complex_power: Option<String>,

    /// Phoenix constant p for the Phoenix kind (z -> z^2 + c + p*z_prev).
    #[arg(long, value_name = "RE,IM")]
    pub phoenix_p: Option<String>,

    /// Lambda constant λ for the Lambda kind (z -> λ*z*(1 - z)).
    #[arg(long, value_name = "RE,IM")]
    pub lambda_l: Option<String>,

    /// Restore a view from a share-link fragment (the part after '#').
    #[arg(long, value_name = "FRAGMENT")]
    pub share: Option<String>,

    /// Jump to a view on startup.
    #[arg(long, value_name = "RE,IM,HALF_HEIGHT[,ITERATIONS]")]
    pub view: Option<String>,

    /// Jump to a specific position on startup.
    #[arg(long, short('p'), value_name = "RE,IM")]
    pub position: Option<String>,

    /// Set a maximum iterations count on startup.
    #[arg(long, short('i'))]
    pub iterations: Option<u32>,

    /// Set the zoom level on startup.
    #[arg(long("zoom"), short('z'))]
    pub half_height: Option<String>,

    /// 3D camera yaw in degrees (with --rendering-kind 3d).
    #[arg(long, value_name = "DEG", allow_hyphen_values = true)]
    pub yaw: Option<f32>,

    /// 3D camera pitch in degrees (with --rendering-kind 3d); negative tilts
    /// the view down toward the fractal. Clamped short of ±90.
    #[arg(long, value_name = "DEG", allow_hyphen_values = true)]
    pub pitch: Option<f32>,

    /// Enable distance-estimation shading.
    #[arg(long)]
    pub de: bool,

    /// Coloring palette index.
    #[arg(long, value_name = "INDEX")]
    pub palette: Option<u32>,

    /// Output path for --headless (default: fractal-<timestamp>.png). When
    /// animating (--to-view/--to-share), this is a directory of
    /// frame-00001.png, frame-00002.png, ... instead (default:
    /// frames-<timestamp>/).
    #[arg(long, value_name = "PATH")]
    pub export_path: Option<String>,

    /// End view for an animation: "re,im,half_height[,iterations]", the same
    /// syntax as --view. Combine with --view (or --share, --kind, --julia...)
    /// for the start view; headless then renders a sequence of frames
    /// interpolating the camera from start to end instead of a single PNG.
    #[arg(long, value_name = "RE,IM,HALF_HEIGHT[,ITERATIONS]")]
    pub to_view: Option<String>,

    /// End view for an animation, as a share-link fragment (only the
    /// position/zoom/iterations are used; alternative to --to-view for
    /// pasting a location copied from the app's "Copy share link").
    #[arg(long, value_name = "FRAGMENT")]
    pub to_share: Option<String>,

    /// Set a maximum iterations count at animation end.
    #[arg(long)]
    pub to_iterations: Option<u32>,

    /// End Julia constant for an animation: c is interpolated from --julia
    /// to this over the frames.
    #[arg(long, value_name = "RE,IM", allow_hyphen_values = true)]
    pub to_julia: Option<String>,

    /// End Phoenix constant p for an animation (from --phoenix-p).
    #[arg(long, value_name = "RE,IM", allow_hyphen_values = true)]
    pub to_phoenix_p: Option<String>,

    /// End Lambda constant λ for an animation (from --lambda-l).
    #[arg(long, value_name = "RE,IM", allow_hyphen_values = true)]
    pub to_lambda_l: Option<String>,

    /// End Complex Multibrot exponent for an animation (from
    /// --complex-power). Shorthand for --to-complex-power-re +
    /// --to-complex-power-im.
    #[arg(long, value_name = "RE,IM", allow_hyphen_values = true)]
    pub to_complex_power: Option<String>,

    /// End real part of the Complex Multibrot exponent for an animation;
    /// the imaginary part stays put unless --to-complex-power-im is given.
    #[arg(long, value_name = "RE", allow_hyphen_values = true)]
    pub to_complex_power_re: Option<f64>,

    /// End imaginary part of the Complex Multibrot exponent for an animation;
    /// the real part stays put unless --to-complex-power-re is given.
    #[arg(long, value_name = "IM", allow_hyphen_values = true)]
    pub to_complex_power_im: Option<f64>,

    /// End 3D camera yaw for an animation, in degrees (from --yaw). Not
    /// wrapped: --yaw 0 --to-yaw 720 orbits twice.
    #[arg(long, value_name = "DEG", allow_hyphen_values = true)]
    pub to_yaw: Option<f32>,

    /// End 3D camera pitch for an animation, in degrees (from --pitch).
    #[arg(long, value_name = "DEG", allow_hyphen_values = true)]
    pub to_pitch: Option<f32>,

    /// Morph the iteration formula from the start kind (--kind) to this one
    /// over the animation. The camera is unaffected (use --to-view for that).
    #[arg(long, value_enum)]
    pub to_kind: Option<KindArg>,

    /// Number of frames to render for an animation. Alternative to --fps +
    /// --duration.
    #[arg(long, value_name = "N")]
    pub frames: Option<u32>,

    /// Frames per second, used with --duration to compute the frame count
    /// (ignored if --frames is given). Also used in the ffmpeg command
    /// hint printed after rendering.
    #[arg(long, value_name = "N", default_value_t = 30.0)]
    pub fps: f64,

    /// Animation duration in seconds, used with --fps to compute the frame
    /// count (ignored if --frames is given).
    #[arg(long, value_name = "SECONDS")]
    pub duration: Option<f64>,

    /// Pace animation frames linearly instead of easing in/out (smoothstep).
    #[arg(long)]
    pub linear: bool,

    /// Run without opening a window: render the current view to a PNG and
    /// exit. Combine with --kind/--julia/--share/--view etc. to pick what to
    /// render, or any --to-* flag (--to-view, --to-julia, --to-kind, ...) to
    /// render an animation instead of a single frame. Not yet supported with --buddhabrot.
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

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum RenderingKindArg {
    Classic,
    Shadow,
    #[value(alias = "3d")]
    Dimension3,
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
