//! `FractalKind`: the enum selecting which iteration formula is in use, plus
//! everything that only needs to switch on it (UI label/description/formula
//! text, share-link tag, default parameter-plane view). The CPU/GPU orbit
//! math itself lives in `reference.rs` (CPU reference orbit) and
//! `shaders/mandelbrot.wgsl` (GPU perturbation delta) since both must also
//! stay in sync with `common.wgsl`'s `KIND_*` constants.

/// The iteration formula. Must be kept in sync with `advance_delta` and the
/// `KIND_*` constants in the shader.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FractalKind {
    /// `z -> z^2 + c`.
    Mandelbrot = 0,
    /// `z -> (|Re z| + i|Im z|)^2 + c`.
    BurningShip = 1,
    /// `z -> conj(z)^2 + c` (the Mandelbar).
    Tricorn = 2,
    /// `z -> z^power + c` (power >= 2).
    Multibrot = 3,
    /// `z -> |Re(z^2)| + i·Im(z^2) + c` (abs on the real output of the square).
    Celtic = 4,
    /// `z -> (x^2 - y^2) - 2·x·|y|·i + c` (abs on the imaginary input).
    Perpendicular = 5,
    /// `z -> |Re(z^2)| - |Im(z^2)|·i + c` (abs on both outputs).
    Buffalo = 6,
    /// `z -> z^2 + c + p·z_{n-1}` (two-term recurrence; `p` is `phoenix_p`).
    Phoenix = 7,
    /// `z -> lambda·z(1 - z) + c` (logistic map).
    Lambda = 8,
    /// `z -> z^power + c`, where `power` is a complex constant (the
    /// `complex_power` argument), via the principal branch `z^p = exp(p·ln z)`.
    ComplexMultibrot = 9,
}

impl FractalKind {
    /// Every kind, in declaration/discriminant order. Sized arrays keyed by
    /// `kind as usize` (`JULIA_PRESETS`, `SET_PRESETS`) must have one slot per
    /// entry here.
    pub const ALL: [FractalKind; 10] = [
        FractalKind::Mandelbrot,
        FractalKind::BurningShip,
        FractalKind::Tricorn,
        FractalKind::Multibrot,
        FractalKind::Celtic,
        FractalKind::Perpendicular,
        FractalKind::Buffalo,
        FractalKind::Phoenix,
        FractalKind::Lambda,
        FractalKind::ComplexMultibrot,
    ];

    pub fn description(&self) -> &'static str {
        match self {
            FractalKind::Mandelbrot => {
                "The Mandelbrot set is the most famous fractal set, obtained with the simplest escape-time formula. This set represents all Julia fractals: each points of the Mandelbrot set is related to a specific Julia fractal."
            }
            FractalKind::BurningShip => {
                "A variation of the famous Mandelbrot set, using absolute values on the real and imaginary part of each iterations."
            }
            FractalKind::Tricorn => {
                "The Tricorn set is obtained using the same formula as the Mandelbrot set, taking the complex conjugate of the previous iteration."
            }
            FractalKind::Multibrot => {
                "Multibrot use the same formula as the Mandelbrot set, with a bigger exposant."
            }
            FractalKind::Celtic => "",
            FractalKind::Perpendicular => "",
            FractalKind::Buffalo => "",
            FractalKind::Phoenix => "",
            FractalKind::Lambda => "",
            FractalKind::ComplexMultibrot => {
                "Like Multibrot, but the exponent itself is a complex number instead of a plain integer, via z^p = exp(p·ln z)."
            }
        }
    }

    /// UI label for this kind (combo box / info panel heading).
    pub fn label(&self) -> &'static str {
        match self {
            FractalKind::Mandelbrot => "Mandelbrot",
            FractalKind::BurningShip => "Burning Ship",
            FractalKind::Tricorn => "Tricorn",
            FractalKind::Multibrot => "Multibrot",
            FractalKind::Celtic => "Celtic",
            FractalKind::Perpendicular => "Perpendicular",
            FractalKind::Buffalo => "Buffalo",
            FractalKind::Phoenix => "Phoenix",
            FractalKind::Lambda => "Lambda",
            FractalKind::ComplexMultibrot => "Complex Multibrot",
        }
    }

    /// The iteration formula in human-readable notation (mirrors the doc
    /// comments on the variants above). `power` is only used by Multibrot;
    /// `complex_power` only by Complex Multibrot.
    pub fn formula(&self, power: u32, complex_power: (f64, f64)) -> String {
        match self {
            FractalKind::Mandelbrot => "z = z² + c".to_string(),
            FractalKind::BurningShip => "z = (|Re(z)| + i|Im(z)|)² + c".to_string(),
            FractalKind::Tricorn => "z = conj(z)² + c".to_string(),
            FractalKind::Multibrot => format!("z = z^{power} + c"),
            FractalKind::Celtic => "z = |Re(z²)| + i·Im(z²) + c".to_string(),
            FractalKind::Perpendicular => "z = (x² − y²) − 2x|y|i + c".to_string(),
            FractalKind::Buffalo => "z = |Re(z²)| − i|Im(z²)| + c".to_string(),
            FractalKind::Phoenix => "z = z² + c + p·z_prev".to_string(),
            FractalKind::Lambda => "z = λ·z(1 − z) + c".to_string(),
            FractalKind::ComplexMultibrot => {
                format!("z = z^({:.3}{:+.3}i) + c", complex_power.0, complex_power.1)
            }
        }
    }

    /// Short tag used to identify this kind in a share-link fragment.
    pub fn share_tag(&self) -> &'static str {
        match self {
            FractalKind::Mandelbrot => "mandel",
            FractalKind::BurningShip => "burning",
            FractalKind::Tricorn => "tricorn",
            FractalKind::Multibrot => "multi",
            FractalKind::Celtic => "celtic",
            FractalKind::Perpendicular => "perp",
            FractalKind::Buffalo => "buffalo",
            FractalKind::Phoenix => "phoenix",
            FractalKind::Lambda => "lambda",
            FractalKind::ComplexMultibrot => "cmulti",
        }
    }

    /// Inverse of `share_tag`; unknown tags fall back to `None` so the caller
    /// can decide the default (matches historical share-link behavior).
    pub fn from_share_tag(tag: &str) -> Option<FractalKind> {
        Some(match tag {
            "mandel" => FractalKind::Mandelbrot,
            "burning" => FractalKind::BurningShip,
            "tricorn" => FractalKind::Tricorn,
            "multi" => FractalKind::Multibrot,
            "celtic" => FractalKind::Celtic,
            "perp" => FractalKind::Perpendicular,
            "buffalo" => FractalKind::Buffalo,
            "phoenix" => FractalKind::Phoenix,
            "lambda" => FractalKind::Lambda,
            "cmulti" => FractalKind::ComplexMultibrot,
            _ => return None,
        })
    }

    /// Default parameter-plane (Mandelbrot-mode) view for this kind, as
    /// `(center_re, center_im, half_height)`. The Julia (dynamical) plane
    /// doesn't vary by kind, so it isn't covered here.
    pub fn default_set_view(&self) -> (f64, f64, f64) {
        match self {
            FractalKind::Mandelbrot => (-0.5, 0.0, 1.25),
            FractalKind::BurningShip => (-0.5, -0.5, 1.3),
            FractalKind::Tricorn => (-0.25, 0.0, 1.7),
            FractalKind::Multibrot => (0.0, 0.0, 1.5),
            FractalKind::Celtic => (-0.5, 0.0, 1.6),
            FractalKind::Perpendicular => (-0.5, 0.0, 1.5),
            FractalKind::Buffalo => (-0.5, 0.5, 1.5),
            FractalKind::Phoenix => (-0.5, 0.0, 1.5),
            FractalKind::Lambda => (-0.5, 0.0, 2.4),
            FractalKind::ComplexMultibrot => (0.0, 0.0, 1.5),
        }
    }
}
