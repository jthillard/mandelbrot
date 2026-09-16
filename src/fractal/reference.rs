//! High-precision reference-orbit computation for perturbation rendering.
//!
//! We iterate the fractal's formula `Z_{n+1} = f(Z_n, C)` at high precision
//! (`dashu-float`), storing each `Z_n` as an `f32` pair. Every pixel is then
//! rendered on the GPU as a small `f32` delta from this orbit — that is what
//! makes deep zoom cheap. See `shaders/mandelbrot.wgsl` for the delta side; the
//! delta formula there must match the orbit formula here.
//!
//! The `(z0, c)` form serves both set types:
//! * Mandelbrot-set: `z0 = 0`, `c = view center` (the c-plane point per pixel).
//! * Julia-set:      `z0 = view center`, `c = fractal constant` (fixed per view).

use crate::view::{Big, big_from_f64};

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
    /// `z -> lambda·z(1 - z)` (logistic map).
    Lambda = 8,
}

/// Reference orbit escapes once |Z|^2 exceeds this. Kept larger than the pixel
/// bailout so pixels escaping alongside the reference can still reach their
/// bailout before the stored orbit runs out.
const REFERENCE_ESCAPE_SQ: f64 = 1.0e10;

/// Compute the reference orbit `Z_0..Z_{len-1}` where `Z_0 = z0` and
/// `Z_{n+1} = f(Z_n, c)` for the given `kind` (and `power`, for Multibrot), up
/// to `max_iter` steps at `precision` bits. Each entry is `[re, im]` in f32.
#[allow(clippy::too_many_arguments)]
pub fn compute_reference(
    z0_re: &Big,
    z0_im: &Big,
    c_re: &Big,
    c_im: &Big,
    max_iter: u32,
    precision: usize,
    kind: FractalKind,
    power: u32,
    phoenix_p: (f64, f64),
    lambda_l: (f64, f64),
) -> Vec<[f32; 2]> {
    let cr = c_re.clone().with_precision(precision).value();
    let ci = c_im.clone().with_precision(precision).value();

    let mut zr = z0_re.clone().with_precision(precision).value();
    let mut zi = z0_im.clone().with_precision(precision).value();
    // Previous iterate, for the Phoenix two-term recurrence (Y_{-1} = 0).
    let mut zr_prev = big_zero(precision);
    let mut zi_prev = big_zero(precision);
    // Phoenix distortion constant `p` (a small fixed complex number).
    let pr = big_from_f64(phoenix_p.0, precision);
    let pi = big_from_f64(phoenix_p.1, precision);
    // Lambda distortion constant `l` (a small fixed complex number).
    let lr = big_from_f64(lambda_l.0, precision);
    let li = big_from_f64(lambda_l.1, precision);

    let mut points: Vec<[f32; 2]> = Vec::with_capacity(max_iter as usize + 1);

    for _ in 0..=max_iter {
        let fr = zr.to_f64().value() as f32;
        let fi = zi.to_f64().value() as f32;
        points.push([fr, fi]);

        let mag = (fr as f64) * (fr as f64) + (fi as f64) * (fi as f64);
        if mag > REFERENCE_ESCAPE_SQ {
            break;
        }

        let (new_zr, new_zi) = match kind {
            FractalKind::Mandelbrot => {
                // Z^2 = (zr^2 - zi^2) + (2 zr zi) i.
                let re = &zr.sqr() - &zi.sqr() + &cr;
                let im = ((&zr * &zi) << 1) + &ci; // << 1 is exact ×2 in base 2
                (re, im)
            }
            FractalKind::BurningShip => {
                // (|zr| + i|zi|)^2 = (zr^2 - zi^2) + 2|zr zi| i.
                let re = &zr.sqr() - &zi.sqr() + &cr;
                let im = big_abs((&zr * &zi) << 1) + &ci;
                (re, im)
            }
            FractalKind::Tricorn => {
                // conj(z)^2 = (zr^2 - zi^2) - 2 zr zi i.
                let re = &zr.sqr() - &zi.sqr() + &cr;
                let im = &ci - ((&zr * &zi) << 1);
                (re, im)
            }
            FractalKind::Multibrot => {
                let (pr, pi) = complex_pow(&zr, &zi, power.max(2), precision);
                (pr + &cr, pi + &ci)
            }
            FractalKind::Celtic => {
                // |Re(z^2)| + i·Im(z^2): abs the real output of the square.
                let re = big_abs(&zr.sqr() - &zi.sqr()) + &cr;
                let im = ((&zr * &zi) << 1) + &ci;
                (re, im)
            }
            FractalKind::Perpendicular => {
                // (x^2 - y^2) - 2·x·|y| i: abs the imaginary input.
                let re = &zr.sqr() - &zi.sqr() + &cr;
                let im = &ci - ((&zr * &big_abs(zi.clone())) << 1);
                (re, im)
            }
            FractalKind::Buffalo => {
                // |Re(z^2)| - |Im(z^2)| i: abs both outputs.
                let re = big_abs(&zr.sqr() - &zi.sqr()) + &cr;
                let im = &ci - big_abs((&zr * &zi) << 1);
                (re, im)
            }
            FractalKind::Phoenix => {
                // z^2 + c + p·z_{n-1}.
                let re2 = &zr.sqr() - &zi.sqr();
                let im2 = (&zr * &zi) << 1;
                let pzr = &pr * &zr_prev - &pi * &zi_prev;
                let pzi = &pr * &zi_prev + &pi * &zr_prev;
                (re2 + &cr + pzr, im2 + &ci + pzi)
            }
            FractalKind::Lambda => {
                // λ·z(1 - z): logistic map.
                let re2 = 1 - &zr;
                let im2 = -&zi;
                let lzr = &lr * &zr - &li * &zi;
                let lzi = &lr * &zi + &li * &zr;
                (&lzr * &re2 - &lzi * &im2, re2 * lzi + lzr * im2)
            }
        };

        // Shift the previous iterate (only the Phoenix arm reads it).
        zr_prev = zr;
        zi_prev = zi;
        zr = new_zr.with_precision(precision).value();
        zi = new_zi.with_precision(precision).value();
    }

    points
}

fn big_zero(precision: usize) -> Big {
    Big::from(0i32).with_precision(precision).value()
}

/// Absolute value of a `Big`. The sign check via f64 is exact except for values
/// so tiny that |x| ≈ x either way — negligible against the f32 orbit storage.
fn big_abs(x: Big) -> Big {
    if x.to_f64().value() < 0.0 { -x } else { x }
}

/// `(zr + i zi)^power` by repeated complex multiply at `precision` bits.
fn complex_pow(zr: &Big, zi: &Big, power: u32, precision: usize) -> (Big, Big) {
    let mut rr = Big::from(1i32).with_precision(precision).value();
    let mut ri = big_zero(precision);
    for _ in 0..power {
        // (rr + i ri)(zr + i zi) = (rr zr - ri zi) + (rr zi + ri zr) i.
        let nr = (&rr * zr - &ri * zi).with_precision(precision).value();
        let ni = (&rr * zi + &ri * zr).with_precision(precision).value();
        rr = nr;
        ri = ni;
    }
    (rr, ri)
}

/// Convenience: parameter-plane ("Mandelbrot-set") reference (`z0 = 0`,
/// `c = center`) for any `kind`.
#[allow(clippy::too_many_arguments)]
pub fn compute_set_reference(
    center_re: &Big,
    center_im: &Big,
    max_iter: u32,
    precision: usize,
    kind: FractalKind,
    power: u32,
    phoenix_p: (f64, f64),
    lambda_l: (f64, f64),
) -> Vec<[f32; 2]> {
    let zero = big_zero(precision);
    compute_reference(
        &zero, &zero, center_re, center_im, max_iter, precision, kind, power, phoenix_p, lambda_l,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The high-precision reference must agree with a plain f64 iteration for a
    /// shallow point (where f64 is accurate).
    #[test]
    fn reference_matches_naive_f64() {
        let cr = Big::try_from(-0.75_f64).unwrap();
        let ci = Big::try_from(0.1_f64).unwrap();
        let points = compute_set_reference(
            &cr,
            &ci,
            60,
            200,
            FractalKind::Mandelbrot,
            2,
            (0.0, 0.0),
            (0.0, 0.0),
        );

        // Independent naive f64 orbit.
        let (c_re, c_im) = (-0.75_f64, 0.1_f64);
        let (mut zr, mut zi) = (0.0_f64, 0.0_f64);
        for point in &points {
            // Tolerance is relative to magnitude: f32 storage only keeps ~7
            // significant figures.
            let tol_re = 1e-4 * (1.0 + zr.abs());
            let tol_im = 1e-4 * (1.0 + zi.abs());
            assert!(
                (point[0] as f64 - zr).abs() < tol_re,
                "re mismatch: {point:?} vs {zr}"
            );
            assert!(
                (point[1] as f64 - zi).abs() < tol_im,
                "im mismatch: {point:?} vs {zi}"
            );
            let nzr = zr * zr - zi * zi + c_re;
            let nzi = 2.0 * zr * zi + c_im;
            zr = nzr;
            zi = nzi;
        }
    }

    /// A point inside the main cardioid never escapes: full-length orbit.
    #[test]
    fn interior_orbit_runs_full_length() {
        let cr = Big::try_from(-0.2_f64).unwrap();
        let ci = Big::try_from(0.0_f64).unwrap();
        let points = compute_set_reference(
            &cr,
            &ci,
            500,
            120,
            FractalKind::Mandelbrot,
            2,
            (0.0, 0.0),
            (0.0, 0.0),
        );
        assert_eq!(points.len(), 501, "interior orbit should not escape");
    }

    /// Burning Ship reference matches a naive f64 iteration of the same formula.
    #[test]
    fn burning_ship_reference_matches_naive_f64() {
        let cr = Big::try_from(-1.75_f64).unwrap();
        let ci = Big::try_from(-0.03_f64).unwrap();
        let points = compute_set_reference(
            &cr,
            &ci,
            60,
            200,
            FractalKind::BurningShip,
            2,
            (0.0, 0.0),
            (0.0, 0.0),
        );

        let (c_re, c_im) = (-1.75_f64, -0.03_f64);
        let (mut zr, mut zi) = (0.0_f64, 0.0_f64);
        for point in &points {
            let tol = 1e-4 * (1.0 + zr.abs().max(zi.abs()));
            assert!((point[0] as f64 - zr).abs() < tol, "re: {point:?} vs {zr}");
            assert!((point[1] as f64 - zi).abs() < tol, "im: {point:?} vs {zi}");
            let nzr = zr * zr - zi * zi + c_re;
            let nzi = 2.0 * (zr * zi).abs() + c_im;
            zr = nzr;
            zi = nzi;
        }
    }

    /// Multibrot (power 3) reference matches a naive f64 cube iteration.
    #[test]
    fn multibrot3_reference_matches_naive_f64() {
        let cr = Big::try_from(0.3_f64).unwrap();
        let ci = Big::try_from(0.2_f64).unwrap();
        let points = compute_set_reference(
            &cr,
            &ci,
            60,
            200,
            FractalKind::Multibrot,
            3,
            (0.0, 0.0),
            (0.0, 0.0),
        );

        let (c_re, c_im) = (0.3_f64, 0.2_f64);
        let (mut zr, mut zi) = (0.0_f64, 0.0_f64);
        for point in &points {
            let tol = 1e-4 * (1.0 + zr.abs().max(zi.abs()));
            assert!((point[0] as f64 - zr).abs() < tol, "re: {point:?} vs {zr}");
            assert!((point[1] as f64 - zi).abs() < tol, "im: {point:?} vs {zi}");
            // z^3 = z * z^2.
            let (r2, i2) = (zr * zr - zi * zi, 2.0 * zr * zi);
            let nzr = zr * r2 - zi * i2 + c_re;
            let nzi = zr * i2 + zi * r2 + c_im;
            zr = nzr;
            zi = nzi;
        }
    }

    /// Julia orbit (fixed c, z0 = center) matches a naive f64 iteration.
    #[test]
    fn julia_reference_matches_naive_f64() {
        let z0_re = Big::try_from(0.15_f64).unwrap();
        let z0_im = Big::try_from(-0.1_f64).unwrap();
        let c_re = Big::try_from(-0.8_f64).unwrap();
        let c_im = Big::try_from(0.156_f64).unwrap();
        let points = compute_reference(
            &z0_re,
            &z0_im,
            &c_re,
            &c_im,
            60,
            200,
            FractalKind::Mandelbrot,
            2,
            (0.0, 0.0),
            (0.0, 0.0),
        );

        let (mut zr, mut zi) = (0.15_f64, -0.1_f64);
        let (cr, ci) = (-0.8_f64, 0.156_f64);
        for point in &points {
            let tol = 1e-4 * (1.0 + zr.abs().max(zi.abs()));
            assert!((point[0] as f64 - zr).abs() < tol);
            assert!((point[1] as f64 - zi).abs() < tol);
            let nzr = zr * zr - zi * zi + cr;
            let nzi = 2.0 * zr * zi + ci;
            zr = nzr;
            zi = nzi;
        }
    }

    /// Celtic reference matches a naive f64 iteration: real = |x^2 - y^2| + cr.
    #[test]
    fn celtic_reference_matches_naive_f64() {
        let cr = Big::try_from(-0.6_f64).unwrap();
        let ci = Big::try_from(0.4_f64).unwrap();
        let points = compute_set_reference(
            &cr,
            &ci,
            60,
            200,
            FractalKind::Celtic,
            2,
            (0.0, 0.0),
            (0.0, 0.0),
        );

        let (c_re, c_im) = (-0.6_f64, 0.4_f64);
        let (mut zr, mut zi) = (0.0_f64, 0.0_f64);
        for point in &points {
            let tol = 1e-4 * (1.0 + zr.abs().max(zi.abs()));
            assert!((point[0] as f64 - zr).abs() < tol, "re: {point:?} vs {zr}");
            assert!((point[1] as f64 - zi).abs() < tol, "im: {point:?} vs {zi}");
            let nzr = (zr * zr - zi * zi).abs() + c_re;
            let nzi = 2.0 * zr * zi + c_im;
            zr = nzr;
            zi = nzi;
        }
    }

    /// Perpendicular reference matches a naive f64 iteration:
    /// real = x^2 - y^2 + cr, imag = -2·x·|y| + ci.
    #[test]
    fn perpendicular_reference_matches_naive_f64() {
        let cr = Big::try_from(-0.7_f64).unwrap();
        let ci = Big::try_from(-0.2_f64).unwrap();
        let points = compute_set_reference(
            &cr,
            &ci,
            60,
            200,
            FractalKind::Perpendicular,
            2,
            (0.0, 0.0),
            (0.0, 0.0),
        );

        let (c_re, c_im) = (-0.7_f64, -0.2_f64);
        let (mut zr, mut zi) = (0.0_f64, 0.0_f64);
        for point in &points {
            let tol = 1e-4 * (1.0 + zr.abs().max(zi.abs()));
            assert!((point[0] as f64 - zr).abs() < tol, "re: {point:?} vs {zr}");
            assert!((point[1] as f64 - zi).abs() < tol, "im: {point:?} vs {zi}");
            let nzr = zr * zr - zi * zi + c_re;
            let nzi = -2.0 * zr * zi.abs() + c_im;
            zr = nzr;
            zi = nzi;
        }
    }

    /// Buffalo reference matches a naive f64 iteration:
    /// real = |x^2 - y^2| + cr, imag = -|2·x·y| + ci.
    #[test]
    fn buffalo_reference_matches_naive_f64() {
        let cr = Big::try_from(-1.2_f64).unwrap();
        let ci = Big::try_from(-0.35_f64).unwrap();
        let points = compute_set_reference(
            &cr,
            &ci,
            60,
            200,
            FractalKind::Buffalo,
            2,
            (0.0, 0.0),
            (0.0, 0.0),
        );

        let (c_re, c_im) = (-1.2_f64, -0.35_f64);
        let (mut zr, mut zi) = (0.0_f64, 0.0_f64);
        for point in &points {
            let tol = 1e-4 * (1.0 + zr.abs().max(zi.abs()));
            assert!((point[0] as f64 - zr).abs() < tol, "re: {point:?} vs {zr}");
            assert!((point[1] as f64 - zi).abs() < tol, "im: {point:?} vs {zi}");
            let nzr = (zr * zr - zi * zi).abs() + c_re;
            let nzi = -(2.0 * zr * zi).abs() + c_im;
            zr = nzr;
            zi = nzi;
        }
    }

    /// Phoenix reference matches a naive f64 two-term iteration
    /// `z_{n+1} = z_n^2 + c + p·z_{n-1}` (z_0 = 0, z_{-1} = 0).
    #[test]
    fn phoenix_reference_matches_naive_f64() {
        let cr = Big::try_from(0.5667_f64).unwrap();
        let ci = Big::try_from(0.0_f64).unwrap();
        let p = (-0.5_f64, 0.0_f64);
        let points =
            compute_set_reference(&cr, &ci, 60, 200, FractalKind::Phoenix, 2, p, (0.0, 0.0));

        let (c_re, c_im) = (0.5667_f64, 0.0_f64);
        let (mut zr, mut zi) = (0.0_f64, 0.0_f64);
        let (mut pr, mut pi) = (0.0_f64, 0.0_f64); // previous iterate
        for point in &points {
            let tol = 1e-4 * (1.0 + zr.abs().max(zi.abs()));
            assert!((point[0] as f64 - zr).abs() < tol, "re: {point:?} vs {zr}");
            assert!((point[1] as f64 - zi).abs() < tol, "im: {point:?} vs {zi}");
            // p·z_{n-1} = (p.0 + i p.1)(pr + i pi).
            let pzr = p.0 * pr - p.1 * pi;
            let pzi = p.0 * pi + p.1 * pr;
            let nzr = zr * zr - zi * zi + c_re + pzr;
            let nzi = 2.0 * zr * zi + c_im + pzi;
            pr = zr;
            pi = zi;
            zr = nzr;
            zi = nzi;
        }
    }
}
