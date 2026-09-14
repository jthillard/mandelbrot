//! High-precision reference-orbit computation for perturbation rendering.
//!
//! We iterate `Z_{n+1} = Z_n^2 + C` at high precision (`dashu-float`), storing
//! each `Z_n` as an `f32` pair. Every pixel is then rendered on the GPU as a
//! small `f32` delta from this orbit — that is what makes deep zoom cheap. See
//! `shaders/mandelbrot.wgsl` for the delta side.
//!
//! The `(z0, c)` form serves both fractals:
//! * Mandelbrot: `z0 = 0`, `c = view center` (the c-plane point per pixel).
//! * Julia:      `z0 = view center`, `c = julia constant` (fixed for all pixels).

use crate::view::Big;

/// Reference orbit escapes once |Z|^2 exceeds this. Kept larger than the pixel
/// bailout so pixels escaping alongside the reference can still reach their
/// bailout before the stored orbit runs out.
const REFERENCE_ESCAPE_SQ: f64 = 1.0e10;

/// Compute the reference orbit `Z_0..Z_{len-1}` where `Z_0 = z0` and
/// `Z_{n+1} = Z_n^2 + c`, up to `max_iter` steps at `precision` bits. Each entry
/// is `[re, im]` in f32.
pub fn compute_reference(
    z0_re: &Big,
    z0_im: &Big,
    c_re: &Big,
    c_im: &Big,
    max_iter: u32,
    precision: usize,
) -> Vec<[f32; 2]> {
    let cr = c_re.clone().with_precision(precision).value();
    let ci = c_im.clone().with_precision(precision).value();

    let mut zr = z0_re.clone().with_precision(precision).value();
    let mut zi = z0_im.clone().with_precision(precision).value();

    let mut points: Vec<[f32; 2]> = Vec::with_capacity(max_iter as usize + 1);

    for _ in 0..=max_iter {
        let fr = zr.to_f64().value() as f32;
        let fi = zi.to_f64().value() as f32;
        points.push([fr, fi]);

        let mag = (fr as f64) * (fr as f64) + (fi as f64) * (fi as f64);
        if mag > REFERENCE_ESCAPE_SQ {
            break;
        }

        // Z = Z^2 + C, with Z^2 = (zr^2 - zi^2) + (2 zr zi) i.
        let zr2 = zr.sqr();
        let zi2 = zi.sqr();
        let new_zr = ((&zr2 - &zi2) + &cr).with_precision(precision).value();
        let two_zr_zi = (&zr * &zi) << 1; // exact multiply-by-2 in base 2
        let new_zi = (two_zr_zi + &ci).with_precision(precision).value();

        zr = new_zr;
        zi = new_zi;
    }

    points
}

fn big_zero(precision: usize) -> Big {
    Big::from(0i32).with_precision(precision).value()
}

/// Convenience: Mandelbrot reference (`z0 = 0`, `c = center`).
pub fn compute_mandelbrot_reference(
    center_re: &Big,
    center_im: &Big,
    max_iter: u32,
    precision: usize,
) -> Vec<[f32; 2]> {
    let zero = big_zero(precision);
    compute_reference(&zero, &zero, center_re, center_im, max_iter, precision)
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
        let points = compute_mandelbrot_reference(&cr, &ci, 60, 200);

        // Independent naive f64 orbit.
        let (c_re, c_im) = (-0.75_f64, 0.1_f64);
        let (mut zr, mut zi) = (0.0_f64, 0.0_f64);
        for point in &points {
            // Tolerance is relative to magnitude: f32 storage only keeps ~7
            // significant figures.
            let tol_re = 1e-4 * (1.0 + zr.abs());
            let tol_im = 1e-4 * (1.0 + zi.abs());
            assert!((point[0] as f64 - zr).abs() < tol_re, "re mismatch: {point:?} vs {zr}");
            assert!((point[1] as f64 - zi).abs() < tol_im, "im mismatch: {point:?} vs {zi}");
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
        let points = compute_mandelbrot_reference(&cr, &ci, 500, 120);
        assert_eq!(points.len(), 501, "interior orbit should not escape");
    }

    /// Julia orbit (fixed c, z0 = center) matches a naive f64 iteration.
    #[test]
    fn julia_reference_matches_naive_f64() {
        let z0_re = Big::try_from(0.15_f64).unwrap();
        let z0_im = Big::try_from(-0.1_f64).unwrap();
        let c_re = Big::try_from(-0.8_f64).unwrap();
        let c_im = Big::try_from(0.156_f64).unwrap();
        let points = compute_reference(&z0_re, &z0_im, &c_re, &c_im, 60, 200);

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
}
