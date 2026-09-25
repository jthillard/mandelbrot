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
//!
//! While switching fractal kinds, the formula is morphed *per iteration*:
//! `Z_{n+1} = (1 - w)·f_kind(Z_n) + w·f_from(Z_n)` (see `morph` below). The
//! map is linear in the two outputs, so the GPU delta is the same blend of
//! the two kinds' deltas and perturbation/rebasing keep working unchanged.

use super::kind::FractalKind;
use crate::view::{Big, big_from_f64};

/// Reference orbit escapes once |Z|^2 exceeds this. Kept larger than the pixel
/// bailout so pixels escaping alongside the reference can still reach their
/// bailout before the stored orbit runs out.
const REFERENCE_ESCAPE_SQ: f64 = 1.0e10;

/// Up to this working precision (bits) the orbit is iterated in plain `f64`
/// instead of `FBig` — orders of magnitude faster, which matters most on the
/// web (where the reference is computed inline on the UI thread).
///
/// `precision_for` asks for `zoom_bits + 48` guard bits, but the GPU only
/// consumes the orbit as f32 deltas, so two things actually matter:
/// * Each f64 step's rounding (~1e-16 relative) acts like a tiny local error
///   in the pixel orbits too (perturbation reproduces whatever orbit it's
///   given), far below the f32 delta noise — the orbit only has to be a
///   consistent orbit, not the exact one.
/// * The reference center gets rounded to f64 (<= ~2.2e-16 absolute for
///   |c| <= 2), which shifts the image. At 80 bits (zoom_bits <= 32, i.e.
///   half-height >= ~2.3e-10) a pixel is >= ~5e-13 wide, so that shift stays
///   below 0.1% of a pixel.
const F64_MAX_PRECISION: usize = 80;

/// Compute the reference orbit `Z_0..Z_{len-1}` where `Z_0 = z0` and
/// `Z_{n+1} = f(Z_n, c)` for the given `kind` (and `power`, for Multibrot), up
/// to `max_iter` steps at `precision` bits. Each entry is `[re, im]` in f32.
///
/// `morph = Some((from, w))` blends in a second kind's formula at every step:
/// `(1 - w)·f_kind + w·f_from` (used by the kind-switch animation).
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
    complex_power: (f64, f64),
    morph: Option<(FractalKind, f64)>,
) -> Vec<[f32; 2]> {
    // A zero-weight morph is just the plain kind; skip the second formula.
    let morph = morph.filter(|&(_, w)| w != 0.0);
    if precision <= F64_MAX_PRECISION {
        let k = StepConstsF64 {
            c: (c_re.to_f64().value(), c_im.to_f64().value()),
            p: phoenix_p,
            l: lambda_l,
            cpow: complex_power,
            power,
        };
        return compute_reference_f64(
            (z0_re.to_f64().value(), z0_im.to_f64().value()),
            max_iter,
            kind,
            &k,
            morph,
        );
    }
    let k = StepConsts {
        cr: c_re.clone().with_precision(precision).value(),
        ci: c_im.clone().with_precision(precision).value(),
        pr: big_from_f64(phoenix_p.0, precision),
        pi: big_from_f64(phoenix_p.1, precision),
        lr: big_from_f64(lambda_l.0, precision),
        li: big_from_f64(lambda_l.1, precision),
        cpow_re: big_from_f64(complex_power.0, precision),
        cpow_im: big_from_f64(complex_power.1, precision),
        power,
        precision,
    };
    compute_reference_big(z0_re, z0_im, max_iter, kind, &k, morph)
}

/// `f64` twin of [`StepConsts`].
struct StepConstsF64 {
    c: (f64, f64),
    p: (f64, f64),
    l: (f64, f64),
    cpow: (f64, f64),
    power: u32,
}

/// [`compute_reference`]'s fast path for shallow views (see
/// [`F64_MAX_PRECISION`]): the same per-kind formulas in plain `f64`.
fn compute_reference_f64(
    z0: (f64, f64),
    max_iter: u32,
    kind: FractalKind,
    k: &StepConstsF64,
    morph: Option<(FractalKind, f64)>,
) -> Vec<[f32; 2]> {
    let (mut zr, mut zi) = z0;
    // Previous iterate, for the Phoenix two-term recurrence (Y_{-1} = 0).
    let mut prev = (0.0f64, 0.0f64);

    let mut points: Vec<[f32; 2]> = Vec::with_capacity(max_iter as usize + 1);
    for _ in 0..=max_iter {
        points.push([zr as f32, zi as f32]);
        if zr * zr + zi * zi > REFERENCE_ESCAPE_SQ {
            break;
        }

        let (mut new_zr, mut new_zi) = step_f64(kind, k, zr, zi, prev);
        if let Some((from, w)) = morph {
            let (br, bi) = step_f64(from, k, zr, zi, prev);
            new_zr += w * (br - new_zr);
            new_zi += w * (bi - new_zi);
        }
        prev = (zr, zi);
        (zr, zi) = (new_zr, new_zi);
    }
    points
}

/// `f64` twin of [`step`]: one step `f(Z_n)` of `kind`'s formula.
fn step_f64(
    kind: FractalKind,
    k: &StepConstsF64,
    zr: f64,
    zi: f64,
    prev: (f64, f64),
) -> (f64, f64) {
    let (cr, ci) = k.c;
    match kind {
        FractalKind::Mandelbrot => ((zr + zi) * (zr - zi) + cr, 2.0 * zr * zi + ci),
        FractalKind::BurningShip => (zr * zr - zi * zi + cr, (2.0 * zr * zi).abs() + ci),
        FractalKind::Tricorn => (zr * zr - zi * zi + cr, ci - 2.0 * zr * zi),
        FractalKind::Multibrot => {
            let (mut rr, mut ri) = (1.0f64, 0.0f64);
            for _ in 0..k.power.max(2) {
                (rr, ri) = (rr * zr - ri * zi, rr * zi + ri * zr);
            }
            (rr + cr, ri + ci)
        }
        FractalKind::Celtic => ((zr * zr - zi * zi).abs() + cr, 2.0 * zr * zi + ci),
        FractalKind::Perpendicular => (zr * zr - zi * zi + cr, ci - 2.0 * zr * zi.abs()),
        FractalKind::Buffalo => ((zr * zr - zi * zi).abs() + cr, ci - (2.0 * zr * zi).abs()),
        FractalKind::Phoenix => {
            let (pr, pi) = k.p;
            let (zr_prev, zi_prev) = prev;
            (
                zr * zr - zi * zi + cr + (pr * zr_prev - pi * zi_prev),
                2.0 * zr * zi + ci + (pr * zi_prev + pi * zr_prev),
            )
        }
        FractalKind::Lambda => {
            // λ·z(1 - z) + c.
            let (lr, li) = k.l;
            let (re2, im2) = (1.0 - zr, -zi);
            let (lzr, lzi) = (lr * zr - li * zi, lr * zi + li * zr);
            (lzr * re2 - lzi * im2 + cr, re2 * lzi + lzr * im2 + ci)
        }
        FractalKind::ComplexMultibrot => {
            let (pr, pi) = complex_pow_complex_f64(zr, zi, k.cpow.0, k.cpow.1);
            (pr + cr, pi + ci)
        }
    }
}

/// `f64` twin of [`complex_pow_complex`] (principal branch, `0^p = 0`).
fn complex_pow_complex_f64(zr: f64, zi: f64, pr: f64, pi: f64) -> (f64, f64) {
    if zr == 0.0 && zi == 0.0 {
        return (0.0, 0.0);
    }
    let ln_r = 0.5 * (zr * zr + zi * zi).ln();
    let theta = zi.atan2(zr);
    let mag = (pr * ln_r - pi * theta).exp();
    let (sin_a, cos_a) = (pr * theta + pi * ln_r).sin_cos();
    (mag * cos_a, mag * sin_a)
}

/// Everything a single formula step needs besides the orbit state, converted
/// to `Big` once up front.
struct StepConsts {
    cr: Big,
    ci: Big,
    /// Phoenix distortion constant `p`.
    pr: Big,
    pi: Big,
    /// Lambda distortion constant `l`.
    lr: Big,
    li: Big,
    /// Complex Multibrot exponent.
    cpow_re: Big,
    cpow_im: Big,
    power: u32,
    precision: usize,
}

/// [`compute_reference`] at arbitrary precision (`FBig`), for deep views.
fn compute_reference_big(
    z0_re: &Big,
    z0_im: &Big,
    max_iter: u32,
    kind: FractalKind,
    k: &StepConsts,
    morph: Option<(FractalKind, f64)>,
) -> Vec<[f32; 2]> {
    let precision = k.precision;
    let morph = morph.map(|(from, w)| (from, big_from_f64(w, precision)));

    let mut zr = z0_re.clone().with_precision(precision).value();
    let mut zi = z0_im.clone().with_precision(precision).value();
    // Previous iterate, for the Phoenix two-term recurrence (Y_{-1} = 0).
    let mut zr_prev = big_zero(precision);
    let mut zi_prev = big_zero(precision);

    let mut points: Vec<[f32; 2]> = Vec::with_capacity(max_iter as usize + 1);

    for _ in 0..=max_iter {
        let fr = zr.to_f64().value() as f32;
        let fi = zi.to_f64().value() as f32;
        points.push([fr, fi]);

        let mag = (fr as f64) * (fr as f64) + (fi as f64) * (fi as f64);
        if mag > REFERENCE_ESCAPE_SQ {
            break;
        }

        let (mut new_zr, mut new_zi) = step(kind, k, &zr, &zi, &zr_prev, &zi_prev);
        if let Some((from, w)) = &morph {
            // (1 - w)·a + w·b = a + w·(b - a).
            let (br, bi) = step(*from, k, &zr, &zi, &zr_prev, &zi_prev);
            new_zr = &new_zr + &(w * &(br - &new_zr));
            new_zi = &new_zi + &(w * &(bi - &new_zi));
        }

        // Shift the previous iterate (only the Phoenix arm reads it).
        zr_prev = zr;
        zi_prev = zi;
        zr = new_zr.with_precision(precision).value();
        zi = new_zi.with_precision(precision).value();
    }

    points
}

/// One step `f(Z_n)` of `kind`'s formula (including its `+ c`), given the
/// current and previous iterate.
fn step(
    kind: FractalKind,
    k: &StepConsts,
    zr: &Big,
    zi: &Big,
    zr_prev: &Big,
    zi_prev: &Big,
) -> (Big, Big) {
    let (cr, ci) = (&k.cr, &k.ci);
    match kind {
        FractalKind::Mandelbrot => {
            // Z^2 = (zr^2 - zi^2) + (2 zr zi) i, with zr^2 - zi^2 as
            // (zr + zi)(zr - zi): one multiply instead of two squares.
            let re = (zr + zi) * (zr - zi) + cr;
            let im = ((zr * zi) << 1) + ci; // << 1 is exact ×2 in base 2
            (re, im)
        }
        FractalKind::BurningShip => {
            // (|zr| + i|zi|)^2 = (zr^2 - zi^2) + 2|zr zi| i.
            let re = &zr.sqr() - &zi.sqr() + cr;
            let im = big_abs((zr * zi) << 1) + ci;
            (re, im)
        }
        FractalKind::Tricorn => {
            // conj(z)^2 = (zr^2 - zi^2) - 2 zr zi i.
            let re = &zr.sqr() - &zi.sqr() + cr;
            let im = ci - ((zr * zi) << 1);
            (re, im)
        }
        FractalKind::Multibrot => {
            let (pr, pi) = complex_pow(zr, zi, k.power.max(2), k.precision);
            (pr + cr, pi + ci)
        }
        FractalKind::Celtic => {
            // |Re(z^2)| + i·Im(z^2): abs the real output of the square.
            let re = big_abs(&zr.sqr() - &zi.sqr()) + cr;
            let im = ((zr * zi) << 1) + ci;
            (re, im)
        }
        FractalKind::Perpendicular => {
            // (x^2 - y^2) - 2·x·|y| i: abs the imaginary input.
            let re = &zr.sqr() - &zi.sqr() + cr;
            let im = if zi.to_f64().value() < 0.0 {
                ci + ((zr * zi) << 1)
            } else {
                ci - ((zr * zi) << 1)
            };
            (re, im)
        }
        FractalKind::Buffalo => {
            // |Re(z^2)| - |Im(z^2)| i: abs both outputs.
            let re = big_abs(&zr.sqr() - &zi.sqr()) + cr;
            let im = ci - big_abs((zr * zi) << 1);
            (re, im)
        }
        FractalKind::Phoenix => {
            // z^2 + c + p·z_{n-1}.
            let re2 = &zr.sqr() - &zi.sqr();
            let im2 = (zr * zi) << 1;
            let pzr = &k.pr * zr_prev - &k.pi * zi_prev;
            let pzi = &k.pr * zi_prev + &k.pi * zr_prev;
            (re2 + cr + pzr, im2 + ci + pzi)
        }
        FractalKind::Lambda => {
            // λ·z(1 - z) + c: logistic map plus the usual additive `c`.
            let re2 = 1 - zr;
            let im2 = -zi;
            let lzr = &k.lr * zr - &k.li * zi;
            let lzi = &k.lr * zi + &k.li * zr;
            let re = &lzr * &re2 - &lzi * &im2;
            let im = re2 * lzi + lzr * im2;
            (re + cr, im + ci)
        }
        FractalKind::ComplexMultibrot => {
            let (pr, pi) = complex_pow_complex(zr, zi, &k.cpow_re, &k.cpow_im, k.precision);
            (pr + cr, pi + ci)
        }
    }
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

/// `true` if `x` is (numerically) zero. The f64 check is exact for a true
/// zero; only matters here to special-case `ln(0)`.
fn is_big_zero(x: &Big) -> bool {
    x.to_f64().value() == 0.0
}

/// `(zr + i zi)^(pr + i pi)` for a complex exponent, via the principal branch
/// `z^p = exp(p·ln z)` where `ln z = ln|z| + i·arg(z)`. Used by
/// `ComplexMultibrot`; must be kept in sync with the shader's `cpow`.
/// `z = 0` is special-cased to `0` (the formula's `ln(0)` would otherwise
/// panic; this is the correct limit for the `Re(p) > 0` region the UI
/// exposes).
fn complex_pow_complex(zr: &Big, zi: &Big, pr: &Big, pi: &Big, precision: usize) -> (Big, Big) {
    if is_big_zero(zr) && is_big_zero(zi) {
        return (big_zero(precision), big_zero(precision));
    }
    let r2 = &zr.sqr() + &zi.sqr();
    let ln_r = r2.ln() >> 1; // 0.5 * ln(r2) = ln(sqrt(r2)); exact halving.
    let theta = zi.atan2(zr);
    let exp_re = (pr * &ln_r - pi * &theta).with_precision(precision).value();
    let exp_im = (pr * &theta + pi * &ln_r).with_precision(precision).value();
    let mag = exp_re.exp();
    let (sin_a, cos_a) = exp_im.sin_cos();
    (&mag * &cos_a, &mag * &sin_a)
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
    complex_power: (f64, f64),
    morph: Option<(FractalKind, f64)>,
) -> Vec<[f32; 2]> {
    let zero = big_zero(precision);
    compute_reference(
        &zero,
        &zero,
        center_re,
        center_im,
        max_iter,
        precision,
        kind,
        power,
        phoenix_p,
        lambda_l,
        complex_power,
        morph,
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
            (0.0, 0.0),
            None,
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

    /// The f64 fast path (shallow views) must produce the same orbit as the
    /// arbitrary-precision path, for every kind, in both planes, with and
    /// without a kind-switch morph.
    #[test]
    fn f64_fast_path_matches_big() {
        let bits_fast = F64_MAX_PRECISION;
        let bits_big = F64_MAX_PRECISION + 64;
        for kind in FractalKind::ALL {
            for julia in [false, true] {
                for morph in [None, Some((FractalKind::Phoenix, 0.3))] {
                    let run = |bits: usize| {
                        let (a, b) = (big_from_f64(-0.3, bits), big_from_f64(0.2, bits));
                        let (jr, ji) = (big_from_f64(-0.4, bits), big_from_f64(0.55, bits));
                        let args = (60, bits, kind, 3, (0.1, -0.2), (0.9, 0.3), (2.3, 0.4));
                        if julia {
                            compute_reference(
                                &a, &b, &jr, &ji, args.0, args.1, args.2, args.3, args.4, args.5,
                                args.6, morph,
                            )
                        } else {
                            compute_set_reference(
                                &a, &b, args.0, args.1, args.2, args.3, args.4, args.5, args.6,
                                morph,
                            )
                        }
                    };
                    let ctx = format!("{kind:?} julia={julia} morph={morph:?}");
                    let (fast, big) = (run(bits_fast), run(bits_big));
                    assert_eq!(fast.len(), big.len(), "{ctx}: length");
                    for (i, (f, b)) in fast.iter().zip(&big).enumerate() {
                        for k in 0..2 {
                            let tol = 1e-5 * (1.0 + b[k].abs());
                            assert!(
                                (f[k] - b[k]).abs() <= tol,
                                "{ctx}: point {i} {f:?} vs {b:?}"
                            );
                        }
                    }
                }
            }
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
            (0.0, 0.0),
            None,
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
            (0.0, 0.0),
            None,
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
            (0.0, 0.0),
            None,
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
            (0.0, 0.0),
            None,
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

    /// Lambda Julia orbit adds the Julia `c`: `z -> λ·z(1 - z) + c`.
    #[test]
    fn lambda_julia_reference_matches_naive_f64() {
        let (lr, li) = (-0.5_f64, 0.2_f64);
        let (cr, ci) = (0.1_f64, -0.3_f64);
        let points = compute_reference(
            &Big::try_from(0.2_f64).unwrap(),
            &Big::try_from(0.1_f64).unwrap(),
            &Big::try_from(cr).unwrap(),
            &Big::try_from(ci).unwrap(),
            60,
            200,
            FractalKind::Lambda,
            2,
            (0.0, 0.0),
            (lr, li),
            (0.0, 0.0),
            None,
        );

        let (mut zr, mut zi) = (0.2_f64, 0.1_f64);
        for point in &points {
            let tol = 1e-4 * (1.0 + zr.abs().max(zi.abs()));
            assert!((point[0] as f64 - zr).abs() < tol, "{point:?} vs {zr}");
            assert!((point[1] as f64 - zi).abs() < tol, "{point:?} vs {zi}");
            let (lzr, lzi) = (lr * zr - li * zi, lr * zi + li * zr);
            let (ar, ai) = (1.0 - zr, -zi);
            zr = lzr * ar - lzi * ai + cr;
            zi = lzr * ai + lzi * ar + ci;
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
            (0.0, 0.0),
            None,
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
            (0.0, 0.0),
            None,
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
            (0.0, 0.0),
            None,
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
        let points = compute_set_reference(
            &cr,
            &ci,
            60,
            200,
            FractalKind::Phoenix,
            2,
            p,
            (0.0, 0.0),
            (0.0, 0.0),
            None,
        );

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

    /// Complex Multibrot (power 2.5 + 0.3i) reference matches a naive f64
    /// iteration of `z^p = exp(p·ln z)`.
    #[test]
    fn complex_multibrot_reference_matches_naive_f64() {
        let cr = Big::try_from(0.1_f64).unwrap();
        let ci = Big::try_from(-0.2_f64).unwrap();
        let power = (2.5_f64, 0.3_f64);
        let points = compute_set_reference(
            &cr,
            &ci,
            60,
            200,
            FractalKind::ComplexMultibrot,
            2,
            (0.0, 0.0),
            (0.0, 0.0),
            power,
            None,
        );

        // Naive f64 complex power via z^p = exp(p * ln z), ln z = ln|z| + i*arg(z).
        fn naive_cpow(zr: f64, zi: f64, pr: f64, pi: f64) -> (f64, f64) {
            if zr == 0.0 && zi == 0.0 {
                return (0.0, 0.0);
            }
            let ln_r = 0.5 * (zr * zr + zi * zi).ln();
            let theta = zi.atan2(zr);
            let exp_re = pr * ln_r - pi * theta;
            let exp_im = pr * theta + pi * ln_r;
            let mag = exp_re.exp();
            (mag * exp_im.cos(), mag * exp_im.sin())
        }

        let (c_re, c_im) = (0.1_f64, -0.2_f64);
        let (mut zr, mut zi) = (0.0_f64, 0.0_f64);
        for point in &points {
            let tol = 1e-4 * (1.0 + zr.abs().max(zi.abs()));
            assert!((point[0] as f64 - zr).abs() < tol, "re: {point:?} vs {zr}");
            assert!((point[1] as f64 - zi).abs() < tol, "im: {point:?} vs {zi}");
            let (pr, pi) = naive_cpow(zr, zi, power.0, power.1);
            let nzr = pr + c_re;
            let nzi = pi + c_im;
            zr = nzr;
            zi = nzi;
        }
    }

    fn set_ref(
        cr: f64,
        ci: f64,
        kind: FractalKind,
        morph: Option<(FractalKind, f64)>,
    ) -> Vec<[f32; 2]> {
        compute_set_reference(
            &Big::try_from(cr).unwrap(),
            &Big::try_from(ci).unwrap(),
            60,
            200,
            kind,
            2,
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            morph,
        )
    }

    /// Morph weight 0 is the plain kind; weight 1 is entirely the from-kind.
    #[test]
    fn morph_endpoints_match_plain_kinds() {
        let (cr, ci) = (-1.75, -0.03);
        let ship = set_ref(cr, ci, FractalKind::BurningShip, None);
        let mandel = set_ref(cr, ci, FractalKind::Mandelbrot, None);
        let w0 = set_ref(
            cr,
            ci,
            FractalKind::BurningShip,
            Some((FractalKind::Mandelbrot, 0.0)),
        );
        let w1 = set_ref(
            cr,
            ci,
            FractalKind::BurningShip,
            Some((FractalKind::Mandelbrot, 1.0)),
        );
        assert_eq!(w0, ship);
        assert_eq!(w1, mandel);
    }

    /// A half-way Mandelbrot / Burning Ship morph matches a naive f64
    /// iteration of the per-step blend.
    #[test]
    fn morph_blend_matches_naive_f64() {
        let (c_re, c_im) = (-0.6_f64, 0.3_f64);
        let w = 0.5_f64;
        let points = set_ref(
            c_re,
            c_im,
            FractalKind::Mandelbrot,
            Some((FractalKind::BurningShip, w)),
        );

        let (mut zr, mut zi) = (0.0_f64, 0.0_f64);
        for point in &points {
            let tol = 1e-4 * (1.0 + zr.abs().max(zi.abs()));
            assert!((point[0] as f64 - zr).abs() < tol, "re: {point:?} vs {zr}");
            assert!((point[1] as f64 - zi).abs() < tol, "im: {point:?} vs {zi}");
            let re = zr * zr - zi * zi + c_re; // identical for both kinds
            let im_m = 2.0 * zr * zi + c_im;
            let im_b = 2.0 * (zr * zi).abs() + c_im;
            zr = re;
            zi = (1.0 - w) * im_m + w * im_b;
        }
    }
}
