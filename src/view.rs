//! Camera / view state over the complex plane.
//!
//! The center is stored in arbitrary precision (`FBig`) — this is what lets us
//! zoom far past f64's ~1e13x limit. The pixel *scale* is a [`Scale`]: an
//! f64 mantissa with its own `i32` binary exponent, so it isn't bound by
//! f64's ~1e-308 range either (the floor, `Scale::MIN`, only keeps the GPU's
//! i32 exponent arithmetic far from overflow). Once a pixel is smaller than
//! `DEEP_PIXEL_SIZE` the GPU switches to rescaled deltas (see `needs_deep`),
//! since f32 alone bottoms out near 1e-38.

use core::str::FromStr;

use dashu_float::round::mode::HalfAway;
use dashu_float::{DBig, FBig};

/// Arbitrary-precision binary float (base 2, round-half-away). One coordinate.
pub type Big = FBig<HalfAway, 2>;

/// Half-height (complex units) of the default view; also the zoom-1 reference.
pub const DEFAULT_HALF_HEIGHT: f64 = 1.25;

/// Below this pixel size (complex units per pixel) the GPU renders with the
/// deep pipeline, whose per-pixel deltas start out as an f32 mantissa times
/// `2^scale_exp`. Plain f32 stays exact as long as the smallest per-pixel
/// offsets (a quarter pixel, for the AA grid) are normal floats (>= 2^-126),
/// i.e. down to 2^-124 per pixel. Measured: pixel-identical to the deep path
/// down to 2^-124 (no AA), first errors at 2^-126, all black by 2^-136. The
/// deep path is slower, so the switch is as late as that allows, with two
/// binades of margin.
pub const DEEP_PIXEL_SIZE: f64 = 1.0 / (1u128 << 122) as f64; // 2^-122

/// Whether a view rendered `height_px` pixels tall needs the deep pipeline
/// (see `DEEP_PIXEL_SIZE`).
pub fn needs_deep(half_height: Scale, height_px: f64) -> bool {
    half_height.mul_f64(2.0 / height_px.max(1.0)) < Scale::from_f64(DEEP_PIXEL_SIZE)
}

/// Binary exponent `E` of the deep view scale: `floor(log2(half_height))`,
/// so the rescaled span is in `[2, 4)`. Never 0, which means "not deep"
/// (see `Uniforms::scale_exp`).
pub fn deep_scale_exp(half_height: Scale) -> i32 {
    let e = half_height.exponent();
    if e == 0 { -1 } else { e }
}

/// `x * 2^k` for any `k`, saturating to 0 / infinity like the true value
/// would (`powi` alone overflows at 2^±1024 even when the product fits).
fn ldexp(mut x: f64, mut k: i32) -> f64 {
    while k > 1000 {
        x *= 2f64.powi(1000);
        k -= 1000;
        if x.is_infinite() || x == 0.0 {
            return x;
        }
    }
    while k < -1000 {
        x *= 2f64.powi(-1000);
        k += 1000;
        if x == 0.0 || x.is_infinite() {
            return x;
        }
    }
    x * 2f64.powi(k)
}

/// A positive real with f64 precision and an `i32` binary exponent:
/// `m · 2^e`, `m` in `[1, 2)`. The view's half-height (and the pixel size
/// derived from it) is one of these, so zoom isn't bound by f64's range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scale {
    m: f64,
    e: i32,
}

impl Scale {
    /// Deepest scale the view can reach: 2^-(2^20) (about 1e-315653). Far
    /// past anything a reference orbit can practically be computed for; it
    /// only keeps the shader's i32 exponent sums (scale × degree) from
    /// overflowing.
    pub const MIN: Scale = Scale {
        m: 1.0,
        e: -(1 << 20),
    };

    /// `m · 2^e`, normalized. Non-positive or NaN input gives `MIN`.
    pub fn from_parts(m: f64, e: i32) -> Self {
        if m.is_nan() || m <= 0.0 {
            return Self::MIN;
        }
        if m.is_infinite() {
            return Scale {
                m: 1.0,
                e: i32::MAX / 2,
            };
        }
        // Bring m into [1, 2) through its own binary exponent (exact).
        let k = m.log2().floor() as i32;
        let mut m = ldexp(m, -k);
        let mut e = e.saturating_add(k);
        // log2 can round across a power of two.
        if m >= 2.0 {
            m /= 2.0;
            e = e.saturating_add(1);
        } else if m < 1.0 {
            m *= 2.0;
            e = e.saturating_sub(1);
        }
        Scale { m, e }.max(Self::MIN)
    }

    pub fn from_f64(x: f64) -> Self {
        Self::from_parts(x, 0)
    }

    /// `2^l`.
    pub fn from_log2(l: f64) -> Self {
        let e = l.floor();
        Self::from_parts((l - e).exp2(), e as i32)
    }

    /// The value as an f64 (0 or infinity outside its range).
    pub fn to_f64(self) -> f64 {
        ldexp(self.m, self.e)
    }

    /// `self · 2^k` as an f64: the value in units of `2^-k`.
    pub fn scaled_f64(self, k: i32) -> f64 {
        ldexp(self.m, self.e.saturating_add(k))
    }

    /// `floor(log2(self))`.
    pub fn exponent(self) -> i32 {
        self.e
    }

    pub fn log2(self) -> f64 {
        self.m.log2() + self.e as f64
    }

    pub fn log10(self) -> f64 {
        self.log2() * core::f64::consts::LOG10_2
    }

    /// `self · f` (`f > 0`).
    pub fn mul_f64(self, f: f64) -> Self {
        Self::from_parts(self.m * f, self.e)
    }

    /// `self / other`, as an f64.
    pub fn ratio(self, other: Scale) -> f64 {
        ldexp(self.m / other.m, self.e.saturating_sub(other.e))
    }

    pub fn max(self, other: Scale) -> Self {
        if other > self { other } else { self }
    }

    pub fn min(self, other: Scale) -> Self {
        if other < self { other } else { self }
    }

    pub fn clamp(self, lo: Scale, hi: Scale) -> Self {
        self.max(lo).min(hi)
    }

    /// `f · self` as an exact `Big` at `bits` of precision (`f` any f64).
    pub fn big_times(self, f: f64, bits: usize) -> Big {
        big_from_f64(f * self.m, bits) << self.e as isize
    }

    /// Exact binary value as a `Big`.
    fn to_big(self) -> Big {
        big_from_f64(self.m, 53) << self.e as isize
    }
}

impl PartialOrd for Scale {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        // Normalized and positive: the exponent decides, then the mantissa.
        Some(self.e.cmp(&other.e).then(self.m.partial_cmp(&other.m)?))
    }
}

impl core::fmt::Display for Scale {
    /// Scientific notation, `1.5e-20` / `3.7e-4000`. The precision flag
    /// (`{:.4}`) sets mantissa digits after the point; without it, enough
    /// digits to parse back to the same value.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let x = self.to_f64();
        if x.is_normal() {
            return match f.precision() {
                Some(p) => write!(f, "{x:.p$e}"),
                None => write!(f, "{x:e}"),
            };
        }
        // Out of f64's range: round the exact decimal expansion instead.
        let sig = f.precision().map_or(17, |p| p + 1);
        let dec = self
            .to_big()
            .to_decimal()
            .value()
            .with_precision(sig)
            .value();
        let repr = dec.repr();
        let digits = repr.significand().to_string();
        let digits = digits.trim_end_matches('0');
        let digits = if digits.is_empty() { "0" } else { digits };
        // value = significand · 10^exponent; move the point after the first digit.
        let exp10 = repr.exponent() + repr.significand().to_string().len() as isize - 1;
        let (head, tail) = digits.split_at(1);
        let tail = match f.precision() {
            Some(p) => format!("{tail:0<p$}"),
            None => tail.to_string(),
        };
        if tail.is_empty() {
            write!(f, "{head}e{exp10}")
        } else {
            write!(f, "{head}.{tail}e{exp10}")
        }
    }
}

impl FromStr for Scale {
    type Err = ();

    /// Parses any positive decimal (`1.25`, `1.5e-20`, `3.7e-4000`), rounding
    /// to the nearest f64 mantissa. Values past `MIN` clamp to it.
    fn from_str(s: &str) -> Result<Self, ()> {
        let s = s.trim();
        if let Ok(x) = s.parse::<f64>()
            && x.is_normal()
        {
            return if x > 0.0 {
                Ok(Self::from_f64(x))
            } else {
                Err(())
            };
        }
        // Too small (or large) for f64: go through an exact decimal.
        let dec = DBig::from_str(s).map_err(|_| ())?;
        let bin: Big = dec.with_base_and_precision::<2>(64).value();
        if bin < Big::ZERO {
            return Err(());
        }
        let repr = bin.repr();
        let digits = repr.digits();
        if digits == 0 {
            return Err(());
        }
        let top = repr.exponent() + digits as isize - 1;
        let m = (bin.clone() >> top).to_f64().value();
        let e = top.clamp(i32::MIN as isize, i32::MAX as isize) as i32;
        Ok(Self::from_parts(m, e))
    }
}

/// Guard bits added on top of the zoom-dictated precision.
const GUARD_BITS: usize = 48;
/// Upper bound on center precision: what `Scale::MIN` needs. Only a guard
/// against pathological input; the reference orbit is impractically slow
/// long before this.
pub const MAX_PRECISION_BITS: usize = (1 << 20) + GUARD_BITS;

#[derive(Clone, Debug)]
pub struct ViewState {
    pub center_re: Big,
    pub center_im: Big,
    /// Half the view height in complex-plane units. Zooming in shrinks this.
    pub half_height: Scale,
}

impl Default for ViewState {
    fn default() -> Self {
        let bits = precision_for(Scale::from_f64(DEFAULT_HALF_HEIGHT));
        Self {
            center_re: big_from_f64(-0.5, bits),
            center_im: big_from_f64(0.0, bits),
            half_height: Scale::from_f64(DEFAULT_HALF_HEIGHT),
        }
    }
}

impl ViewState {
    /// Complex-plane span (width, height) for the given pixel aspect ratio.
    pub fn span(&self, aspect: f64) -> (Scale, Scale) {
        let h = self.half_height.mul_f64(2.0);
        (h.mul_f64(aspect), h)
    }

    /// Complex-plane units per pixel, given the viewport height in pixels.
    pub fn complex_per_pixel(&self, height_px: f64) -> Scale {
        self.half_height.mul_f64(2.0 / height_px)
    }

    /// log10 of the current magnification relative to the default view.
    pub fn magnification_log10(&self) -> f64 {
        DEFAULT_HALF_HEIGHT.log10() - self.half_height.log10()
    }

    /// Current zoom level.
    pub fn zoom(&self) -> Scale {
        self.half_height
    }

    /// Bits of precision the center currently needs for this zoom level.
    pub fn precision_bits(&self) -> usize {
        precision_for(self.half_height)
    }

    /// Ensure the center carries enough precision for the current zoom. Must be
    /// called before mutating the center so arithmetic keeps the needed digits.
    pub fn sync_precision(&mut self) {
        let bits = self.precision_bits();
        if self.center_re.precision() < bits {
            self.center_re = self.center_re.clone().with_precision(bits).value();
        }
        if self.center_im.precision() < bits {
            self.center_im = self.center_im.clone().with_precision(bits).value();
        }
    }

    /// Pan by a pixel delta (screen space: +x right, +y down).
    pub fn pan_pixels(&mut self, dx: f64, dy: f64, height_px: f64) {
        self.sync_precision();
        let cpp = self.complex_per_pixel(height_px);
        let bits = self.precision_bits();
        // Grab-and-drag: moving the mouse right shows content to the left.
        self.center_re = &self.center_re - &cpp.big_times(dx, bits);
        self.center_im = &self.center_im - &cpp.big_times(dy, bits);
    }

    /// Zoom by `factor` (<1 zooms in) keeping the complex point currently under
    /// the cursor fixed on screen. `off_*` is the cursor offset from the
    /// viewport center in pixels.
    pub fn zoom_at_pixel(&mut self, off_x: f64, off_y: f64, height_px: f64, factor: f64) {
        self.sync_precision();
        let cpp = self.complex_per_pixel(height_px);
        let bits = self.precision_bits();
        // The cursor's complex offset from the center is (off * cpp). Keeping it
        // fixed while scaling the view by `factor` moves the center by
        // off * cpp * (1 - factor). (Derivation: new_c = fixed + (c-fixed)*f.)
        let k = 1.0 - factor;
        self.center_re = &self.center_re + &cpp.big_times(off_x * k, bits);
        self.center_im = &self.center_im + &cpp.big_times(off_y * k, bits);
        self.half_height = self.half_height.mul_f64(factor);
    }

    /// Build a view from full-precision center coordinates and a half-height.
    pub fn with_center(center_re: Big, center_im: Big, half_height: Scale) -> Self {
        let mut v = Self {
            center_re,
            center_im,
            half_height,
        };
        v.sync_precision();
        v
    }
}

/// Parse a decimal string (any number of digits) losslessly into a `Big` with at
/// least `bits` of precision. Used for share links and debug view specs.
pub fn big_from_decimal_str(s: &str, bits: usize) -> Option<Big> {
    let dec = DBig::from_str(s.trim()).ok()?;
    Some(dec.with_base_and_precision::<2>(bits.max(53)).value())
}

/// Parse a "re,im,half_height[,iterations]" spec (re/im decimal, parsed at
/// full precision) into a view and an optional iteration count. Shared by
/// `FractalApp::apply_view_spec` (the `--view` CLI flag) and headless
/// animation's `--to-view`.
pub fn parse_view_spec(spec: &str) -> Option<(ViewState, Option<u32>)> {
    let parts: Vec<&str> = spec.split(',').collect();
    if parts.len() < 3 {
        return None;
    }
    let half_height = parse_half_height_spec(parts[2])?;
    let bits = precision_for(half_height);
    let re = big_from_decimal_str(parts[0], bits)?;
    let im = big_from_decimal_str(parts[1], bits)?;
    let iterations = parts.get(3).and_then(|s| s.trim().parse::<u32>().ok());
    Some((ViewState::with_center(re, im, half_height), iterations))
}

/// Parse a half_height spec. Shared by
/// `FractalApp::apply_half_height_spec` (the `--zoom` CLI flag) and headless
/// animation's `--to-zoom`.
pub fn parse_half_height_spec(spec: &str) -> Option<Scale> {
    spec.parse::<Scale>().ok()
}
/// Parse a "re,im" spec (re/im decimal, parsed at
/// full precision) into a view. Shared by
/// `FractalApp::apply_re_im_spec` (the `--position` CLI flag) and headless
/// animation's `--to-position`.
pub fn parse_re_im_spec(spec: &str, bits: usize) -> Option<(Big, Big)> {
    let parts: Vec<&str> = spec.split(',').collect();
    if parts.len() != 2 {
        return None;
    }
    let re = big_from_decimal_str(parts[0], bits)?;
    let im = big_from_decimal_str(parts[1], bits)?;
    Some((re, im))
}

/// Interpolate between two views for an animation frame, `t` in `[0, 1]`.
/// The half-height interpolates geometrically (log-linear), since zoom depth
/// spans many decades and a linear sweep would crawl at the start and blow
/// past the target at the end. The center has to shrink its offset from the
/// target at that *same* geometric rate: blending it linearly in `t` instead
/// barely moves it while the view is still huge (early frames), so the
/// target stays effectively off-screen — offset/half_height ratio blows up —
/// for nearly the whole animation, and only lands on `to`'s center in the
/// literal last frame where `t == 1` forces an exact match. `g(t)` below
/// tracks the same `q^t` decay used for `half_height` (keeping the
/// offset/half_height ratio roughly constant, i.e. the target's on-screen
/// position steady) but is shifted so it lands on exactly 1 at `t = 0` and
/// exactly 0 at `t = 1`.
///
/// With `d = log2(q)`, `g = q^t · (1 - q^(1-t)) / (1 - q)`, all in `Scale`
/// / `expm1` form: zooming in by more than f64's range, `q` (and `q^t`)
/// underflow, yet `g · (from - to)` must keep tracking the half-height.
pub fn interpolate_view(from: &ViewState, to: &ViewState, t: f64) -> ViewState {
    let (l0, l1) = (from.half_height.log2(), to.half_height.log2());
    let d = l1 - l0;
    let half_height = if t <= 0.0 {
        from.half_height
    } else if t >= 1.0 {
        to.half_height
    } else {
        Scale::from_log2(l0 + t * d)
    };
    let bits = precision_for(half_height);
    let ln2 = core::f64::consts::LN_2;
    let g_big = if d.abs() < 1e-12 {
        big_from_f64(1.0 - t, bits)
    } else if d < 0.0 {
        // Zooming in: q^t may be far below f64's range, keep it as a Scale.
        let f = ((1.0 - t) * d * ln2).exp_m1() / (d * ln2).exp_m1();
        Scale::from_log2(t * d).big_times(f, bits)
    } else {
        // Zooming out: g = (1 - q^(t-1)) / (1 - q^-1), every term bounded.
        big_from_f64(((t - 1.0) * d * ln2).exp_m1() / (-d * ln2).exp_m1(), bits)
    };
    let re0 = from.center_re.clone().with_precision(bits).value();
    let im0 = from.center_im.clone().with_precision(bits).value();
    let re1 = to.center_re.clone().with_precision(bits).value();
    let im1 = to.center_im.clone().with_precision(bits).value();
    let center_re = &re1 + &(&(&re0 - &re1) * &g_big);
    let center_im = &im1 + &(&(&im0 - &im1) * &g_big);
    ViewState::with_center(center_re, center_im, half_height)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn interpolate_f64(from: f64, to: f64, t: f64) -> f64 {
    from + (to - from) * t
}

/// Render a `Big` as a decimal string with `sig_digits` significant digits.
pub fn big_to_decimal_str(x: &Big, sig_digits: usize) -> String {
    let dec = x
        .to_decimal()
        .value()
        .with_precision(sig_digits.max(1))
        .value();
    format!("{dec}")
}

/// Precision (bits) needed to resolve the center at a given half-height.
pub fn precision_for(half_height: Scale) -> usize {
    // We need enough bits to distinguish points a pixel apart, i.e. roughly
    // log2(1 / half_height) significant bits, plus a guard margin.
    let zoom_bits = (-half_height.log2()).ceil().max(0.0) as usize;
    (zoom_bits + GUARD_BITS).clamp(53, MAX_PRECISION_BITS)
}

/// Build an `FBig` from an f64 with an explicit precision context.
pub fn big_from_f64(x: f64, bits: usize) -> Big {
    Big::try_from(x)
        .unwrap_or_default()
        .with_precision(bits)
        .value()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sc(x: f64) -> Scale {
        Scale::from_f64(x)
    }

    fn re_im_f64(v: &ViewState) -> (f64, f64) {
        let re: f64 = v.center_re.to_decimal().value().to_f64().value();
        let im: f64 = v.center_im.to_decimal().value().to_f64().value();
        (re, im)
    }

    #[test]
    fn interpolate_view_hits_exact_endpoints() {
        let bits = precision_for(sc(1.0));
        let from =
            ViewState::with_center(big_from_f64(-0.5, bits), big_from_f64(0.0, bits), sc(1.5));
        let to = ViewState::with_center(
            big_from_f64(-0.7515, precision_for(sc(1e-20))),
            big_from_f64(0.1013, precision_for(sc(1e-20))),
            sc(1e-20),
        );

        let start = interpolate_view(&from, &to, 0.0);
        assert_eq!(re_im_f64(&start), re_im_f64(&from));
        assert_eq!(start.half_height, from.half_height);

        let end = interpolate_view(&from, &to, 1.0);
        assert_eq!(re_im_f64(&end), re_im_f64(&to));
        assert_eq!(end.half_height, to.half_height);
    }

    /// Regression test: a deep zoom's center used to be blended linearly in
    /// `t` while `half_height` shrank geometrically, so partway through the
    /// animation the offset from the target would already be far larger than
    /// the (tiny, geometrically-shrunk) view — the target only snapped into
    /// frame on the very last frame. The offset/half_height ratio should
    /// instead stay roughly bounded throughout.
    #[test]
    fn interpolate_view_keeps_target_offset_bounded() {
        let bits = precision_for(sc(1.0));
        let from =
            ViewState::with_center(big_from_f64(-0.5, bits), big_from_f64(0.0, bits), sc(1.5));
        let to = ViewState::with_center(
            big_from_f64(-0.7515, precision_for(sc(1e-20))),
            big_from_f64(0.1013, precision_for(sc(1e-20))),
            sc(1e-20),
        );
        let (to_re, to_im) = re_im_f64(&to);

        for i in 1..10 {
            let t = i as f64 / 10.0;
            let mid = interpolate_view(&from, &to, t);
            let (re, im) = re_im_f64(&mid);
            let offset = ((re - to_re).powi(2) + (im - to_im).powi(2)).sqrt();
            let ratio = offset / mid.half_height.to_f64();
            assert!(
                ratio < 10.0,
                "t={t}: offset/half_height ratio {ratio} blew up (offset={offset}, half_height={})",
                mid.half_height
            );
        }
    }

    #[test]
    fn scale_parse_display_round_trip() {
        for s in [
            "1.25",
            "1e-20",
            "1.5e-20",
            "3.7e-4000",
            "1e-400",
            "9.99999e-310",
        ] {
            let a: Scale = s.parse().unwrap();
            let b: Scale = a.to_string().parse().unwrap();
            assert_eq!(a, b, "{s} -> {a}");
        }
        assert_eq!("1.25".parse::<Scale>().unwrap().to_f64(), 1.25);
        assert_eq!(sc(1.5e-20).to_string(), "1.5e-20");
        let deep: Scale = "3.7e-4000".parse().unwrap();
        assert_eq!(deep.to_string(), "3.7e-4000");
        assert_eq!(format!("{deep:.2}"), "3.70e-4000");
        assert!((deep.log10() - (3.7f64.log10() - 4000.0)).abs() < 1e-9);
        assert!("0".parse::<Scale>().is_err());
        assert!("-1e-500".parse::<Scale>().is_err());
        assert!("abc".parse::<Scale>().is_err());
    }

    #[test]
    fn scale_arithmetic() {
        let a: Scale = "1e-1000".parse().unwrap();
        let b = a.mul_f64(0.25);
        assert!((b.ratio(a) - 0.25).abs() < 1e-15);
        assert!(b < a && a > b);
        assert_eq!(a.mul_f64(3.0).mul_f64(1.0 / 3.0).exponent(), a.exponent());
        assert_eq!(sc(1.0).exponent(), 0);
        assert_eq!(sc(0.75).exponent(), -1);
        assert_eq!(sc(4.0).scaled_f64(-2), 1.0);
        assert_eq!(
            a.scaled_f64(-a.exponent()),
            a.mul_f64(1.0).scaled_f64(-a.exponent())
        );
        assert!((1.0..2.0).contains(&a.scaled_f64(-a.exponent())));
        assert_eq!(a.to_f64(), 0.0);
        assert_eq!(Scale::MIN.mul_f64(0.5), Scale::MIN);
        let p = precision_for(a);
        assert!((3322 + 48..=3323 + 48).contains(&p), "{p}");
    }

    /// Past f64's range, the center must still land on the target at the
    /// same geometric pace as the half-height.
    #[test]
    fn interpolate_view_past_f64_range() {
        let from = ViewState::with_center(big_from_f64(-0.5, 64), big_from_f64(0.0, 64), sc(1.5));
        let hh: Scale = "1e-1000".parse().unwrap();
        let bits = precision_for(hh);
        let to =
            ViewState::with_center(big_from_f64(-0.7515, bits), big_from_f64(0.1013, bits), hh);
        let end = interpolate_view(&from, &to, 1.0);
        assert_eq!(end.half_height, hh);
        assert_eq!(re_im_f64(&end), re_im_f64(&to));
        let mut prev = from.half_height;
        for i in 1..20 {
            let t = i as f64 / 20.0;
            let mid = interpolate_view(&from, &to, t);
            assert!(mid.half_height < prev);
            prev = mid.half_height;
            // Offset from the target, in units of the view's half-height.
            let k = -mid.half_height.exponent() as isize;
            let dre = ((&mid.center_re - &to.center_re) << k).to_f64().value();
            let dim = ((&mid.center_im - &to.center_im) << k).to_f64().value();
            let ratio = (dre * dre + dim * dim).sqrt() / mid.half_height.scaled_f64(k as i32);
            assert!(ratio > 0.01 && ratio < 10.0, "t={t}: ratio {ratio}");
        }
    }
}
