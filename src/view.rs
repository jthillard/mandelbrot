//! Camera / view state over the complex plane.
//!
//! The center is stored in arbitrary precision (`FBig`) — this is what lets us
//! zoom far past f64's ~1e13x limit. The pixel *scale* stays `f64`, which
//! bounds zoom at ~10^300x (`MIN_HALF_HEIGHT`). Only the center needs the
//! extra digits. Once a pixel is smaller than `DEEP_PIXEL_SIZE` the GPU
//! switches to rescaled deltas (see `needs_deep`), since f32 alone bottoms
//! out near 1e-38.

use core::str::FromStr;

use dashu_float::round::mode::HalfAway;
use dashu_float::{DBig, FBig};

/// Arbitrary-precision binary float (base 2, round-half-away). One coordinate.
pub type Big = FBig<HalfAway, 2>;

/// Half-height (complex units) of the default view; also the zoom-1 reference.
pub const DEFAULT_HALF_HEIGHT: f64 = 1.25;

/// Smallest half-height the view can zoom to: f64's range (the pixel scale,
/// and the rescaled GPU uniforms, are computed in f64).
pub const MIN_HALF_HEIGHT: f64 = 1e-300;

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
pub fn needs_deep(half_height: f64, height_px: f64) -> bool {
    2.0 * half_height / height_px.max(1.0) < DEEP_PIXEL_SIZE
}

/// Binary exponent `E` of the deep view scale: `floor(log2(half_height))`,
/// so the rescaled span is in `[2, 4)`. Never 0, which means "not deep"
/// (see `Uniforms::scale_exp`).
pub fn deep_scale_exp(half_height: f64) -> i32 {
    let e = half_height.max(MIN_HALF_HEIGHT).log2().floor() as i32;
    if e == 0 { -1 } else { e }
}

/// Guard bits added on top of the zoom-dictated precision.
const GUARD_BITS: usize = 48;
/// Upper bound on center precision (f32 GPU perturbation degrades long before
/// this; the cap just prevents pathological allocation).
const MAX_PRECISION_BITS: usize = 2048;

#[derive(Clone, Debug)]
pub struct ViewState {
    pub center_re: Big,
    pub center_im: Big,
    /// Half the view height in complex-plane units. Zooming in shrinks this.
    pub half_height: f64,
}

impl Default for ViewState {
    fn default() -> Self {
        let bits = precision_for(DEFAULT_HALF_HEIGHT);
        Self {
            center_re: big_from_f64(-0.5, bits),
            center_im: big_from_f64(0.0, bits),
            half_height: DEFAULT_HALF_HEIGHT,
        }
    }
}

impl ViewState {
    /// Complex-plane span (width, height) for the given pixel aspect ratio.
    pub fn span(&self, aspect: f64) -> (f64, f64) {
        let h = self.half_height * 2.0;
        (h * aspect, h)
    }

    /// Complex-plane units per pixel, given the viewport height in pixels.
    pub fn complex_per_pixel(&self, height_px: f64) -> f64 {
        (self.half_height * 2.0) / height_px
    }

    /// Current magnification relative to the default view.
    pub fn magnification(&self) -> f64 {
        DEFAULT_HALF_HEIGHT / self.half_height
    }

    /// Current zoom level.
    pub fn zoom(&self) -> f64 {
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
        self.center_re = &self.center_re - &big_from_f64(dx * cpp, bits);
        self.center_im = &self.center_im - &big_from_f64(dy * cpp, bits);
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
        let k = cpp * (1.0 - factor);
        self.center_re = &self.center_re + &big_from_f64(off_x * k, bits);
        self.center_im = &self.center_im + &big_from_f64(off_y * k, bits);
        self.half_height = (self.half_height * factor).max(MIN_HALF_HEIGHT);
    }

    /// Build a view from full-precision center coordinates and a half-height.
    pub fn with_center(center_re: Big, center_im: Big, half_height: f64) -> Self {
        let mut v = Self {
            center_re,
            center_im,
            half_height: half_height.max(MIN_HALF_HEIGHT),
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
    let half_height = parts[2].trim().parse::<f64>().ok()?;
    if !(half_height > 0.0 && half_height.is_finite()) {
        return None;
    }
    let half_height = half_height.max(MIN_HALF_HEIGHT);
    let bits = precision_for(half_height);
    let re = big_from_decimal_str(parts[0], bits)?;
    let im = big_from_decimal_str(parts[1], bits)?;
    let iterations = parts.get(3).and_then(|s| s.trim().parse::<u32>().ok());
    Some((ViewState::with_center(re, im, half_height), iterations))
}

/// Parse a half_height spec. Shared by
/// `FractalApp::apply_half_height_spec` (the `--zoom` CLI flag) and headless
/// animation's `--to-zoom`.
pub fn parse_half_height_spec(spec: &str) -> Option<f64> {
    let half_height = spec.trim().parse::<f64>().ok()?;
    if !(half_height > 0.0 && half_height.is_finite()) {
        return None;
    }
    Some(half_height.max(MIN_HALF_HEIGHT))
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
pub fn interpolate_view(from: &ViewState, to: &ViewState, t: f64) -> ViewState {
    let q = to.half_height / from.half_height;
    let half_height = from.half_height * q.powf(t);
    let bits = precision_for(half_height);
    let g = if (q - 1.0).abs() < 1e-12 {
        1.0 - t
    } else {
        (q.powf(t) - q) / (1.0 - q)
    };
    let g_big = big_from_f64(g, bits);
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
pub fn precision_for(half_height: f64) -> usize {
    // We need enough bits to distinguish points a pixel apart, i.e. roughly
    // log2(1 / half_height) significant bits, plus a guard margin.
    let zoom_bits = if half_height > 0.0 && half_height.is_finite() {
        (-half_height.log2()).ceil().max(0.0) as usize
    } else {
        0
    };
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

    fn re_im_f64(v: &ViewState) -> (f64, f64) {
        let re: f64 = v.center_re.to_decimal().value().to_f64().value();
        let im: f64 = v.center_im.to_decimal().value().to_f64().value();
        (re, im)
    }

    #[test]
    fn interpolate_view_hits_exact_endpoints() {
        let bits = precision_for(1.0);
        let from = ViewState::with_center(big_from_f64(-0.5, bits), big_from_f64(0.0, bits), 1.5);
        let to = ViewState::with_center(
            big_from_f64(-0.7515, precision_for(1e-20)),
            big_from_f64(0.1013, precision_for(1e-20)),
            1e-20,
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
        let bits = precision_for(1.0);
        let from = ViewState::with_center(big_from_f64(-0.5, bits), big_from_f64(0.0, bits), 1.5);
        let to = ViewState::with_center(
            big_from_f64(-0.7515, precision_for(1e-20)),
            big_from_f64(0.1013, precision_for(1e-20)),
            1e-20,
        );
        let (to_re, to_im) = re_im_f64(&to);

        for i in 1..10 {
            let t = i as f64 / 10.0;
            let mid = interpolate_view(&from, &to, t);
            let (re, im) = re_im_f64(&mid);
            let offset = ((re - to_re).powi(2) + (im - to_im).powi(2)).sqrt();
            let ratio = offset / mid.half_height;
            assert!(
                ratio < 10.0,
                "t={t}: offset/half_height ratio {ratio} blew up (offset={offset}, half_height={})",
                mid.half_height
            );
        }
    }
}
