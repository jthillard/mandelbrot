//! [`Big`] over `malachite-float`, pure Rust, for the web build.

use malachite_base::num::basic::traits::Zero;
use malachite_base::num::conversion::string::options::ToSciOptions;
use malachite_base::num::conversion::traits::{RoundingFrom, ToSci};
use malachite_base::rounding_modes::RoundingMode::Nearest;
use malachite_float::Float;

use super::DecimalParts;

/// Arbitrary-precision binary float. See the module docs.
///
/// The precision is stored next to the value: malachite's zero has none,
/// but "zero at `p` bits" must still lift a sum with it to `p` bits.
#[derive(Clone, Debug)]
pub struct Big {
    f: Float,
    prec: u64,
}

impl PartialEq for Big {
    fn eq(&self, other: &Self) -> bool {
        self.f == other.f
    }
}

fn prec(bits: usize) -> u64 {
    (bits as u64).max(1)
}

impl Big {
    fn new(f: Float, prec: u64) -> Self {
        // Only finite values reach here; anything else (overflow) is a bug.
        debug_assert!(f.is_finite(), "non-finite Big: {f}");
        Self { f, prec }
    }

    /// `x` at `bits` of precision (exact when `bits >= 53`). Non-finite `x`
    /// reads as 0.
    pub fn from_f64(x: f64, bits: usize) -> Self {
        let p = prec(bits);
        if !x.is_finite() {
            return Self::zero(bits);
        }
        Self::new(Float::from_primitive_float_prec(x, p).0, p)
    }

    pub fn zero(bits: usize) -> Self {
        Self::new(Float::ZERO, prec(bits))
    }

    /// Parse a decimal (`-0.75`, `1.5e-20`, any number of digits) at `bits`
    /// (at least 53) of precision.
    pub fn from_decimal_str(s: &str, bits: usize) -> Option<Self> {
        let p = prec(bits.max(53));
        let (f, _) = Float::from_sci_string_prec(s.trim(), p)?;
        f.is_finite().then(|| Self::new(f, p))
    }

    pub fn precision(&self) -> usize {
        self.prec as usize
    }

    /// Change the precision, rounding to nearest if it shrinks.
    pub fn with_precision(mut self, bits: usize) -> Self {
        let p = prec(bits);
        if self.f.get_prec().is_some() {
            self.f.set_prec(p);
        }
        self.prec = p;
        self
    }

    pub fn to_f64(&self) -> f64 {
        f64::rounding_from(&self.f, Nearest).0
    }

    pub fn is_zero(&self) -> bool {
        self.f.is_zero()
    }

    pub fn is_negative(&self) -> bool {
        self.f.is_sign_negative() && !self.f.is_zero()
    }

    pub fn abs(self) -> Self {
        if self.f.is_sign_negative() {
            self.negated()
        } else {
            self
        }
    }

    pub fn sqr(&self) -> Self {
        Self::new(self.f.square_prec_ref(self.prec).0, self.prec)
    }

    /// `floor(log2|x|)`, or `None` for zero. Exact, at any exponent.
    pub fn log2_floor(&self) -> Option<isize> {
        // The significand is normalized to [0.5, 1).
        self.f.get_exponent().map(|e| e as isize - 1)
    }

    pub fn ln(&self) -> Self {
        Self::new(self.f.ln_prec_ref(self.prec).0, self.prec)
    }

    pub fn exp(&self) -> Self {
        Self::new(self.f.exp_prec_ref(self.prec).0, self.prec)
    }

    /// `atan2(self, x)`: the angle of `(x, self)`.
    pub fn atan2(&self, x: &Big) -> Self {
        let p = self.prec.max(x.prec);
        Self::new(self.f.atan2_prec_ref_ref(&x.f, p).0, p)
    }

    pub fn sin_cos(&self) -> (Self, Self) {
        let (s, c, _, _) = self.f.sin_cos_prec_ref(self.prec);
        (Self::new(s, self.prec), Self::new(c, self.prec))
    }

    /// `sig` significant decimal digits (rounded to nearest).
    pub fn to_decimal_parts(&self, sig: usize) -> DecimalParts {
        if self.f.is_zero() {
            return DecimalParts::new(false, "", 0);
        }
        let mut options = ToSciOptions::default();
        options.set_precision(sig as u64);
        let s = self.f.to_sci_with_options(options).to_string();
        // `[-]int[.frac][e±N]`, positional or scientific depending on size.
        let (negative, s) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s.as_str()),
        };
        let (mantissa, exp) = match s.split_once(['e', 'E']) {
            Some((m, e)) => (m, e.trim_start_matches('+').parse::<isize>().unwrap_or(0)),
            None => (s, 0),
        };
        let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        let all = format!("{int}{frac}");
        let digits = all.trim_start_matches('0');
        let leading_zeros = (all.len() - digits.len()) as isize;
        // 0.digits · 10^exp10 = int.frac · 10^exp.
        let exp10 = int.len() as isize - leading_zeros + exp;
        DecimalParts::new(negative, digits, exp10)
    }

    pub(super) fn add_ref(&self, rhs: &Big) -> Big {
        let p = self.prec.max(rhs.prec);
        Big::new(self.f.add_prec_ref_ref(&rhs.f, p).0, p)
    }

    pub(super) fn sub_ref(&self, rhs: &Big) -> Big {
        let p = self.prec.max(rhs.prec);
        Big::new(self.f.sub_prec_ref_ref(&rhs.f, p).0, p)
    }

    pub(super) fn mul_ref(&self, rhs: &Big) -> Big {
        let p = self.prec.max(rhs.prec);
        Big::new(self.f.mul_prec_ref_ref(&rhs.f, p).0, p)
    }

    pub(super) fn negated(self) -> Big {
        Big::new(-self.f, self.prec)
    }

    /// `self · 2^k`, exact (malachite's exponent range is ±2^30, far past
    /// what zoom depth needs).
    pub(super) fn mul_pow2(self, k: isize) -> Big {
        Big::new(self.f << k as i64, self.prec)
    }
}
