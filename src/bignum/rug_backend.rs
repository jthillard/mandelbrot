//! [`Big`] over `rug::Float` (MPFR), for native builds.

use core::cmp::Ordering;

use rug::{Assign, Float};

use super::DecimalParts;

/// Arbitrary-precision binary float. See the module docs.
#[derive(Clone, Debug, PartialEq)]
pub struct Big(Float);

/// MPFR precision for `bits`, clamped to what it accepts.
fn prec(bits: usize) -> u32 {
    (bits as u32).clamp(rug::float::prec_min(), rug::float::prec_max())
}

impl Big {
    /// `x` at `bits` of precision (exact when `bits >= 53`). Non-finite `x`
    /// reads as 0.
    pub fn from_f64(x: f64, bits: usize) -> Self {
        let x = if x.is_finite() { x } else { 0.0 };
        Self(Float::with_val(prec(bits), x))
    }

    pub fn zero(bits: usize) -> Self {
        Self(Float::new(prec(bits)))
    }

    /// Parse a decimal (`-0.75`, `1.5e-20`, any number of digits) at `bits`
    /// (at least 53) of precision.
    pub fn from_decimal_str(s: &str, bits: usize) -> Option<Self> {
        let parsed = Float::parse(s.trim()).ok()?;
        let x = Float::with_val(prec(bits.max(53)), parsed);
        x.is_finite().then_some(Self(x))
    }

    pub fn precision(&self) -> usize {
        self.0.prec() as usize
    }

    /// Change the precision, rounding to nearest if it shrinks.
    pub fn with_precision(mut self, bits: usize) -> Self {
        self.0.set_prec(prec(bits));
        self
    }

    pub fn to_f64(&self) -> f64 {
        self.0.to_f64()
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    pub fn is_negative(&self) -> bool {
        self.0.cmp0() == Some(Ordering::Less)
    }

    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    pub fn sqr(&self) -> Self {
        Self(Float::with_val(self.0.prec(), self.0.square_ref()))
    }

    /// `floor(log2|x|)`, or `None` for zero. Exact, at any exponent.
    pub fn log2_floor(&self) -> Option<isize> {
        // MPFR normalizes the significand to [0.5, 1).
        self.0.get_exp().map(|e| e as isize - 1)
    }

    pub fn ln(&self) -> Self {
        Self(Float::with_val(self.0.prec(), self.0.ln_ref()))
    }

    pub fn exp(&self) -> Self {
        Self(Float::with_val(self.0.prec(), self.0.exp_ref()))
    }

    /// `atan2(self, x)`: the angle of `(x, self)`.
    pub fn atan2(&self, x: &Big) -> Self {
        let p = self.0.prec().max(x.0.prec());
        Self(Float::with_val(p, self.0.atan2_ref(&x.0)))
    }

    pub fn sin_cos(&self) -> (Self, Self) {
        let p = self.0.prec();
        let (mut s, mut c) = (Float::new(p), Float::new(p));
        (&mut s, &mut c).assign(self.0.sin_cos_ref());
        (Self(s), Self(c))
    }

    /// `sig` significant decimal digits (rounded to nearest).
    pub fn to_decimal_parts(&self, sig: usize) -> DecimalParts {
        if self.0.is_zero() {
            return DecimalParts::new(false, "", 0);
        }
        let (negative, digits, exp) = self.0.to_sign_string_exp(10, Some(sig));
        DecimalParts::new(negative, &digits, exp.unwrap_or(0) as isize)
    }

    pub(super) fn add_ref(&self, rhs: &Big) -> Big {
        let p = self.0.prec().max(rhs.0.prec());
        Big(Float::with_val(p, &self.0 + &rhs.0))
    }

    pub(super) fn sub_ref(&self, rhs: &Big) -> Big {
        let p = self.0.prec().max(rhs.0.prec());
        Big(Float::with_val(p, &self.0 - &rhs.0))
    }

    pub(super) fn mul_ref(&self, rhs: &Big) -> Big {
        let p = self.0.prec().max(rhs.0.prec());
        Big(Float::with_val(p, &self.0 * &rhs.0))
    }

    pub(super) fn negated(self) -> Big {
        Big(-self.0)
    }

    /// `self · 2^k`, exact. `|k|` stays far below `i32::MAX` (precision and
    /// zoom depth are capped around 2^20 bits).
    pub(super) fn mul_pow2(self, k: isize) -> Big {
        Big(self.0 << k as i32)
    }
}
