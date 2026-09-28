//! Arbitrary-precision binary float [`Big`] (one coordinate of a deep-zoom
//! center, or of a reference orbit point), over one of two backends:
//!
//! - native (default): `rug` (GMP/MPFR), the fastest, but C code that can't
//!   target `wasm32-unknown-unknown`;
//! - `wasm` feature: a pure-Rust library, required for the web build.
//!
//! Both expose the same inherent API plus the operators below, and follow
//! the same precision rule: an operation's result has the larger of its
//! operands' precisions (in bits), rounded to nearest. Shifts (`<<`/`>>` by
//! an `isize`) are exact multiplications by powers of two.

#[cfg(not(feature = "wasm"))]
mod rug_backend;
#[cfg(not(feature = "wasm"))]
pub use rug_backend::Big;

#[cfg(feature = "wasm")]
mod web_backend;
#[cfg(feature = "wasm")]
pub use web_backend::Big;
// Native `--features wasm` builds (to test the web backend) still link rug.
#[cfg(all(feature = "wasm", not(target_arch = "wasm32")))]
use rug as _;

#[cfg(all(target_arch = "wasm32", not(feature = "wasm")))]
compile_error!("the web build needs `--features wasm` (rug can't target wasm32)");

use core::ops::{Add, Mul, Neg, Shl, Shr, Sub};

/// Implements `$Trait` for every owned/borrowed combination of `Big`
/// operands, through the backend's by-reference `$imp`.
macro_rules! forward_binop {
    ($Trait:ident, $method:ident, $imp:ident) => {
        impl $Trait<&Big> for &Big {
            type Output = Big;
            fn $method(self, rhs: &Big) -> Big {
                self.$imp(rhs)
            }
        }
        impl $Trait<Big> for &Big {
            type Output = Big;
            fn $method(self, rhs: Big) -> Big {
                self.$imp(&rhs)
            }
        }
        impl $Trait<&Big> for Big {
            type Output = Big;
            fn $method(self, rhs: &Big) -> Big {
                (&self).$imp(rhs)
            }
        }
        impl $Trait<Big> for Big {
            type Output = Big;
            fn $method(self, rhs: Big) -> Big {
                (&self).$imp(&rhs)
            }
        }
    };
}

forward_binop!(Add, add, add_ref);
forward_binop!(Sub, sub, sub_ref);
forward_binop!(Mul, mul, mul_ref);

impl Neg for Big {
    type Output = Big;
    fn neg(self) -> Big {
        self.negated()
    }
}

impl Neg for &Big {
    type Output = Big;
    fn neg(self) -> Big {
        self.clone().negated()
    }
}

impl Shl<isize> for Big {
    type Output = Big;
    fn shl(self, k: isize) -> Big {
        self.mul_pow2(k)
    }
}

impl Shl<isize> for &Big {
    type Output = Big;
    fn shl(self, k: isize) -> Big {
        self.clone().mul_pow2(k)
    }
}

impl Shr<isize> for Big {
    type Output = Big;
    fn shr(self, k: isize) -> Big {
        self.mul_pow2(-k)
    }
}

impl Shr<isize> for &Big {
    type Output = Big;
    fn shr(self, k: isize) -> Big {
        self.clone().mul_pow2(-k)
    }
}

/// Significant decimal digits of a value, as a backend's
/// `to_decimal_parts` returns them: the value is `±0.digits × 10^exp10`,
/// `digits` has no trailing zeros, and it is empty for zero.
pub struct DecimalParts {
    pub negative: bool,
    pub digits: String,
    pub exp10: isize,
}

impl DecimalParts {
    /// Build from a significand digit string (maybe with trailing zeros)
    /// whose value is `0.digits × 10^exp10`.
    fn new(negative: bool, digits: &str, exp10: isize) -> Self {
        let digits = digits.trim_end_matches('0').to_string();
        Self {
            negative: negative && !digits.is_empty(),
            digits,
            exp10,
        }
    }
}

impl Big {
    /// Decimal string with `sig_digits` significant digits, in plain
    /// positional notation (`-0.000123`, `4500`), trailing zeros trimmed.
    pub fn to_decimal_string(&self, sig_digits: usize) -> String {
        let DecimalParts {
            negative,
            digits,
            exp10,
        } = self.to_decimal_parts(sig_digits.max(1));
        if digits.is_empty() {
            return "0".to_string();
        }
        let sign = if negative { "-" } else { "" };
        let len = digits.len() as isize;
        if exp10 <= 0 {
            let zeros = "0".repeat((-exp10) as usize);
            format!("{sign}0.{zeros}{digits}")
        } else if exp10 >= len {
            let zeros = "0".repeat((exp10 - len) as usize);
            format!("{sign}{digits}{zeros}")
        } else {
            let (int, frac) = digits.split_at(exp10 as usize);
            format!("{sign}{int}.{frac}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(x: f64) -> Big {
        Big::from_f64(x, 200)
    }

    #[test]
    fn decimal_string_is_positional() {
        assert_eq!(
            big(-0.7515).to_decimal_string(20),
            "-0.75149999999999994582"
        );
        assert_eq!(big(-0.7515).to_decimal_string(5), "-0.7515");
        assert_eq!(big(0.0).to_decimal_string(5), "0");
        assert_eq!(big(1.0).to_decimal_string(5), "1");
        assert_eq!(big(123.456).to_decimal_string(5), "123.46");
        assert_eq!(big(1e20).to_decimal_string(5), "100000000000000000000");
        assert_eq!(big(0.1).to_decimal_string(5), "0.1");
        let tiny = big(1.0) >> 200;
        let s = tiny.to_decimal_string(5);
        assert_eq!(s, format!("0.{}6223", "0".repeat(60)));
    }

    #[test]
    fn decimal_round_trip() {
        let s = "-0.743643887037158704752191506114774";
        let x = Big::from_decimal_str(s, 200).unwrap();
        assert_eq!(x.to_decimal_string(33), s);
        assert_eq!(
            Big::from_decimal_str("1.5e-20", 64).unwrap().to_f64(),
            1.5e-20
        );
        assert!(Big::from_decimal_str("abc", 64).is_none());
    }

    #[test]
    fn precision_is_max_of_operands() {
        let a = Big::from_f64(1.0, 100);
        let b = Big::from_f64(3.0, 200);
        assert_eq!((&a + &b).precision(), 200);
        assert_eq!((&a * &b).precision(), 200);
        assert_eq!((&b - &a).precision(), 200);
        assert_eq!(a.clone().with_precision(300).precision(), 300);
    }

    #[test]
    fn exact_shifts_and_log2() {
        let x = Big::from_f64(0.75, 64) >> 5000;
        assert_eq!(x.log2_floor(), Some(-5001));
        assert_eq!((x << 5000).to_f64(), 0.75);
        assert_eq!(Big::zero(64).log2_floor(), None);
        assert!(Big::from_f64(-1.0, 64).is_negative());
        assert!(!Big::zero(64).is_negative());
    }

    #[test]
    fn transcendentals() {
        let x = Big::from_f64(0.5, 128);
        assert!((x.ln().to_f64() - 0.5f64.ln()).abs() < 1e-15);
        assert!((x.exp().to_f64() - 0.5f64.exp()).abs() < 1e-15);
        let (s, c) = x.sin_cos();
        assert!((s.to_f64() - 0.5f64.sin()).abs() < 1e-15);
        assert!((c.to_f64() - 0.5f64.cos()).abs() < 1e-15);
        let y = Big::from_f64(-0.3, 128);
        assert!((y.atan2(&x).to_f64() - (-0.3f64).atan2(0.5)).abs() < 1e-15);
    }
}
