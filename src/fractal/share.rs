//! Encode/decode a full view (fractal mode, high-precision center, zoom,
//! iterations, Julia constant, coloring) as a compact URL fragment so deep-zoom
//! locations can be shared or bookmarked.
//!
//! Format: `m=m&f=<str>&re=<dec>&im=<dec>&hh=<f64>&it=<u32>&cs=<f32>&co=<f32>` with
//! `m=j&jr=<f64>&ji=<f64>` added for Julia. `re`/`im` are full-precision decimal
//! strings.

use std::collections::HashMap;

use crate::fractal::FractalKind;

#[derive(Clone, Debug)]
pub struct ShareState {
    pub julia: bool,
    pub kind: FractalKind,
    /// Exponent for the Multibrot kind (ignored by others).
    pub power: u32,
    pub center_re: String,
    pub center_im: String,
    pub half_height: f64,
    pub iterations: u32,
    pub julia_c: (f64, f64),
    /// Distortion constant for the Phoenix kind (ignored by others).
    pub phoenix_p: (f64, f64),
    /// Distortion constant for the Lambda kind (ignored by others).
    pub lambda_l: (f64, f64),
    pub color_scale: f32,
    pub color_offset: f32,
    /// Palette index (`palette_id` in the shader).
    pub palette: u32,
}

impl ShareState {
    pub fn encode(&self) -> String {
        let mut s = String::new();
        s.push_str(if self.julia { "m=j" } else { "m=m" });
        s.push_str(&format!(
            "&f={}",
            match self.kind {
                FractalKind::Mandelbrot => "mandel",
                FractalKind::BurningShip => "burning",
                FractalKind::Multibrot => "multi",
                FractalKind::Tricorn => "tricorn",
                FractalKind::Celtic => "celtic",
                FractalKind::Perpendicular => "perp",
                FractalKind::Buffalo => "buffalo",
                FractalKind::Phoenix => "phoenix",
                FractalKind::Lambda => "lambda",
            }
        ));
        s.push_str(&format!("&pw={}", self.power));
        s.push_str(&format!(
            "&re={}&im={}&hh={}&it={}",
            self.center_re, self.center_im, self.half_height, self.iterations
        ));
        s.push_str(&format!("&jr={}&ji={}", self.julia_c.0, self.julia_c.1));
        s.push_str(&format!("&px={}&py={}", self.phoenix_p.0, self.phoenix_p.1));
        s.push_str(&format!("&lx={}&ly={}", self.lambda_l.0, self.lambda_l.1));
        s.push_str(&format!(
            "&cs={}&co={}&pal={}",
            self.color_scale, self.color_offset, self.palette
        ));
        s
    }

    pub fn decode(fragment: &str) -> Option<ShareState> {
        let fragment = fragment.trim_start_matches(['#', '?']);
        let mut map: HashMap<&str, &str> = HashMap::new();
        for kv in fragment.split('&') {
            if let Some((k, v)) = kv.split_once('=') {
                map.insert(k, v);
            }
        }

        Some(ShareState {
            julia: map.get("m").map(|m| *m == "j").unwrap_or(false),
            kind: map
                .get("f")
                .map(|f| match *f {
                    "mandel" => FractalKind::Mandelbrot,
                    "multi" => FractalKind::Multibrot,
                    "burning" => FractalKind::BurningShip,
                    "tricorn" => FractalKind::Tricorn,
                    "celtic" => FractalKind::Celtic,
                    "perp" => FractalKind::Perpendicular,
                    "buffalo" => FractalKind::Buffalo,
                    "phoenix" => FractalKind::Phoenix,
                    "lambda" => FractalKind::Lambda,
                    _ => FractalKind::Mandelbrot,
                })
                .unwrap_or(FractalKind::Mandelbrot),
            power: map.get("pw").and_then(|s| s.parse().ok()).unwrap_or(2),
            center_re: (*map.get("re")?).to_string(),
            center_im: (*map.get("im")?).to_string(),
            half_height: map.get("hh")?.parse().ok()?,
            iterations: map.get("it").and_then(|s| s.parse().ok()).unwrap_or(512),
            julia_c: (
                map.get("jr").and_then(|s| s.parse().ok()).unwrap_or(-0.8),
                map.get("ji").and_then(|s| s.parse().ok()).unwrap_or(0.156),
            ),
            phoenix_p: (
                map.get("px").and_then(|s| s.parse().ok()).unwrap_or(-0.5),
                map.get("py").and_then(|s| s.parse().ok()).unwrap_or(0.0),
            ),
            lambda_l: (
                map.get("lx").and_then(|s| s.parse().ok()).unwrap_or(-0.5),
                map.get("ly").and_then(|s| s.parse().ok()).unwrap_or(0.0),
            ),
            color_scale: map.get("cs").and_then(|s| s.parse().ok()).unwrap_or(0.02),
            color_offset: map.get("co").and_then(|s| s.parse().ok()).unwrap_or(0.0),
            palette: map.get("pal").and_then(|s| s.parse().ok()).unwrap_or(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let s = ShareState {
            julia: true,
            kind: FractalKind::Phoenix,
            power: 5,
            center_re: "-0.743643887037158704752191506114774".into(),
            center_im: "0.131825904205311970493132056385139".into(),
            half_height: 1.5e-20,
            iterations: 4000,
            julia_c: (-0.123, 0.745),
            phoenix_p: (-0.5, 0.1),
            lambda_l: (-0.5, 0.0),
            color_scale: 0.02,
            color_offset: 0.25,
            palette: 3,
        };
        let d = ShareState::decode(&s.encode()).unwrap();
        assert_eq!(d.julia, s.julia);
        assert_eq!(d.kind, s.kind);
        assert_eq!(d.power, s.power);
        assert_eq!(d.center_re, s.center_re);
        assert_eq!(d.center_im, s.center_im);
        assert_eq!(d.half_height, s.half_height);
        assert_eq!(d.iterations, s.iterations);
        assert_eq!(d.julia_c, s.julia_c);
        assert_eq!(d.phoenix_p, s.phoenix_p);
        assert_eq!(d.palette, s.palette);
    }

    #[test]
    fn decode_with_leading_hash() {
        let d = ShareState::decode("#m=m&re=0.0&im=0.0&hh=1.25&it=256").unwrap();
        assert!(!d.julia);
        assert_eq!(d.iterations, 256);
    }
}
