use std::f32::consts::PI;

use bytemuck::{Pod, Zeroable};
use ecolor::Color32;
#[cfg(feature = "gui")]
use egui::Ui;

/// Maximum number of simultaneous lights.
pub const MAX_LIGHT_COUNT: usize = 16;

#[derive(Clone, Copy, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct Light {
    pub azimuth: f32,
    pub altitude: f32,
    pub color: Color32,
    pub _pad: u32,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            azimuth: PI / 4.,
            altitude: PI / 4.,
            color: Color32::WHITE,
            _pad: 0,
        }
    }
}

impl Light {
    #[cfg(feature = "gui")]
    pub fn widget(&mut self, ui: &mut Ui) -> bool {
        let formater = |v, _| format!("{}°", ((v * 180. / std::f64::consts::PI) as u32));
        let parser = |s: &str| {
            s.parse::<u32>()
                .ok()
                .map(|x| x as f64 * std::f64::consts::PI / 180.)
        };
        ui.horizontal(|ui| {
            let del = ui.button("-").clicked();
            ui.label("color:");
            ui.color_edit_button_srgba(&mut self.color);
            ui.label("θ:");
            ui.add(
                egui::DragValue::new(&mut self.azimuth)
                    .range(0.0..=PI * 2.)
                    .custom_formatter(formater)
                    .custom_parser(parser)
                    .speed(0.02),
            );
            ui.label("φ:");
            ui.add(
                egui::DragValue::new(&mut self.altitude)
                    .range(0.0..=PI / 2.)
                    .custom_formatter(formater)
                    .custom_parser(parser)
                    .speed(0.02),
            );

            del
        })
        .inner
    }
}

/// GPU-side light, matching WGSL `Light` in `iterate_uniforms.wgsl`: the unit
/// direction toward the light (precomputed from azimuth/altitude so the
/// shader does no per-pixel trig) plus the packed RGBA colour, whose alpha is
/// the intensity. 16 bytes, so `array<Light, 16>` has a uniform-legal stride.
#[derive(Clone, Copy, PartialEq, Zeroable, Pod, Default)]
#[repr(C)]
pub struct GpuLight {
    pub dir: [f32; 3],
    pub color: Color32,
}

/// The light buffer's contents: the UI lights with a non-zero colour (the
/// only ones that contribute, and the ones the filmic white point counts),
/// packed to the front, plus how many there are (`Uniforms::light_count`).
pub fn gpu_lights(lights: &[Light]) -> ([GpuLight; MAX_LIGHT_COUNT], u32) {
    let mut out = [GpuLight::default(); MAX_LIGHT_COUNT];
    let mut n = 0;
    for l in lights.iter().filter(|l| l.color != Color32::TRANSPARENT) {
        if n == MAX_LIGHT_COUNT {
            break;
        }
        let (sa, ca) = l.altitude.sin_cos();
        let (sz, cz) = l.azimuth.sin_cos();
        out[n] = GpuLight {
            dir: [cz * ca, sz * ca, sa],
            color: l.color,
        };
        n += 1;
    }
    (out, n as u32)
}
