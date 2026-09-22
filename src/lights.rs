use std::f32::consts::PI;

use bytemuck::{Pod, Zeroable};
use egui::{Color32, Ui};

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
