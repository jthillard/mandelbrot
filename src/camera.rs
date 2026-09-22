use std::f32::consts::PI;

use glam::Vec3;

#[derive(Default, Clone)]
pub struct Camera {
    pub position: glam::Vec3,

    pub yaw: f32,
    pub pitch: f32,

    /// Demi-hauteur du volume visible (remplace fov_y_radians)
    ortho_height: f32,
    aspect_ratio: f32,
    z_near: f32,
    z_far: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0., 0., -1.),
            yaw: 0. * PI / 180.,
            pitch: 0. * PI / 180.,
            ortho_height: 1.0,
            aspect_ratio: 1.,
            z_near: 0.1,
            z_far: 100.,
        }
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
    }

    /// Camera-local right vector: perpendicular to yaw, ignoring pitch (so
    /// strafing stays level regardless of where the camera is looking).
    pub fn right(&self) -> glam::Vec3 {
        glam::Mat3::from_rotation_y(-self.yaw) * glam::Vec3::X
    }

    /// Move the camera in its own local space: `forward`/`right` follow the
    /// (pitch-aware) view direction and its horizontal right vector, `up`
    /// moves along the fixed world Y axis.
    pub fn translate(&mut self, forward: f32, right: f32, up: f32) {
        self.position += self.direction() * forward + self.right() * right + Vec3::Y * up;
    }

    /// Adjust yaw/pitch by the given deltas (radians). Pitch is clamped just
    /// short of straight up/down to avoid the view flipping past the pole.
    pub fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        const PITCH_LIMIT: f32 = PI / 2.0 - 0.01;
        self.yaw += dyaw;
        self.pitch = (self.pitch + dpitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Scale the visible ortho volume by `factor` (<1 zooms in, >1 zooms
    /// out), clamped to a sane range.
    pub fn zoom(&mut self, factor: f32) {
        self.ortho_height = (self.ortho_height * factor).clamp(0.001, 1000.0);
    }

    pub fn orthographic(&self) -> glam::Mat4 {
        let view = glam::Mat4::from_translation(self.position)
            * glam::Mat4::from_rotation_y(-self.yaw)
            * glam::Mat4::from_rotation_x(-self.pitch);

        let half_height = self.ortho_height;
        let half_width = half_height * self.aspect_ratio;

        glam::camera::lh::proj::directx::orthographic(
            -half_width,
            half_width,
            -half_height,
            half_height,
            self.z_near,
            self.z_far,
        ) * view.inverse()
    }

    pub fn direction(&self) -> glam::Vec3 {
        let forward = glam::Mat3::from_rotation_y(-self.yaw)
            * glam::Mat3::from_rotation_x(-self.pitch)
            * glam::Vec3::Z;

        forward.normalize()
    }
}
