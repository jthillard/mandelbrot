use std::f32::consts::{PI, TAU};

use glam::Vec3;

#[derive(Default, Clone)]
pub struct Camera {
    pub position: glam::Vec3,

    pub yaw: f32,
    pub pitch: f32,

    pub aspect_ratio: f32,
    z_near: f32,
    z_far: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0., 0., -1.),
            yaw: 0. * PI / 180.,
            pitch: -30. * PI / 180.,
            aspect_ratio: 1.,
            z_near: 0.1,
            z_far: 100.,
        }
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
    }

    /// Adjust yaw/pitch by the given deltas (radians). Pitch is clamped just
    /// short of straight up/down to avoid the view flipping past the pole.
    /// Yaw is wrapped to [-π, π) so the 2D <-> 3D transition (which scales
    /// yaw by `t`) always unwinds the short way instead of every past turn.
    pub fn rotate(&mut self, dyaw: f32, dpitch: f32) {
        const PITCH_LIMIT: f32 = PI / 2.0 - 0.01;
        self.yaw = (self.yaw + dyaw + PI).rem_euclid(TAU) - PI;
        self.pitch = (self.pitch + dpitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    pub fn orthographic(&self, t: f32) -> glam::Mat4 {
        let yaw = self.yaw * t;
        let pitch = self.pitch * t;

        let zoom = 0.5 * (1. + t);
        // Orbit pivot: the center of the fractal texture, which the raymarcher's
        // `sdf` lays out over world x ∈ [0, aspect], y ∈ [0, 1] on the z = 0 plane.
        let view = glam::Mat4::from_translation(Vec3::new(0.5 * self.aspect_ratio, 0.5, 0.))
            * glam::Mat4::from_rotation_z(-yaw)
            * glam::Mat4::from_rotation_x(-pitch)
            * glam::Mat4::from_translation(self.position);

        glam::camera::lh::proj::directx::orthographic(
            -self.aspect_ratio / 4. / zoom,
            self.aspect_ratio / 4. / zoom,
            -0.25 / zoom,
            0.25 / zoom,
            self.z_near,
            self.z_far,
        ) * view.inverse()
    }

    pub fn direction(&self, t: f32) -> glam::Vec3 {
        let yaw = self.yaw * t;
        let pitch = self.pitch * t;

        let forward =
            glam::Mat3::from_rotation_z(-yaw) * glam::Mat3::from_rotation_x(-pitch) * glam::Vec3::Z;

        forward.normalize()
    }
}
