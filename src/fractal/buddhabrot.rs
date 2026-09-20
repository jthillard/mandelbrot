//! Buddhabrot / Nebulabrot rendering: a Monte-Carlo orbit-density histogram,
//! accumulated progressively across frames by a compute pass and tone-mapped
//! to colour by a fragment pass. See `shaders/buddhabrot.wgsl` for the "why"
//! this is a separate pipeline from the escape-time perturbation renderer.

use eframe::egui_wgpu::{self, wgpu};

/// Random samples dispatched per accumulating frame. Chosen so a frame stays
/// interactive on a modest GPU even when most samples run the full `b_cap`
/// (e.g. the view sits entirely inside the set, so nothing escapes).
const SAMPLES_PER_DISPATCH: u32 = 150_000;
const WORKGROUP_SIZE: u32 = 64;

/// GPU-side parameters for both the accumulate (compute) and tonemap
/// (fragment) passes. Layout must match `Uniforms` in `buddhabrot.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BuddhabrotUniforms {
    pub center: [f32; 2],
    pub half_height: f32,
    pub aspect: f32,
    pub phoenix_p: [f32; 2],
    pub lambda_l: [f32; 2],
    pub bailout_sq: f32,
    /// Iteration formula (`FractalKind::shader_id`); `KIND_LAMBDA` samples z0
    /// instead of c (see the shader's doc comment).
    pub kind: u32,
    /// Exponent for the Multibrot kind.
    pub power: u32,
    /// Nested escape-iteration caps (r_cap <= g_cap <= b_cap) that bucket an
    /// orbit's points into the R/G/B histogram planes.
    pub r_cap: u32,
    pub g_cap: u32,
    pub b_cap: u32,
    /// RNG nonce, bumped every dispatch so each frame samples fresh points.
    pub seed: u32,
    pub samples_this_dispatch: u32,
    /// Tonemap brightness multiplier (user-controlled).
    pub exposure: f32,
    pub width: u32,
    pub height: u32,
    /// Running total of samples accumulated into the current histogram
    /// (across all dispatches since the last reset); normalizes brightness.
    pub total_samples: f32,
    /// Tonemap colour style: 0 = classic (R/G/B = raw caps), 1 = nebula
    /// (yellow core, blue halo), 2 = grayscale. Display-only, like `exposure`
    /// — excluded from `ContentKey` so changing it doesn't reset accumulation.
    pub palette: u32,
    /// Padding so `complex_power` (a vec2, 8-byte aligned in the shader)
    /// starts on an 8-byte boundary.
    pub _pad0: u32,
    /// Complex exponent for the Complex Multibrot kind; ignored by other kinds.
    pub complex_power: [f32; 2],
}

/// The subset of `BuddhabrotUniforms` that determines the *content* of the
/// histogram (as opposed to `exposure`, a display-only rescale). A change in
/// any of these invalidates the accumulated histogram.
#[derive(Copy, Clone, PartialEq)]
struct ContentKey {
    center: [f32; 2],
    half_height: f32,
    aspect: f32,
    phoenix_p: [f32; 2],
    lambda_l: [f32; 2],
    bailout_sq: f32,
    kind: u32,
    power: u32,
    complex_power: [f32; 2],
    r_cap: u32,
    g_cap: u32,
    b_cap: u32,
}

impl From<&BuddhabrotUniforms> for ContentKey {
    fn from(u: &BuddhabrotUniforms) -> Self {
        Self {
            center: u.center,
            half_height: u.half_height,
            aspect: u.aspect,
            phoenix_p: u.phoenix_p,
            lambda_l: u.lambda_l,
            bailout_sq: u.bailout_sq,
            kind: u.kind,
            power: u.power,
            complex_power: u.complex_power,
            r_cap: u.r_cap,
            g_cap: u.g_cap,
            b_cap: u.b_cap,
        }
    }
}

/// The histogram buffer and its two bind groups, sized to the widget.
struct Histogram {
    buffer: wgpu::Buffer,
    compute_bind_group: wgpu::BindGroup,
    tonemap_bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

pub struct BuddhabrotRenderer {
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    tonemap_pipeline: wgpu::RenderPipeline,
    tonemap_bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    histogram: Option<Histogram>,
    /// What the current histogram's content was last accumulated for; a
    /// mismatch clears the histogram and restarts accumulation.
    last_content: Option<ContentKey>,
    /// Running sample count since the last reset (mirrors what was written
    /// into `total_samples`, since the callback doesn't own that state).
    total_samples: f32,
    seed: u32,
}

impl BuddhabrotRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("buddhabrot"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/buddhabrot.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buddhabrot uniforms"),
            size: std::mem::size_of::<BuddhabrotUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("buddhabrot compute bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("buddhabrot compute pipeline layout"),
                bind_group_layouts: &[Some(&compute_bind_group_layout)],
                immediate_size: 0,
            });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("buddhabrot compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let tonemap_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("buddhabrot tonemap bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let tonemap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("buddhabrot tonemap pipeline layout"),
                bind_group_layouts: &[Some(&tonemap_bind_group_layout)],
                immediate_size: 0,
            });
        let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("buddhabrot tonemap pipeline"),
            layout: Some(&tonemap_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_tonemap"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            compute_pipeline,
            compute_bind_group_layout,
            tonemap_pipeline,
            tonemap_bind_group_layout,
            uniform_buffer,
            histogram: None,
            last_content: None,
            total_samples: 0.0,
            seed: 0,
        }
    }

    /// Ensure the histogram buffer exists at `width`×`height`, recreating (and
    /// resetting accumulation) on a size change.
    fn ensure_histogram(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if let Some(h) = &self.histogram
            && h.width == width
            && h.height == height
        {
            return;
        }

        let plane = (width as u64) * (height as u64);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("buddhabrot histogram"),
            size: plane * 3 * std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("buddhabrot compute bind group"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });
        let tonemap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("buddhabrot tonemap bind group"),
            layout: &self.tonemap_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });

        self.histogram = Some(Histogram {
            buffer,
            compute_bind_group,
            tonemap_bind_group,
            width,
            height,
        });
        // New (zero-initialized) buffer: accumulation starts fresh.
        self.last_content = None;
        self.total_samples = 0.0;
    }
}

/// Per-frame paint callback. `accumulate` controls whether a new batch of
/// samples is dispatched this frame (a content change always forces one
/// dispatch regardless, so a parameter/view change is never left blank).
pub struct BuddhabrotCallback {
    pub uniforms: BuddhabrotUniforms,
    pub accumulate: bool,
    /// Widget size in physical pixels — the histogram resolution.
    pub size_px: [u32; 2],
}

impl egui_wgpu::CallbackTrait for BuddhabrotCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(renderer) = resources.get_mut::<BuddhabrotRenderer>() else {
            return Vec::new();
        };

        let width = self.size_px[0].max(1);
        let height = self.size_px[1].max(1);
        renderer.ensure_histogram(device, width, height);

        let content = ContentKey::from(&self.uniforms);
        let content_changed = renderer.last_content != Some(content);
        let should_dispatch = content_changed || self.accumulate;

        if let Some(histogram) = &renderer.histogram {
            if content_changed {
                egui_encoder.clear_buffer(&histogram.buffer, 0, None);
                renderer.total_samples = 0.0;
                renderer.last_content = Some(content);
            }

            let mut uniforms = self.uniforms;
            uniforms.width = width;
            uniforms.height = height;
            if should_dispatch {
                renderer.seed = renderer.seed.wrapping_add(1);
                renderer.total_samples += SAMPLES_PER_DISPATCH as f32;
                uniforms.seed = renderer.seed;
                uniforms.samples_this_dispatch = SAMPLES_PER_DISPATCH;
            } else {
                uniforms.samples_this_dispatch = 0;
            }
            uniforms.total_samples = renderer.total_samples;
            queue.write_buffer(&renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

            if should_dispatch {
                let mut pass = egui_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("buddhabrot accumulate pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&renderer.compute_pipeline);
                pass.set_bind_group(0, &histogram.compute_bind_group, &[]);
                let workgroups = SAMPLES_PER_DISPATCH.div_ceil(WORKGROUP_SIZE);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(renderer) = resources.get::<BuddhabrotRenderer>()
            && let Some(histogram) = &renderer.histogram
        {
            render_pass.set_pipeline(&renderer.tonemap_pipeline);
            render_pass.set_bind_group(0, &histogram.tonemap_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }
}
