//! wgpu resources for the fractal: the render pipeline, the uniform buffer, the
//! reference-orbit storage buffer, and the egui paint callback that drives them.
//!
//! Rendering strategy: a single fullscreen triangle is drawn into the rectangle
//! egui allocates for the fractal widget (egui presets the render pass viewport
//! for us). The fragment shader iterates each pixel as an f32 perturbation delta
//! from the high-precision reference orbit stored in `ref_buffer`.

use std::sync::Arc;

use eframe::egui_wgpu::{self, wgpu};

/// Maximum reference-orbit length (points) the storage buffer can hold. Also
/// bounds the iteration count. 128k points * 8 bytes = 1 MiB.
pub const MAX_REF_POINTS: usize = 1 << 17;

/// GPU-side view + coloring parameters. Layout must match `Uniforms` in the
/// WGSL shader; total size is a multiple of 16 bytes for uniform-buffer rules.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    /// Complex-plane span (width, height) covered by the view. Per-pixel `dc`
    /// is `centered * span`, where `centered` is in [-0.5, 0.5].
    pub span: [f32; 2],
    pub max_iter: u32,
    pub ref_len: u32,
    pub color_offset: f32,
    pub color_scale: f32,
    pub bailout_sq: f32,
    /// 0 = Mandelbrot, 1 = Julia.
    pub is_julia: u32,
    pub palette_id: u32,
    pub _pad0: u32,
    /// Complex offset of the view center from the reference center, so a stale
    /// or reused reference (computed at a slightly different center) still maps
    /// correctly. Added to every pixel's per-pixel offset.
    pub dc_offset: [f32; 2],
}

pub struct FractalRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    ref_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    target_format: wgpu::TextureFormat,
    /// Generation of the reference orbit currently uploaded to `ref_buffer`.
    uploaded_generation: u64,
}

impl FractalRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mandelbrot"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mandelbrot.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fractal uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ref_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reference orbit"),
            size: (MAX_REF_POINTS * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fractal bind group layout"),
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
                    binding: 1,
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fractal bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ref_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fractal pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fractal pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
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
            pipeline,
            uniform_buffer,
            ref_buffer,
            bind_group,
            target_format,
            uploaded_generation: u64::MAX,
        }
    }

    /// Upload a reference orbit to the storage buffer (used by PNG export to
    /// guarantee the buffer is current before an offscreen render).
    pub fn upload_reference(&self, queue: &wgpu::Queue, points: &[[f32; 2]]) {
        let count = points.len().min(MAX_REF_POINTS);
        if count > 0 {
            queue.write_buffer(&self.ref_buffer, 0, bytemuck::cast_slice(&points[..count]));
        }
    }

    /// True if the render target stores bytes as BGRA (so a PNG needs R/B
    /// swapped). Surfaces are usually `Bgra8UnormSrgb`.
    pub fn needs_rb_swap(&self) -> bool {
        matches!(
            self.target_format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        )
    }

    /// Render the current fractal (using `uniforms` and the already-uploaded
    /// reference orbit) into an offscreen texture at `width`x`height`, then copy
    /// it into a mappable buffer. Returns the buffer and its padded row stride.
    /// The caller maps the buffer (blocking on native, async on web).
    pub fn render_to_readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        uniforms: Uniforms,
    ) -> (wgpu::Buffer, u32) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("export target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded_bpr = width * 4;
        let padded_bpr = unpadded_bpr.div_ceil(align) * align;

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export readback"),
            size: (padded_bpr * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("export"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("export pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bpr),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(std::iter::once(encoder.finish()));
        (readback, padded_bpr)
    }
}

/// Convert a padded BGRA/RGBA readback into tightly-packed RGBA8 and encode it
/// as PNG bytes.
pub fn encode_png(
    padded: &[u8],
    width: u32,
    height: u32,
    padded_bpr: u32,
    swap_rb: bool,
) -> Vec<u8> {
    let row = (width * 4) as usize;
    let mut rgba = vec![0u8; row * height as usize];
    for y in 0..height as usize {
        let src_off = y * padded_bpr as usize;
        let src = &padded[src_off..src_off + row];
        let dst = &mut rgba[y * row..y * row + row];
        if swap_rb {
            for x in 0..width as usize {
                dst[x * 4] = src[x * 4 + 2];
                dst[x * 4 + 1] = src[x * 4 + 1];
                dst[x * 4 + 2] = src[x * 4];
                dst[x * 4 + 3] = src[x * 4 + 3];
            }
        } else {
            dst.copy_from_slice(src);
        }
    }

    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        writer.write_image_data(&rgba).expect("png data");
    }
    out
}

/// A per-frame paint callback. Carries this frame's uniforms plus a reference to
/// the current reference orbit (cheap `Arc` clone). The orbit is only re-uploaded
/// to the GPU when its `generation` changes.
pub struct FractalCallback {
    pub uniforms: Uniforms,
    pub reference: Arc<Vec<[f32; 2]>>,
    pub generation: u64,
}

impl egui_wgpu::CallbackTrait for FractalCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(renderer) = resources.get_mut::<FractalRenderer>() {
            queue.write_buffer(
                &renderer.uniform_buffer,
                0,
                bytemuck::bytes_of(&self.uniforms),
            );

            if renderer.uploaded_generation != self.generation && !self.reference.is_empty() {
                let count = self.reference.len().min(MAX_REF_POINTS);
                queue.write_buffer(
                    &renderer.ref_buffer,
                    0,
                    bytemuck::cast_slice(&self.reference[..count]),
                );
                renderer.uploaded_generation = self.generation;
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
        if let Some(renderer) = resources.get::<FractalRenderer>() {
            render_pass.set_pipeline(&renderer.pipeline);
            render_pass.set_bind_group(0, &renderer.bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }
}
