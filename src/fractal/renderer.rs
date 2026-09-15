//! wgpu resources for the fractal: the render pipeline, the uniform buffer, the
//! reference-orbit storage buffer, and the egui paint callback that drives them.
//!
//! Rendering strategy: the expensive per-pixel perturbation shader renders into
//! an offscreen **cache texture**, and only when the view/coloring/size actually
//! change (tracked by `rendered`). Every egui frame then just blits that cached
//! texture onto egui's surface with a cheap textured fullscreen triangle — so
//! incidental repaints (mouse-move, hover, the worker-pending poll) cost a blit,
//! not a full fractal recompute. The fragment shader iterates each pixel as an
//! f32 perturbation delta from the reference orbit stored in `ref_buffer`.

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
    /// Supersampling factor per axis: 1 = off, 2 = 2×2 (4 samples).
    pub aa_level: u32,
    /// Iteration formula (`FractalKind::shader_id`).
    pub kind: u32,
    /// Exponent for the Multibrot kind.
    pub power: u32,
    /// Complex offset of the view center from the reference center, so a stale
    /// or reused reference (computed at a slightly different center) still maps
    /// correctly. Added to every pixel's per-pixel offset.
    pub dc_offset: [f32; 2],
    /// Padding to a 16-byte multiple (uniform buffer requirement).
    pub _pad: [u32; 2],
}

/// Offscreen texture the fractal is rendered into, plus the bind group used to
/// blit it. Recreated whenever the widget's pixel size changes.
struct CacheTarget {
    view: wgpu::TextureView,
    blit_bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

/// State the cache texture was last rendered with. If the next frame's inputs
/// match this, the cache is still valid and the fractal shader is skipped.
struct RenderedState {
    uniforms: Uniforms,
    generation: u64,
    width: u32,
    height: u32,
}

pub struct FractalRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    ref_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    target_format: wgpu::TextureFormat,
    /// Generation of the reference orbit currently uploaded to `ref_buffer`.
    uploaded_generation: u64,

    /// Blit pipeline + resources that copy the cache texture to egui's surface.
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    /// The offscreen cache; `None` until the first frame sizes it.
    cache: Option<CacheTarget>,
    /// What the cache currently holds; `None` forces a re-render.
    rendered: Option<RenderedState>,
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

        // Blit pipeline: samples the cache texture onto egui's surface.
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
        });

        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("blit bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit pipeline layout"),
            bind_group_layouts: &[Some(&blit_bind_group_layout)],
            immediate_size: 0,
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
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
            bind_group_layout,
            uniform_buffer,
            ref_buffer,
            bind_group,
            target_format,
            uploaded_generation: u64::MAX,
            blit_pipeline,
            blit_bind_group_layout,
            blit_sampler,
            cache: None,
            rendered: None,
        }
    }

    /// Ensure the cache texture exists at `width`×`height`. Recreates it (and its
    /// blit bind group) on a size change, invalidating any previous render.
    fn ensure_cache(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if let Some(c) = &self.cache
            && c.width == width
            && c.height == height
        {
            return;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fractal cache"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit bind group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        });

        self.cache = Some(CacheTarget {
            view,
            blit_bind_group,
            width,
            height,
        });
        // New texture → old render is gone.
        self.rendered = None;
    }

    /// Handles needed to build a standalone [`ExportRender`] off the UI thread:
    /// the (immutable) pipeline and its bind-group layout, plus the target
    /// format. Cloned so the caller can drop the render-state lock before use.
    pub fn export_handles(&self) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::TextureFormat) {
        (
            self.pipeline.clone(),
            self.bind_group_layout.clone(),
            self.target_format,
        )
    }
}

/// A self-contained render of one export image. It owns its own uniform and
/// reference buffers (a snapshot of the view at export time), so it is unaffected
/// by panning/zooming on the main thread, and can run on a background thread.
/// The image is rendered in horizontal tiles so progress can be reported as the
/// GPU works through it.
pub struct ExportRender {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    /// Padded bytes-per-row of the readback buffer.
    pub padded_bpr: u32,
    pub width: u32,
    pub height: u32,
    /// Number of horizontal tiles the render is split into.
    pub tiles: u32,
    pub swap_rb: bool,
}

impl ExportRender {
    /// Allocate the export's dedicated GPU resources and upload the snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline: wgpu::RenderPipeline,
        bind_group_layout: &wgpu::BindGroupLayout,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        uniforms: Uniforms,
        reference: &[[f32; 2]],
    ) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let count = reference.len().min(MAX_REF_POINTS);
        let ref_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export reference orbit"),
            size: (count.max(1) * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if count > 0 {
            queue.write_buffer(&ref_buffer, 0, bytemuck::cast_slice(&reference[..count]));
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("export bind group"),
            layout: bind_group_layout,
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
            format: target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = (width * 4).div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export readback"),
            size: (padded_bpr * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // ~128px bands, kept to a sane range so progress is smooth without too
        // many submissions.
        let tiles = (height / 128).clamp(8, 64).min(height.max(1));

        let swap_rb = matches!(
            target_format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );

        Self {
            pipeline,
            bind_group,
            texture,
            view,
            readback,
            padded_bpr,
            width,
            height,
            tiles,
            swap_rb,
        }
    }

    /// Pixel row range `[y0, y1)` covered by tile `t`.
    fn tile_rows(&self, t: u32) -> (u32, u32) {
        let band = self.height.div_ceil(self.tiles);
        let y0 = (t * band).min(self.height);
        let y1 = (y0 + band).min(self.height);
        (y0, y1)
    }

    /// Render one horizontal tile into the export texture and submit it. Tile 0
    /// clears the whole attachment; later tiles preserve earlier ones.
    pub fn render_tile(&self, device: &wgpu::Device, queue: &wgpu::Queue, t: u32) {
        let (y0, y1) = self.tile_rows(t);
        if y1 <= y0 {
            return;
        }
        let load = if t == 0 {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        } else {
            wgpu::LoadOp::Load
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("export tile"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("export tile pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // Full-viewport triangle (so pixel→plane mapping matches the whole
            // image), scissored to this tile's rows.
            pass.set_scissor_rect(0, y0, self.width, y1 - y0);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Copy the finished texture into the mappable readback buffer and submit.
    pub fn copy_to_readback(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("export copy"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bpr),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// The mappable readback buffer (valid after [`copy_to_readback`]).
    pub fn readback(&self) -> &wgpu::Buffer {
        &self.readback
    }
}

/// Convert a padded BGRA/RGBA readback into tightly-packed RGBA8 and encode it
/// as PNG bytes, reporting progress in `[0, 1]` via `on_progress` as rows are
/// streamed to the compressor (encoding is the slow, subdividable phase).
pub fn encode_png_with_progress(
    padded: &[u8],
    width: u32,
    height: u32,
    padded_bpr: u32,
    swap_rb: bool,
    mut on_progress: impl FnMut(f32),
) -> Vec<u8> {
    use std::io::Write as _;

    let row = (width * 4) as usize;
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("png header");
        let mut stream = writer.stream_writer().expect("png stream");
        let mut line = vec![0u8; row];
        for y in 0..height as usize {
            let src_off = y * padded_bpr as usize;
            let src = &padded[src_off..src_off + row];
            if swap_rb {
                for x in 0..width as usize {
                    line[x * 4] = src[x * 4 + 2];
                    line[x * 4 + 1] = src[x * 4 + 1];
                    line[x * 4 + 2] = src[x * 4];
                    line[x * 4 + 3] = src[x * 4 + 3];
                }
                stream.write_all(&line).expect("png data");
            } else {
                stream.write_all(src).expect("png data");
            }
            if y % 64 == 0 {
                on_progress(y as f32 / height as f32);
            }
        }
        stream.finish().expect("png finish");
    }
    on_progress(1.0);
    out
}

/// A per-frame paint callback. Carries this frame's uniforms plus a reference to
/// the current reference orbit (cheap `Arc` clone). The orbit is only re-uploaded
/// to the GPU when its `generation` changes, and the fractal is only re-rendered
/// into the cache when the uniforms, generation, or `size_px` change.
pub struct FractalCallback {
    pub uniforms: Uniforms,
    pub reference: Arc<Vec<[f32; 2]>>,
    pub generation: u64,
    /// Widget size in physical pixels — the cache texture resolution.
    pub size_px: [u32; 2],
}

impl egui_wgpu::CallbackTrait for FractalCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(renderer) = resources.get_mut::<FractalRenderer>() else {
            return Vec::new();
        };

        let width = self.size_px[0].max(1);
        let height = self.size_px[1].max(1);
        renderer.ensure_cache(device, width, height);

        if renderer.uploaded_generation != self.generation && !self.reference.is_empty() {
            let count = self.reference.len().min(MAX_REF_POINTS);
            queue.write_buffer(
                &renderer.ref_buffer,
                0,
                bytemuck::cast_slice(&self.reference[..count]),
            );
            renderer.uploaded_generation = self.generation;
        }

        // Re-render the cache only when what it depends on changed.
        let dirty = renderer.rendered.as_ref().is_none_or(|r| {
            r.generation != self.generation
                || r.width != width
                || r.height != height
                || bytemuck::bytes_of(&r.uniforms) != bytemuck::bytes_of(&self.uniforms)
        });
        if !dirty {
            return Vec::new();
        }

        queue.write_buffer(
            &renderer.uniform_buffer,
            0,
            bytemuck::bytes_of(&self.uniforms),
        );

        if let Some(cache) = &renderer.cache {
            let mut pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fractal cache pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &cache.view,
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
            pass.set_pipeline(&renderer.pipeline);
            pass.set_bind_group(0, &renderer.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        renderer.rendered = Some(RenderedState {
            uniforms: self.uniforms,
            generation: self.generation,
            width,
            height,
        });
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(renderer) = resources.get::<FractalRenderer>()
            && let Some(cache) = &renderer.cache
        {
            render_pass.set_pipeline(&renderer.blit_pipeline);
            render_pass.set_bind_group(0, &cache.blit_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }
}
