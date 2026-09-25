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

use std::collections::HashMap;
use std::sync::Arc;

use eframe::egui_wgpu::{self, wgpu};

use super::reference::RefOrbit;
use crate::lights::{GpuLight, Light, MAX_LIGHT_COUNT, gpu_lights};

/// Maximum reference-orbit length (points) the storage buffer can hold. Also
/// bounds the iteration count. 128k points * 8 bytes = 1 MiB.
pub const MAX_REF_POINTS: usize = 1 << 17;

/// Format of the intermediate iteration-data texture holding, per pixel,
/// `(ci, DE factor, interior fraction)`. 32-bit float keeps the smooth iteration
/// count precise at deep zoom. Color-renderable and read with nearest sampling
/// (iteration data must never be linearly filtered across escape boundaries), so
/// no `float32-filterable` feature is needed.
const DATA_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// Cap on the interactive cache's pixel count (the texture is scaled down,
/// aspect kept, above it). Each pixel costs 36 bytes across the data, AA and
/// colour textures, and the 3D view renders at a configurable multiple per axis
/// (default 2×), so a HiDPI screen
/// in 3D would otherwise want 0.5 GB+. Browsers cap WebGPU memory well below
/// what native gets, so the web budget is ~4K (≈300 MB); native, ~8K.
#[cfg(target_arch = "wasm32")]
const MAX_CACHE_PIXELS: u32 = 3840 * 2160;
#[cfg(not(target_arch = "wasm32"))]
const MAX_CACHE_PIXELS: u32 = 7680 * 4320;

/// True when the two uniforms differ in any field the iteration pass depends on
/// (i.e. anything except the palette / colour scale / offset / camera).
fn geom_differs(a: &Uniforms, b: &Uniforms) -> bool {
    a.span != b.span
        || a.max_iter != b.max_iter
        || a.ref_len != b.ref_len
        || a.bailout_sq != b.bailout_sq
        || a.is_julia != b.is_julia
        || a.aa_level != b.aa_level
        || a.kind != b.kind
        || a.power != b.power
        || a.complex_power != b.complex_power
        || a.dc_offset != b.dc_offset
        || a.scale_exp != b.scale_exp
        || a.phoenix_p != b.phoenix_p
        || a.lambda_l != b.lambda_l
        || a.morph_from != b.morph_from
        || a.morph_w != b.morph_w
        || a.de_coloring != b.de_coloring
        // The iterate pass's DE clamp (`max_de`) depends on whether any
        // shadow-style mode is on.
        || (a.rendering_mode != 0) != (b.rendering_mode != 0)
}

/// True when the two uniforms differ in a field only the colourise pass reads
/// (remappable without re-iterating): palette / colour scale / offset, the
/// shadow style and light count, and the 3D raymarch camera.
fn color_differs(a: &Uniforms, b: &Uniforms) -> bool {
    a.color_offset != b.color_offset
        || a.color_scale != b.color_scale
        || a.palette_id != b.palette_id
        || a.shadow_palette_id != b.shadow_palette_id
        || a.rendering_mode != b.rendering_mode
        || a.light_count != b.light_count
        || a.camera_direction != b.camera_direction
        || a.camera_inv_proj != b.camera_inv_proj
        || a.screen_dim != b.screen_dim
}

/// Specialization of the iteration shader (`mandelbrot.wgsl`'s `override`
/// constants). Everything the per-iteration loop branches on is baked into
/// the pipeline instead of tested per step; one pipeline set per key is built
/// lazily on first use (a new `FractalKind` needs nothing here).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct PipelineKey {
    kind: u32,
    julia: bool,
    de: bool,
    /// A kind-switch morph is in progress (`morph_w > 0`).
    morph: bool,
    /// Deep view: the delta starts out in rescaled (mantissa + exponent)
    /// form (`scale_exp != 0`).
    deep: bool,
}

impl PipelineKey {
    pub fn from_uniforms(u: &Uniforms) -> Self {
        Self {
            kind: u.kind,
            julia: u.is_julia != 0,
            de: u.de_coloring != 0,
            morph: u.morph_w > 0.0,
            deep: u.scale_exp != 0,
        }
    }

    fn constants(&self) -> [(&'static str, f64); 5] {
        [
            ("KIND", self.kind as f64),
            ("IS_JULIA", self.julia as u32 as f64),
            ("DE", self.de as u32 as f64),
            ("MORPH", self.morph as u32 as f64),
            ("DEEP", self.deep as u32 as f64),
        ]
    }
}

/// Build a fullscreen-triangle render pipeline (`vs_main` + `fs_entry`)
/// writing a single `format` target, with `constants` for the shader's
/// `override`s.
fn fullscreen_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    fs_entry: &str,
    format: wgpu::TextureFormat,
    constants: &[(&str, f64)],
) -> wgpu::RenderPipeline {
    let compilation_options = wgpu::PipelineCompilationOptions {
        constants,
        ..Default::default()
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: compilation_options.clone(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options,
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// The interactive iteration pipelines for one [`PipelineKey`].
struct IteratePipelines {
    /// 1-spp perturbation iterate → data texture (`fs_data`).
    iterate: wgpu::RenderPipeline,
    /// Adaptive AA: data texture → AA data texture (`fs_refine`).
    refine: wgpu::RenderPipeline,
}

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
    pub shadow_palette_id: u32,
    /// Supersampling factor per axis: 1 = off, 2 = 2×2 (4 samples).
    pub aa_level: u32,
    /// Iteration formula (`FractalKind::shader_id`).
    pub kind: u32,
    /// Exponent for the Multibrot kind.
    pub power: u32,
    /// Kind-switch morph: the kind being blended *from* (a `FractalKind`
    /// discriminant); only read when `morph_w > 0`.
    pub morph_from: u32,
    /// Complex offset of the view center from the reference center, so a stale
    /// or reused reference (computed at a slightly different center) still maps
    /// correctly. Added to every pixel's per-pixel offset.
    pub dc_offset: [f32; 2],
    /// Distortion constant `p` for the Phoenix map (`z^2 + c + p·z_{n-1}`);
    /// ignored by other kinds. Kept next to `dc_offset` so both `vec2`s land on
    /// 8-byte boundaries, matching the shader's layout.
    pub phoenix_p: [f32; 2],
    /// Distortion constant `l` for the Lambda map (`l·z(1 - z)`);
    /// ignored by other kinds.
    pub lambda_l: [f32; 2],
    /// Complex exponent for the Complex Multibrot kind (`z^power + c`);
    /// ignored by other kinds.
    pub complex_power: [f32; 2],
    /// 0 = escape-time coloring, 1 = distance-estimation shading.
    pub de_coloring: u32,
    // 0 = classic colors, 1 = shadows, 2 = 3D raymarching rendering
    pub rendering_mode: u32,
    // camera direction vector
    pub camera_direction: [f32; 3],
    /// Number of live entries in the lights buffer (see `gpu_lights`).
    pub light_count: u32,
    /// Inverse of the camera's view-projection matrix (column-major), for
    /// reconstructing a world-space ray origin per pixel in the raymarcher.
    pub camera_inv_proj: [f32; 16],
    /// Screen dimension
    pub screen_dim: [f32; 2],
    /// Kind-switch morph weight: each step is `(1 - w)·f_kind + w·f_from`.
    /// 0 = no morph (and the iteration pipeline is then specialized without
    /// the morph path, see [`PipelineKey`]).
    pub morph_w: f32,
    /// Binary exponent `E` of the deep (rescaled) view scale: `span` and
    /// `dc_offset` are uploaded multiplied by `2^-E`, so they stay inside
    /// f32's exponent range at any depth. Non-zero exactly when the deep
    /// pipeline is used (see [`PipelineKey`] and `mandelbrot.wgsl`'s `DEEP`).
    pub scale_exp: i32,
    /// Complex binomial coefficients `C(complex_power, k)`, k = 1..16, two per
    /// row (odd k in `[0..2]`, even k in `[2..4]`), for the Complex Multibrot
    /// delta series. Derived from `complex_power` alone.
    pub cm_coef: [[f32; 4]; 8],
}

/// Offscreen textures for the two-pass render, recreated whenever the widget's
/// pixel size changes:
/// * `data_view` — the 1-spp iteration pass's output (see [`DATA_FORMAT`]).
/// * `aa` — the adaptive-AA refine pass's output (only when AA is on).
/// * `color_view` — the colourise pass's output; the blit source.
///   plus the bind groups that read them.
struct CacheTarget {
    /// Kept so they can be `destroy()`ed on resize (see `ensure_cache`).
    textures: Vec<wgpu::Texture>,
    data_view: wgpu::TextureView,
    color_view: wgpu::TextureView,
    /// Refine pass input (group 1): the 1-spp data texture.
    refine_bind_group: wgpu::BindGroup,
    /// Colourise pass input: uniforms + the 1-spp data texture.
    colorize_bind_group: wgpu::BindGroup,
    /// Refine pass output + the colourise bind group reading it. Only
    /// allocated while AA is on: it's a second full-size `Rgba32Float`.
    aa: Option<(wgpu::TextureView, wgpu::BindGroup)>,
    /// Blit pass input: the colour texture + sampler.
    blit_bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

/// What the iteration-data texture was last computed with. If the next frame's
/// geometry inputs match, iteration is skipped and only colour may be redone.
struct IterState {
    uniforms: Uniforms,
    generation: u64,
    width: u32,
    height: u32,
}

/// What the colour texture was last computed with. If the next frame's colour
/// inputs (and size) match and iteration did not re-run, colourise is skipped.
struct ColorState {
    uniforms: Uniforms,
    lights: [GpuLight; MAX_LIGHT_COUNT],
    width: u32,
    height: u32,
}

pub struct FractalRenderer {
    /// `mandelbrot.wgsl`, specialized per [`PipelineKey`] at pipeline creation.
    shader: wgpu::ShaderModule,
    /// Layout of the iterate + export pipelines (group 0 only).
    pipeline_layout: wgpu::PipelineLayout,
    /// Layout of the refine pipeline (group 0 + the 1-spp texture in group 1).
    refine_pipeline_layout: wgpu::PipelineLayout,
    refine_bind_group_layout: wgpu::BindGroupLayout,
    /// Lazily built interactive pipelines, per shader specialization.
    pipelines: HashMap<PipelineKey, IteratePipelines>,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    ref_buffer: wgpu::Buffer,
    ref_exp_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    target_format: wgpu::TextureFormat,
    /// Generation of the reference orbit currently uploaded to `ref_buffer`.
    uploaded_generation: u64,
    /// Contents of `lights_buffer`, so it's only re-uploaded on change.
    uploaded_lights: Option<[GpuLight; MAX_LIGHT_COUNT]>,

    /// Colourise pass: data texture → colour texture (palette mapping).
    colorize_pipeline: wgpu::RenderPipeline,
    colorize_bind_group_layout: wgpu::BindGroupLayout,

    /// Blit pipeline + resources that copy the colour texture to egui's surface.
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    /// The offscreen textures; `None` until the first frame sizes them.
    cache: Option<CacheTarget>,
    /// What the data texture holds; `None` forces re-iteration.
    iterated: Option<IterState>,
    /// What the colour texture holds; `None` forces a recolour.
    colored: Option<ColorState>,
}

impl FractalRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mandelbrot"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../shaders/common.wgsl"),
                    include_str!("../shaders/iterate_uniforms.wgsl"),
                    include_str!("../shaders/mandelbrot.wgsl"),
                )
                .into(),
            ),
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

        // Per-point exponents of the reference orbit (`RefOrbit::exps`), only
        // read by deep pipelines.
        let ref_exp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reference orbit exponents"),
            size: (MAX_REF_POINTS * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lights parameters"),
            size: std::mem::size_of::<[GpuLight; MAX_LIGHT_COUNT]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
                // Only read by the export pipeline's shadow branch (`fs_color`
                // with the custom-lights palette); the iterate pipeline
                // (`fs_data`) ignores it, but both pipelines share this layout.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ref_exp_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fractal pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // The iterate/refine/export pipelines are specialized per fractal
        // kind (see `PipelineKey`) and built lazily; only their layouts are
        // fixed. Refine additionally reads the 1-spp data texture (group 1).
        let refine_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("refine bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let refine_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("refine pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout), Some(&refine_bind_group_layout)],
                immediate_size: 0,
            });

        // Colourise pass: data texture + colour uniforms → colour texture.
        let colorize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("colorize"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../shaders/common.wgsl"),
                    include_str!("../shaders/iterate_uniforms.wgsl"),
                    include_str!("../shaders/colorize.wgsl"),
                )
                .into(),
            ),
        });
        let colorize_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("colorize bind group layout"),
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
                        ty: wgpu::BindingType::Texture {
                            // Nearest only: iteration data must not be filtered.
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let colorize_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("colorize pipeline layout"),
                bind_group_layouts: &[Some(&colorize_bind_group_layout)],
                immediate_size: 0,
            });
        let colorize_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("colorize pipeline"),
            layout: Some(&colorize_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &colorize_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &colorize_shader,
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
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../shaders/common.wgsl"),
                    include_str!("../shaders/blit.wgsl"),
                )
                .into(),
            ),
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
            shader,
            pipeline_layout,
            refine_pipeline_layout,
            refine_bind_group_layout,
            pipelines: HashMap::new(),
            bind_group_layout,
            uniform_buffer,
            ref_buffer,
            ref_exp_buffer,
            lights_buffer,
            bind_group,
            target_format,
            uploaded_generation: u64::MAX,
            uploaded_lights: None,
            colorize_pipeline,
            colorize_bind_group_layout,
            blit_pipeline,
            blit_bind_group_layout,
            blit_sampler,
            cache: None,
            iterated: None,
            colored: None,
        }
    }

    /// Ensure the cache textures exist at `width`×`height` (plus the AA refine
    /// target iff `aa`). Recreates them (and their bind groups) on a change,
    /// invalidating any previous render.
    fn ensure_cache(&mut self, device: &wgpu::Device, width: u32, height: u32, aa: bool) {
        if let Some(c) = &self.cache
            && c.width == width
            && c.height == height
            && c.aa.is_some() == aa
        {
            return;
        }

        // Free the old textures explicitly. On the web backend dropping a
        // `wgpu::Texture` does not release its GPU memory — that waits for the
        // JS garbage collector — and this runs on every size change (including
        // each switch in/out of the downscaled interactive resolution), so the
        // stale Rgba32Float textures piled up until WebGPU ran out of memory.
        // Any commands using them were submitted on earlier frames, and
        // `destroy()` defers the actual free until those finish.
        if let Some(old) = self.cache.take() {
            for t in &old.textures {
                t.destroy();
            }
        }

        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // Iteration-data texture (color-independent escape data).
        let data_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fractal data"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DATA_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let data_view = data_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Adaptive-AA output: same format, written by the refine pass.
        let data_aa_texture = aa.then(|| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fractal data (AA)"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DATA_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        });

        // Colour texture (colourise output; blit source).
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fractal color cache"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let colorize_bind_group_for = |data: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("colorize bind group"),
                layout: &self.colorize_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(data),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.lights_buffer.as_entire_binding(),
                    },
                ],
            })
        };
        let colorize_bind_group = colorize_bind_group_for(&data_view);
        let aa_target = data_aa_texture.as_ref().map(|t| {
            let view = t.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = colorize_bind_group_for(&view);
            (view, bind_group)
        });

        let refine_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("refine bind group"),
            layout: &self.refine_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&data_view),
            }],
        });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit bind group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        });

        self.cache = Some(CacheTarget {
            textures: [Some(data_texture), data_aa_texture, Some(color_texture)]
                .into_iter()
                .flatten()
                .collect(),
            data_view,
            color_view,
            refine_bind_group,
            colorize_bind_group,
            aa: aa_target,
            blit_bind_group,
            width,
            height,
        });
        // New textures → old renders are gone.
        self.iterated = None;
        self.colored = None;
    }

    /// Build (on first use) and cache the interactive pipelines for `key`.
    fn ensure_pipelines(&mut self, device: &wgpu::Device, key: PipelineKey) {
        if self.pipelines.contains_key(&key) {
            return;
        }
        let constants = key.constants();
        let iterate = fullscreen_pipeline(
            device,
            "fractal iterate pipeline",
            &self.shader,
            &self.pipeline_layout,
            "fs_data",
            DATA_FORMAT,
            &constants,
        );
        let refine = fullscreen_pipeline(
            device,
            "fractal AA refine pipeline",
            &self.shader,
            &self.refine_pipeline_layout,
            "fs_refine",
            DATA_FORMAT,
            &constants,
        );
        self.pipelines
            .insert(key, IteratePipelines { iterate, refine });
    }

    /// Handles needed to build a standalone [`ExportRender`] off the UI thread:
    /// a combined iterate + colour pipeline (`fs_color`) specialized for
    /// `uniforms` (built fresh — exports are rare, and this only needs a read
    /// lock on the renderer), its bind-group layout, and the target format.
    /// In 3D mode (`rendering_mode == 2`) also the interactive iterate →
    /// refine → colourise chain, since the raymarcher needs a whole data
    /// texture to march over and `fs_color` has no 3D path.
    pub fn export_handles(&self, device: &wgpu::Device, uniforms: &Uniforms) -> ExportHandles {
        let constants = PipelineKey::from_uniforms(uniforms).constants();
        let pipeline = fullscreen_pipeline(
            device,
            "fractal export pipeline",
            &self.shader,
            &self.pipeline_layout,
            "fs_color",
            self.target_format,
            &constants,
        );
        let raymarch = (uniforms.rendering_mode == 2).then(|| RaymarchHandles {
            iterate: fullscreen_pipeline(
                device,
                "fractal export iterate pipeline",
                &self.shader,
                &self.pipeline_layout,
                "fs_data",
                DATA_FORMAT,
                &constants,
            ),
            refine: fullscreen_pipeline(
                device,
                "fractal export AA refine pipeline",
                &self.shader,
                &self.refine_pipeline_layout,
                "fs_refine",
                DATA_FORMAT,
                &constants,
            ),
            colorize: self.colorize_pipeline.clone(),
            refine_bind_group_layout: self.refine_bind_group_layout.clone(),
            colorize_bind_group_layout: self.colorize_bind_group_layout.clone(),
        });
        ExportHandles {
            pipeline,
            bind_group_layout: self.bind_group_layout.clone(),
            format: self.target_format,
            raymarch,
        }
    }
}

/// Everything an [`ExportRender`] needs from the [`FractalRenderer`], cloned
/// out so the export can run off the UI thread (see `export_handles`).
#[derive(Clone)]
pub struct ExportHandles {
    /// Combined iterate + colour pipeline (`fs_color`), for 2D modes.
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
    /// The two-pass chain, for 3D mode only.
    raymarch: Option<RaymarchHandles>,
}

/// The interactive two-pass pipelines, for a 3D export.
#[derive(Clone)]
struct RaymarchHandles {
    iterate: wgpu::RenderPipeline,
    refine: wgpu::RenderPipeline,
    colorize: wgpu::RenderPipeline,
    refine_bind_group_layout: wgpu::BindGroupLayout,
    colorize_bind_group_layout: wgpu::BindGroupLayout,
}

/// A 3D export's own data textures and the passes that fill them: the tiles
/// iterate into `data_view`, then one refine (if AA) + colourise pass
/// raymarches the finished height field into the export target.
struct RaymarchExport {
    iterate: wgpu::RenderPipeline,
    /// Refine pipeline, output view and input bind group, when AA is on.
    refine: Option<(wgpu::RenderPipeline, wgpu::TextureView, wgpu::BindGroup)>,
    colorize: wgpu::RenderPipeline,
    colorize_bind_group: wgpu::BindGroup,
    data_view: wgpu::TextureView,
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
    /// 3D mode: tiles fill a data texture instead of the target.
    raymarch: Option<RaymarchExport>,
}

impl ExportRender {
    /// Allocate the export's dedicated GPU resources and upload the snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        handles: &ExportHandles,
        width: u32,
        height: u32,
        uniforms: Uniforms,
        reference: &RefOrbit,
        lights: &[Light],
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
        let ref_exp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export reference orbit exponents"),
            size: (count.max(1) * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if count > 0 {
            queue.write_buffer(&ref_buffer, 0, bytemuck::cast_slice(&reference[..count]));
            queue.write_buffer(
                &ref_exp_buffer,
                0,
                bytemuck::cast_slice(&reference.exps[..count]),
            );
        }

        // Only read by the shadow branch's custom-lights palette; harmless
        // (zeroed) for every other coloring mode.
        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export lights"),
            size: std::mem::size_of::<[GpuLight; MAX_LIGHT_COUNT]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (gpu_lights, _) = gpu_lights(lights);
        queue.write_buffer(&lights_buffer, 0, bytemuck::cast_slice(&gpu_lights));

        let target_format = handles.format;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("export bind group"),
            layout: &handles.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ref_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ref_exp_buffer.as_entire_binding(),
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

        let raymarch = handles.raymarch.as_ref().map(|rm| {
            let data_texture = |label| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d {
                            width,
                            height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: DATA_FORMAT,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&wgpu::TextureViewDescriptor::default())
            };
            let data_view = data_texture("export data");
            let refine = (uniforms.aa_level > 1).then(|| {
                let refine_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("export refine bind group"),
                    layout: &rm.refine_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&data_view),
                    }],
                });
                (
                    rm.refine.clone(),
                    data_texture("export data (AA)"),
                    refine_bind_group,
                )
            });
            // Colourise reads the refined texture when AA is on.
            let colorize_input = refine.as_ref().map_or(&data_view, |(_, v, _)| v);
            let colorize_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("export colorize bind group"),
                layout: &rm.colorize_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(colorize_input),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: lights_buffer.as_entire_binding(),
                    },
                ],
            });
            RaymarchExport {
                iterate: rm.iterate.clone(),
                refine,
                colorize: rm.colorize.clone(),
                colorize_bind_group,
                data_view,
            }
        });

        Self {
            pipeline: handles.pipeline.clone(),
            bind_group,
            texture,
            view,
            readback,
            padded_bpr,
            width,
            height,
            tiles,
            swap_rb,
            raymarch,
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
    /// clears the whole attachment; later tiles preserve earlier ones. In 3D
    /// mode the tiles iterate into the data texture instead, and the last one
    /// also runs the (whole-image) refine + raymarching colourise passes.
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
        let (target, pipeline) = match &self.raymarch {
            Some(rm) => (&rm.data_view, &rm.iterate),
            None => (&self.view, &self.pipeline),
        };
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("export tile pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
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
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        if let Some(rm) = &self.raymarch
            && y1 == self.height
        {
            if let Some((refine, aa_view, refine_bind_group)) = &rm.refine {
                data_pass(
                    &mut encoder,
                    "export AA refine pass",
                    aa_view,
                    refine,
                    &[&self.bind_group, refine_bind_group],
                );
            }
            data_pass(
                &mut encoder,
                "export colorize pass",
                &self.view,
                &rm.colorize,
                &[&rm.colorize_bind_group],
            );
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

/// Render `er` tile by tile (blocking on the GPU after each tile so progress
/// reflects real work), read it back, and encode the result as PNG bytes.
/// Blocks the calling thread throughout, so it's only for native targets:
/// the UI export path runs it on a background thread, headless rendering
/// runs it directly since it has no frame loop to share a thread with.
#[cfg(not(target_arch = "wasm32"))]
pub fn export_to_png_blocking(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    er: &ExportRender,
    mut on_progress: impl FnMut(&'static str, f32),
) -> Vec<u8> {
    // Progress budget: rendering fills [0, RENDER_END], encoding the rest.
    const RENDER_END: f32 = 0.6;

    for t in 0..er.tiles {
        er.render_tile(device, queue, t);
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let done = (t + 1) as f32 / er.tiles as f32;
        on_progress("Rendering", RENDER_END * done);
    }
    er.copy_to_readback(device, queue);

    let (tx, rx) = std::sync::mpsc::channel();
    er.readback()
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let _ = rx.recv();

    on_progress("Encoding", RENDER_END);
    let png = {
        let data = er
            .readback()
            .slice(..)
            .get_mapped_range()
            .expect("map readback buffer");
        encode_png_with_progress(&data, er.width, er.height, er.padded_bpr, er.swap_rb, |f| {
            on_progress("Encoding", RENDER_END + (0.97 - RENDER_END) * f)
        })
    };
    er.readback().unmap();
    png
}

/// Render every tile of `er` in one go (no per-tile GPU stall, unlike
/// [`export_to_png_blocking`]), read it back, and return a copy of the padded
/// readback bytes (`er.padded_bpr` per row) for [`encode_png`]. Used by the
/// headless animation pipeline, which encodes on other threads.
#[cfg(not(target_arch = "wasm32"))]
pub fn render_readback_blocking(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    er: &ExportRender,
) -> Vec<u8> {
    for t in 0..er.tiles {
        er.render_tile(device, queue, t);
    }
    er.copy_to_readback(device, queue);

    let (tx, rx) = std::sync::mpsc::channel();
    er.readback()
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let _ = rx.recv();

    let bytes = er
        .readback()
        .slice(..)
        .get_mapped_range()
        .expect("map readback buffer")
        .to_vec();
    er.readback().unmap();
    bytes
}

/// Like [`encode_png_with_progress`], but encodes the whole image at once
/// (no progress) at the given compression level. Non-streaming, so the fast
/// `fdeflate` levels don't pay the streaming-mode size penalty.
#[cfg(not(target_arch = "wasm32"))]
pub fn encode_png(
    padded: &[u8],
    width: u32,
    height: u32,
    padded_bpr: u32,
    swap_rb: bool,
    compression: png::Compression,
) -> Vec<u8> {
    let row = (width * 4) as usize;
    let mut pixels = Vec::with_capacity(row * height as usize);
    for y in 0..height as usize {
        let src_off = y * padded_bpr as usize;
        let src = &padded[src_off..src_off + row];
        if swap_rb {
            pixels.extend(
                src.as_chunks::<4>()
                    .0
                    .iter()
                    .flat_map(|&[b, g, r, a]| [r, g, b, a]),
            );
        } else {
            pixels.extend_from_slice(src);
        }
    }

    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(compression);
        let mut writer = encoder.write_header().expect("png header");
        writer.write_image_data(&pixels).expect("png data");
    }
    out
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

/// Record one fullscreen-triangle pass drawing `pipeline` into `target`
/// (cleared first), with `bind_groups` bound to groups 0, 1, ...
fn data_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_groups: &[&wgpu::BindGroup],
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
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
    pass.set_pipeline(pipeline);
    for (i, bg) in bind_groups.iter().enumerate() {
        pass.set_bind_group(i as u32, *bg, &[]);
    }
    pass.draw(0..3, 0..1);
}

/// A per-frame paint callback. Carries this frame's uniforms plus a reference to
/// the current reference orbit (cheap `Arc` clone). The orbit is only re-uploaded
/// when its `generation` changes; the expensive iteration pass re-runs only when
/// a geometry input changes, colour-only changes re-run just the cheap
/// colourise pass, and a frame where nothing changed (e.g. a hover repaint)
/// uploads and renders nothing — `paint` just blits the cache (see `prepare`).
pub struct FractalCallback {
    pub uniforms: Uniforms,
    /// Lights buffer contents, from [`gpu_lights`] (its count is in
    /// `uniforms.light_count`).
    pub lights: [GpuLight; MAX_LIGHT_COUNT],
    pub reference: Arc<RefOrbit>,
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

        // Clamp to the device's texture-size limit, keeping the aspect ratio
        // (the iterate pass maps pixels through NDC, so the view is unchanged;
        // the blit just upsamples). The 3D supersample (default 2×) on a large/HiDPI
        // screen can otherwise exceed it.
        // Also cap the total pixel count (`MAX_CACHE_PIXELS`), same way.
        let max_dim = device.limits().max_texture_dimension_2d;
        let [w, h] = self.size_px.map(|v| v.max(1));
        let scale = (max_dim as f64 / w.max(h) as f64)
            .min((MAX_CACHE_PIXELS as f64 / (w as f64 * h as f64)).sqrt())
            .min(1.0);
        let width = ((w as f64 * scale) as u32).clamp(1, max_dim);
        let height = ((h as f64 * scale) as u32).clamp(1, max_dim);
        let aa = self.uniforms.aa_level > 1;
        renderer.ensure_cache(device, width, height, aa);

        // Iteration (expensive) re-runs only when the geometry inputs change;
        // colourise (cheap) re-runs when it did, or when only a colour/camera/
        // light input changed — so palette tweaks, colour cycling, and 3D
        // camera moves skip the perturbation entirely.
        let iter_dirty = renderer.iterated.as_ref().is_none_or(|r| {
            r.generation != self.generation
                || r.width != width
                || r.height != height
                || geom_differs(&r.uniforms, &self.uniforms)
        });
        let color_dirty = iter_dirty
            || renderer.colored.as_ref().is_none_or(|c| {
                c.width != width
                    || c.height != height
                    || c.lights != self.lights
                    || color_differs(&c.uniforms, &self.uniforms)
            });

        if !color_dirty {
            return Vec::new(); // cache still valid; paint() just blits it
        }

        if iter_dirty
            && renderer.uploaded_generation != self.generation
            && !self.reference.is_empty()
        {
            let count = self.reference.len().min(MAX_REF_POINTS);
            queue.write_buffer(
                &renderer.ref_buffer,
                0,
                bytemuck::cast_slice(&self.reference[..count]),
            );
            queue.write_buffer(
                &renderer.ref_exp_buffer,
                0,
                bytemuck::cast_slice(&self.reference.exps[..count]),
            );
            renderer.uploaded_generation = self.generation;
        }

        // Every pass reads the uniform buffer; refresh it once.
        queue.write_buffer(
            &renderer.uniform_buffer,
            0,
            bytemuck::bytes_of(&self.uniforms),
        );
        if renderer.uploaded_lights.as_ref() != Some(&self.lights) {
            queue.write_buffer(
                &renderer.lights_buffer,
                0,
                bytemuck::cast_slice(&self.lights),
            );
            renderer.uploaded_lights = Some(self.lights);
        }

        if iter_dirty {
            renderer.ensure_pipelines(device, PipelineKey::from_uniforms(&self.uniforms));
        }
        let pipelines = &renderer.pipelines[&PipelineKey::from_uniforms(&self.uniforms)];
        if let Some(cache) = &renderer.cache {
            if iter_dirty {
                // Iteration pass: 1-spp perturbation iterate → data texture.
                data_pass(
                    egui_encoder,
                    "fractal iterate pass",
                    &cache.data_view,
                    &pipelines.iterate,
                    &[&renderer.bind_group],
                );
                if let Some((data_aa_view, _)) = &cache.aa {
                    // Adaptive AA: supersample only the non-smooth pixels.
                    data_pass(
                        egui_encoder,
                        "fractal AA refine pass",
                        data_aa_view,
                        &pipelines.refine,
                        &[&renderer.bind_group, &cache.refine_bind_group],
                    );
                }
            }

            // Colourise pass: data texture → colour texture.
            let colorize_bind_group = cache
                .aa
                .as_ref()
                .map_or(&cache.colorize_bind_group, |(_, bg)| bg);
            data_pass(
                egui_encoder,
                "fractal colorize pass",
                &cache.color_view,
                &renderer.colorize_pipeline,
                &[colorize_bind_group],
            );
        }

        if iter_dirty {
            renderer.iterated = Some(IterState {
                uniforms: self.uniforms,
                generation: self.generation,
                width,
                height,
            });
        }
        renderer.colored = Some(ColorState {
            uniforms: self.uniforms,
            lights: self.lights,
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
