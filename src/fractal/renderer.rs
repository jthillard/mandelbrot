//! wgpu resources for the fractal: the render pipeline, the uniform buffer, the
//! reference-orbit storage buffer, and the egui paint callback that drives them.
//!
//! Rendering strategy: the expensive per-pixel perturbation shader renders into
//! an offscreen **cache texture**, and only when the view/coloring/size actually
//! change (tracked by `rendered`). Every egui frame then just blits that cached
//! texture onto egui's surface with a cheap textured fullscreen triangle — so
//! incidental repaints (mouse-move, hover, the worker-pending poll) cost a blit,
//! not a full fractal recompute. The fragment shader iterates each pixel as an
//! f32 perturbation delta from the reference orbit stored in `RefData`.

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "gui")]
use eframe::egui_wgpu;
use wgpu::util::DeviceExt as _;

use super::bla::BlaTable;
use super::kind::FractalKind;
use super::reference::RefOrbit;
use crate::lights::{GpuLight, Light, MAX_LIGHT_COUNT, gpu_lights};

/// Maximum reference-orbit length (points), and so the hard ceiling on the
/// iteration count (the shader treats an exhausted reference as escaped).
/// 16M points * 8 bytes = 128 MiB, WebGPU's default
/// `max_storage_buffer_binding_size`, so every device can bind it.
pub const MAX_REF_POINTS: usize = 1 << 24;

/// Initial capacity (points) of the interactive reference buffers; they grow
/// (by powers of two, up to `MAX_REF_POINTS`) when a longer orbit arrives.
const INITIAL_REF_POINTS: usize = 1 << 17;

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
    /// BLA jumps (`bla::applies`; disabling BLA uploads an empty table
    /// instead, see `BlaTable::empty`).
    bla: bool,
}

impl PipelineKey {
    pub fn from_uniforms(u: &Uniforms) -> Self {
        Self {
            kind: u.kind,
            julia: u.is_julia != 0,
            de: u.de_coloring != 0,
            morph: u.morph_w > 0.0,
            deep: u.scale_exp != 0,
            bla: super::bla::applies(u),
        }
    }

    fn constants(&self) -> [(&'static str, f64); 6] {
        [
            ("KIND", self.kind as f64),
            ("IS_JULIA", self.julia as u32 as f64),
            ("DE", self.de as u32 as f64),
            ("MORPH", self.morph as u32 as f64),
            ("DEEP", self.deep as u32 as f64),
            ("BLA", self.bla as u32 as f64),
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

/// Stride between the per-pass step uniforms in [`Lipschitz::steps`]
/// (WebGPU's minimum uniform-buffer offset alignment).
const LIPSCHITZ_STRIDE: u32 = 256;
/// Step uniforms held: slot k holds step 2^k, enough for any texture size.
const LIPSCHITZ_SLOTS: u32 = 32;

/// Whether the shadow / 3D view of `u` rebuilds its DE height field as a
/// distance field (see `lipschitz.wgsl`): only Complex Multibrot, whose
/// branch cut makes the DE jump (seams in shadow, walls in 3D). Every other
/// kind and classic colouring keep the DE as is.
fn wants_envelope(u: &Uniforms) -> bool {
    let cm = FractalKind::ComplexMultibrot as u32;
    u.rendering_mode != 0 && (u.kind == cm || (u.morph_w > 0.0 && u.morph_from == cm))
}

/// The distance-field passes (`lipschitz.wgsl`): seed, jump-flood and
/// compose pipelines, their shared input layout (data texture, seed texture,
/// step uniform at a dynamic offset) and the buffer of every pass's step.
#[derive(Clone)]
struct Lipschitz {
    seed: wgpu::RenderPipeline,
    jump: wgpu::RenderPipeline,
    compose: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    steps: wgpu::Buffer,
}

impl Lipschitz {
    fn new(device: &wgpu::Device) -> Self {
        let module = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("lipschitz"),
                    source: wgpu::ShaderSource::Wgsl(
                        concat!(
                            include_str!("../shaders/common.wgsl"),
                            include_str!("../shaders/lipschitz.wgsl"),
                        )
                        .into(),
                    ),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lipschitz bind group layout"),
            entries: &[
                texture_entry(0),
                texture_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lipschitz pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |label, entry| {
            fullscreen_pipeline(
                device,
                label,
                &module,
                &pipeline_layout,
                entry,
                DATA_FORMAT,
                &[],
            )
        };
        let mut contents = vec![0u8; (LIPSCHITZ_STRIDE * LIPSCHITZ_SLOTS) as usize];
        for k in 0..LIPSCHITZ_SLOTS {
            let at = (k * LIPSCHITZ_STRIDE) as usize;
            contents[at..at + 4].copy_from_slice(&(1i32 << k.min(30)).to_ne_bytes());
        }
        let steps = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lipschitz steps"),
            contents: &contents,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        Self {
            seed: pipeline("lipschitz seed pipeline", "fs_seed"),
            jump: pipeline("lipschitz jump pipeline", "fs_jump"),
            compose: pipeline("lipschitz compose pipeline", "fs_compose"),
            layout,
            steps,
        }
    }

    /// A pass input reading the data texture `data` and seed texture `seeds`.
    fn bind_group(
        &self,
        device: &wgpu::Device,
        data: &wgpu::TextureView,
        seeds: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lipschitz bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(data),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(seeds),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.steps,
                        offset: 0,
                        size: wgpu::BufferSize::new(16),
                    }),
                },
            ],
        })
    }
}

/// Targets for the distance field of one data texture: two ping-pong seed
/// textures for jump flooding, and the output (data with the rebuilt DE).
struct Envelope {
    /// `[seeds A, seeds B, output]`, kept so they can be `destroy()`ed with
    /// the rest of the cache.
    textures: [wgpu::Texture; 3],
    views: [wgpu::TextureView; 3],
    /// Pass inputs: the data texture with seeds A, and with seeds B.
    inputs: [wgpu::BindGroup; 2],
    /// Step slot of each jump pass, in order (see [`lipschitz_slots`]).
    slots: Vec<u32>,
}

impl Envelope {
    fn new(
        device: &wgpu::Device,
        lp: &Lipschitz,
        data: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let make = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
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
        };
        let textures = [
            make("DE distance seeds A"),
            make("DE distance seeds B"),
            make("DE distance field"),
        ];
        let views = textures
            .each_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));
        let inputs = [
            lp.bind_group(device, data, &views[0]),
            lp.bind_group(device, data, &views[1]),
        ];
        Self {
            textures,
            views,
            inputs,
            slots: lipschitz_slots(width, height),
        }
    }

    /// The data texture with the rebuilt DE.
    fn output(&self) -> &wgpu::TextureView {
        &self.views[2]
    }

    /// Record seed → jump passes → compose into [`Self::output`].
    fn record(&self, encoder: &mut wgpu::CommandEncoder, lp: &Lipschitz) {
        let pass = |encoder: &mut wgpu::CommandEncoder,
                    target: &wgpu::TextureView,
                    pipeline: &wgpu::RenderPipeline,
                    input: &wgpu::BindGroup,
                    slot: u32| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("lipschitz pass"),
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
            pass.set_bind_group(0, input, &[slot * LIPSCHITZ_STRIDE]);
            pass.draw(0..3, 0..1);
        };
        // Seeds into A (the bound seed texture, B, is unread).
        pass(encoder, &self.views[0], &lp.seed, &self.inputs[1], 0);
        // Pass i reads seeds i % 2 and writes the other.
        for (i, &slot) in self.slots.iter().enumerate() {
            pass(
                encoder,
                &self.views[(i + 1) % 2],
                &lp.jump,
                &self.inputs[i % 2],
                slot,
            );
        }
        let last = self.slots.len() % 2;
        pass(encoder, &self.views[2], &lp.compose, &self.inputs[last], 0);
    }
}

/// Jump-flooding step slots (step = 2^slot) for a `width`×`height` texture:
/// from about half its larger side down to 1, then one more step-1 pass,
/// which fixes most of jump flooding's residual errors.
fn lipschitz_slots(width: u32, height: u32) -> Vec<u32> {
    let top = (width.max(height) / 2).max(1).ilog2();
    (0..=top).rev().chain(std::iter::once(0)).collect()
}

/// Histogram bins of `ci_stats.wgsl` (must match its `CI_BINS`).
const CI_BINS: usize = 1024;
/// Upper end of the histogram's log2(1 + ci) range (`CI_LOG2_MAX` in WGSL).
const CI_LOG2_MAX: f32 = 25.0;
/// Fraction of escaped pixels ignored at each end of the `ci` range the auto
/// colour scale fits: the few pixels hugging the boundary have `ci` near
/// `max_iter` and would otherwise squash the rest into one palette band.
const CI_TAIL: f64 = 0.005;

/// Longest side (px) of an export's auto-colour prepass
/// ([`ExportRender::ci_range_blocking`]).
#[cfg(not(target_arch = "wasm32"))]
const CI_PROBE_MAX_DIM: u32 = 1024;

/// Where the `ci` histogram readback is (see [`CiStats`]).
enum CiStatsState {
    Idle,
    /// Histogram copied into `staging` by an encoder egui hasn't submitted
    /// yet: mapping must wait for the next frame.
    Copied,
    /// `map_async` issued; the flag is set once the mapping completes.
    Mapping(Arc<std::sync::atomic::AtomicBool>),
}

/// Side (texels) of [`CiHistogram::Readback`]'s sample grid. Must match
/// `CI_SAMPLE_DIM` in `ci_sample.wgsl`.
const CI_SAMPLE_DIM: u32 = 256;
/// Format of the sample grid: (ci, interior fraction, 0, 0). Rgba rather
/// than Rg because WebGL2 only guarantees float readback of RGBA.
const CI_SAMPLE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// The `ci` histogram of a data texture, shared by the interactive
/// [`CiStats`] and headless exports ([`ExportRender::ci_range_blocking`]).
#[derive(Clone)]
enum CiHistogram {
    /// A compute pass bins every pixel with atomics (`ci_stats.wgsl`).
    Compute {
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
    },
    /// No compute shaders (WebGL2, [`GpuPath::Texture`]): a render pass
    /// point-samples the data onto a `CI_SAMPLE_DIM`² grid
    /// (`ci_sample.wgsl`), which is read back and binned on the CPU.
    Readback {
        pipeline: wgpu::RenderPipeline,
        layout: wgpu::BindGroupLayout,
        /// The sample grid, shared by every user (passes are ordered on the
        /// queue, so they don't overlap).
        target: wgpu::Texture,
    },
}

impl CiHistogram {
    fn new(device: &wgpu::Device, path: GpuPath) -> Self {
        let data_entry = |visibility| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        if path == GpuPath::Texture {
            let module = unsafe {
                device.create_shader_module_trusted(
                    wgpu::ShaderModuleDescriptor {
                        label: Some("ci sample"),
                        source: wgpu::ShaderSource::Wgsl(
                            concat!(
                                include_str!("../shaders/common.wgsl"),
                                include_str!("../shaders/ci_sample.wgsl"),
                            )
                            .into(),
                        ),
                    },
                    wgpu::ShaderRuntimeChecks::unchecked(),
                )
            };
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ci sample bind group layout"),
                entries: &[data_entry(wgpu::ShaderStages::FRAGMENT)],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ci sample pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = fullscreen_pipeline(
                device,
                "ci sample pipeline",
                &module,
                &pipeline_layout,
                "fs_main",
                CI_SAMPLE_FORMAT,
                &[],
            );
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("ci sample grid"),
                size: wgpu::Extent3d {
                    width: CI_SAMPLE_DIM,
                    height: CI_SAMPLE_DIM,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: CI_SAMPLE_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            return Self::Readback {
                pipeline,
                layout,
                target,
            };
        }
        let module = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("ci stats"),
                    source: wgpu::ShaderSource::Wgsl(
                        include_str!("../shaders/ci_stats.wgsl").into(),
                    ),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ci stats bind group layout"),
            entries: &[
                data_entry(wgpu::ShaderStages::COMPUTE),
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ci stats pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ci stats pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self::Compute { pipeline, layout }
    }

    /// Bytes per row of the `Readback` grid's copy (already a multiple of
    /// `COPY_BYTES_PER_ROW_ALIGNMENT`).
    const SAMPLE_BPR: u32 = CI_SAMPLE_DIM * 16;

    /// The histogram buffer (`Compute` only) and the mappable readback copy.
    fn buffers(&self, device: &wgpu::Device) -> (Option<wgpu::Buffer>, wgpu::Buffer) {
        let hist_size = (CI_BINS * std::mem::size_of::<u32>()) as u64;
        let (hist, staging_size) = match self {
            Self::Compute { .. } => {
                let hist = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ci histogram"),
                    size: hist_size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                (Some(hist), hist_size)
            }
            Self::Readback { .. } => (None, (Self::SAMPLE_BPR * CI_SAMPLE_DIM) as u64),
        };
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ci histogram readback"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (hist, staging)
    }

    /// Record the histogram of `data` (`width`×`height`) into `hist` (or the
    /// sample grid), and its copy into `staging`.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        data: &wgpu::TextureView,
        width: u32,
        height: u32,
        hist: Option<&wgpu::Buffer>,
        staging: &wgpu::Buffer,
    ) {
        match self {
            Self::Compute { pipeline, layout } => {
                let hist = hist.expect("compute histogram without its buffer");
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ci stats bind group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(data),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: hist.as_entire_binding(),
                        },
                    ],
                });
                encoder.clear_buffer(hist, 0, None);
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("ci stats pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
                }
                encoder.copy_buffer_to_buffer(hist, 0, staging, 0, None);
            }
            Self::Readback {
                pipeline,
                layout,
                target,
            } => {
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ci sample bind group"),
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(data),
                    }],
                });
                let view = target.create_view(&wgpu::TextureViewDescriptor::default());
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("ci sample pass"),
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
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }
                encoder.copy_texture_to_buffer(
                    target.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: staging,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(Self::SAMPLE_BPR),
                            rows_per_image: None,
                        },
                    },
                    target.size(),
                );
            }
        }
    }

    /// The histogram from `staging`'s mapped contents.
    fn read(&self, bytes: &[u8]) -> Vec<u32> {
        match self {
            Self::Compute { .. } => bytemuck::cast_slice(bytes).to_vec(),
            Self::Readback { .. } => {
                // Same binning as `ci_stats.wgsl`.
                let mut hist = vec![0u32; CI_BINS];
                for px in bytemuck::cast_slice::<u8, [f32; 4]>(bytes) {
                    let [ci, interior, ..] = *px;
                    if interior >= 1.0 {
                        continue;
                    }
                    let x = (1.0 + ci.max(0.0)).log2() * (CI_BINS as f32 / CI_LOG2_MAX);
                    hist[(x.max(0.0) as usize).min(CI_BINS - 1)] += 1;
                }
                hist
            }
        }
    }
}

/// Auto colour scale: the data texture's `ci` histogram ([`CiHistogram`]),
/// read back asynchronously one or two frames later by
/// [`FractalRenderer::take_ci_range`].
struct CiStats {
    histogram: CiHistogram,
    /// The compute pass's histogram buffer (`None` for `Readback`).
    hist: Option<wgpu::Buffer>,
    staging: wgpu::Buffer,
    state: CiStatsState,
    /// The data texture changed since the last histogram was recorded.
    stale: bool,
}

impl CiStats {
    fn new(histogram: CiHistogram, device: &wgpu::Device) -> Self {
        let (hist, staging) = histogram.buffers(device);
        Self {
            histogram,
            hist,
            staging,
            state: CiStatsState::Idle,
            stale: true,
        }
    }

    /// Record the histogram of `data` (`width`×`height`) and its copy into
    /// `staging`. The caller checks the state is `Idle`.
    fn record(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        data: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        self.histogram.record(
            device,
            encoder,
            data,
            width,
            height,
            self.hist.as_ref(),
            &self.staging,
        );
        self.state = CiStatsState::Copied;
        self.stale = false;
    }

    /// Start mapping a histogram copied on an earlier (now submitted) frame.
    fn start_map(&mut self) {
        if let CiStatsState::Copied = self.state {
            let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = Arc::clone(&done);
            self.staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |res| {
                    if res.is_ok() {
                        flag.store(true, std::sync::atomic::Ordering::Release);
                    }
                });
            self.state = CiStatsState::Mapping(done);
        }
    }
}

/// The `ci` range `[lo, hi]` between the [`CI_TAIL`] percentiles of a
/// `ci_stats.wgsl` histogram, or `None` if no pixel escaped.
fn ci_range(hist: &[u32]) -> Option<(f32, f32)> {
    let total: u64 = hist.iter().map(|&n| n as u64).sum();
    if total == 0 {
        return None;
    }
    let bin_ci = |i: usize| {
        let x = (i as f32 + 0.5) * (CI_LOG2_MAX / CI_BINS as f32);
        x.exp2() - 1.0
    };
    let percentile = |p: f64| {
        let target = (p * total as f64).ceil().max(1.0) as u64;
        let mut acc = 0;
        for (i, &n) in hist.iter().enumerate() {
            acc += n as u64;
            if acc >= target {
                return i;
            }
        }
        hist.len() - 1
    };
    Some((
        bin_ci(percentile(CI_TAIL)),
        bin_ci(percentile(1.0 - CI_TAIL)),
    ))
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
    /// The distance-field rebuild of the refined (or 1-spp) data + the
    /// colourise bind group reading it. Allocated on first use, only for
    /// shadow / 3D Complex Multibrot (see [`wants_envelope`]).
    envelope: Option<(Envelope, wgpu::BindGroup)>,
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
    bla_generation: u64,
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

/// How the iteration shader reads the reference orbit and the BLA table
/// (bindings 1 and 3-5 of the iterate group).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum GpuPath {
    /// Storage buffers (`data_storage.wgsl`): WebGPU and every native backend.
    Storage,
    /// Textures (`data_texture.wgsl`), for devices without fragment-stage
    /// storage buffers (WebGL2). Slower: each read is a texel fetch plus
    /// index math, and frexp/ldexp/ctz are emulated with bit tricks.
    Texture,
}

impl GpuPath {
    /// The fastest path `device` supports. Natively, `MANDELBROT_GPU_PATH`
    /// (`storage` / `texture`) overrides it, to check the WebGL2 path
    /// against the default one on the same GPU.
    pub fn for_device(device: &wgpu::Device) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        match std::env::var("MANDELBROT_GPU_PATH").as_deref() {
            Ok("texture") => return Self::Texture,
            Ok("storage") => return Self::Storage,
            Ok(other) => log::warn!("ignoring unknown MANDELBROT_GPU_PATH={other}"),
            Err(_) => {}
        }
        if device.limits().max_storage_buffers_per_shader_stage >= 4 {
            Self::Storage
        } else {
            Self::Texture
        }
    }

    /// `mandelbrot.wgsl` with its shared prefix and this path's data fragment.
    fn mandelbrot_source(self) -> String {
        let data = match self {
            Self::Storage => include_str!("../shaders/data_storage.wgsl"),
            Self::Texture => include_str!("../shaders/data_texture.wgsl"),
        };
        [
            include_str!("../shaders/common.wgsl"),
            include_str!("../shaders/iterate_uniforms.wgsl"),
            include_str!("../shaders/mandelbrot.wgsl"),
            data,
        ]
        .concat()
    }
}

/// Width (texels) of the [`GpuPath::Texture`] data textures: WebGL2's
/// guaranteed minimum texture size. Must match `DATA_TEX_LOG2_W` in
/// `data_texture.wgsl`.
const DATA_TEX_WIDTH: u32 = 2048;

/// One of the arrays the iteration shader reads.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum DataKind {
    /// Reference orbit points (`[f32; 2]`), binding 1.
    Orbit,
    /// Their exponents (`RefOrbit::exps`, `i32`), binding 3.
    Exps,
    /// BLA nodes (`GpuBla`, 48 bytes), binding 4.
    BlaNodes,
    /// BLA metadata (`u32`), binding 5.
    BlaMeta,
}

impl DataKind {
    fn binding(self) -> u32 {
        match self {
            Self::Orbit => 1,
            Self::Exps => 3,
            Self::BlaNodes => 4,
            Self::BlaMeta => 5,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Orbit => "reference orbit",
            Self::Exps => "reference orbit exponents",
            Self::BlaNodes => "bla nodes",
            Self::BlaMeta => "bla meta",
        }
    }

    fn elem_bytes(self) -> usize {
        match self {
            Self::Orbit => std::mem::size_of::<[f32; 2]>(),
            Self::Exps => std::mem::size_of::<i32>(),
            Self::BlaNodes => std::mem::size_of::<super::bla::GpuBla>(),
            Self::BlaMeta => std::mem::size_of::<u32>(),
        }
    }

    /// Texels per element on the texture path (a node is three `Rgba32Uint`).
    fn texels_per_elem(self) -> u32 {
        match self {
            Self::BlaNodes => 3,
            _ => 1,
        }
    }

    fn texture_format(self) -> wgpu::TextureFormat {
        match self {
            Self::Orbit => wgpu::TextureFormat::Rg32Float,
            Self::Exps => wgpu::TextureFormat::R32Sint,
            Self::BlaNodes => wgpu::TextureFormat::Rgba32Uint,
            Self::BlaMeta => wgpu::TextureFormat::R32Uint,
        }
    }

    fn layout_entry(self, path: GpuPath) -> wgpu::BindGroupLayoutEntry {
        let ty = match path {
            GpuPath::Storage => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            GpuPath::Texture => wgpu::BindingType::Texture {
                sample_type: match self {
                    Self::Orbit => wgpu::TextureSampleType::Float { filterable: false },
                    Self::Exps => wgpu::TextureSampleType::Sint,
                    Self::BlaNodes | Self::BlaMeta => wgpu::TextureSampleType::Uint,
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        };
        wgpu::BindGroupLayoutEntry {
            binding: self.binding(),
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        }
    }

    /// Most elements `device` can hold on `path`: a texture is
    /// `DATA_TEX_WIDTH` wide and at most the device's 2D limit tall.
    fn max_elems(self, device: &wgpu::Device, path: GpuPath) -> usize {
        match path {
            GpuPath::Storage => usize::MAX,
            GpuPath::Texture => {
                let rows = device.limits().max_texture_dimension_2d as usize;
                DATA_TEX_WIDTH as usize * rows / self.texels_per_elem() as usize
            }
        }
    }
}

/// GPU copy of one [`DataKind`] array: a storage buffer or a texture.
struct DataArray {
    kind: DataKind,
    store: DataStore,
    /// Elements it holds.
    capacity: usize,
}

enum DataStore {
    Buffer(wgpu::Buffer),
    Texture(wgpu::Texture, wgpu::TextureView),
}

impl DataArray {
    /// Room for at least `capacity` elements (at least one), zeroed. The
    /// caller keeps `capacity` within [`DataKind::max_elems`].
    fn new(device: &wgpu::Device, path: GpuPath, kind: DataKind, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        match path {
            GpuPath::Storage => {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(kind.label()),
                    size: (capacity * kind.elem_bytes()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                Self {
                    kind,
                    store: DataStore::Buffer(buffer),
                    capacity,
                }
            }
            GpuPath::Texture => {
                let tpe = kind.texels_per_elem() as usize;
                let rows = (capacity * tpe).div_ceil(DATA_TEX_WIDTH as usize);
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(kind.label()),
                    size: wgpu::Extent3d {
                        width: DATA_TEX_WIDTH,
                        height: rows as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: kind.texture_format(),
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                Self {
                    kind,
                    store: DataStore::Texture(texture, view),
                    capacity: rows * DATA_TEX_WIDTH as usize / tpe,
                }
            }
        }
    }

    /// Upload `bytes` (whole elements, at most `capacity`) from element 0.
    fn write(&self, queue: &wgpu::Queue, bytes: &[u8]) {
        debug_assert!(bytes.len() <= self.capacity * self.kind.elem_bytes());
        match &self.store {
            DataStore::Buffer(buffer) => queue.write_buffer(buffer, 0, bytes),
            DataStore::Texture(texture, _) => {
                let texel = self.kind.elem_bytes() / self.kind.texels_per_elem() as usize;
                let row_bytes = DATA_TEX_WIDTH as usize * texel;
                let full_rows = bytes.len() / row_bytes;
                let rest = (bytes.len() % row_bytes) / texel;
                // Full rows, then the partial last one.
                let put = |data: &[u8], y: usize, width: usize, rows: usize| {
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d {
                                x: 0,
                                y: y as u32,
                                z: 0,
                            },
                            aspect: wgpu::TextureAspect::All,
                        },
                        data,
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some((width * texel) as u32),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width: width as u32,
                            height: rows as u32,
                            depth_or_array_layers: 1,
                        },
                    );
                };
                if full_rows > 0 {
                    put(
                        &bytes[..full_rows * row_bytes],
                        0,
                        DATA_TEX_WIDTH as usize,
                        full_rows,
                    );
                }
                if rest > 0 {
                    put(&bytes[full_rows * row_bytes..], full_rows, rest, 1);
                }
            }
        }
    }

    fn bind_entry(&self) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding: self.kind.binding(),
            resource: match &self.store {
                DataStore::Buffer(buffer) => buffer.as_entire_binding(),
                DataStore::Texture(_, view) => wgpu::BindingResource::TextureView(view),
            },
        }
    }
}

/// The reference orbit, its exponents and the BLA table, as the iteration
/// shader reads them on `path`.
struct RefData {
    path: GpuPath,
    orbit: DataArray,
    exps: DataArray,
    bla_nodes: DataArray,
    bla_meta: DataArray,
    /// Most orbit points / BLA nodes / meta entries the device can hold.
    max_points: usize,
    max_nodes: usize,
    max_meta: usize,
}

impl RefData {
    /// Arrays with room for `points` orbit points and `nodes` / `meta` BLA
    /// entries, zeroed (the empty BLA table is valid as all zeros: no levels,
    /// no segments).
    fn new(device: &wgpu::Device, path: GpuPath, points: usize, nodes: usize, meta: usize) -> Self {
        let max_points = DataKind::Orbit.max_elems(device, path).min(MAX_REF_POINTS);
        let max_nodes = DataKind::BlaNodes.max_elems(device, path);
        let max_meta = DataKind::BlaMeta.max_elems(device, path);
        let points = points.min(max_points);
        Self {
            path,
            orbit: DataArray::new(device, path, DataKind::Orbit, points),
            exps: DataArray::new(device, path, DataKind::Exps, points),
            bla_nodes: DataArray::new(device, path, DataKind::BlaNodes, nodes.min(max_nodes)),
            bla_meta: DataArray::new(device, path, DataKind::BlaMeta, meta.min(max_meta)),
            max_points,
            max_nodes,
            max_meta,
        }
    }

    /// Exactly `reference` and `bla` (an export's snapshot).
    fn with_contents(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: GpuPath,
        reference: &RefOrbit,
        bla: &BlaTable,
    ) -> Self {
        let mut data = Self::new(device, path, reference.len(), 0, 0);
        data.write_orbit(queue, reference);
        data.upload_bla(device, queue, bla);
        data
    }

    /// The bind group layout entries of bindings 1 and 3-5.
    fn layout_entries(path: GpuPath) -> [wgpu::BindGroupLayoutEntry; 4] {
        [
            DataKind::Orbit,
            DataKind::Exps,
            DataKind::BlaNodes,
            DataKind::BlaMeta,
        ]
        .map(|k| k.layout_entry(path))
    }

    fn bind_entries(&self) -> [wgpu::BindGroupEntry<'_>; 4] {
        [
            self.orbit.bind_entry(),
            self.exps.bind_entry(),
            self.bla_nodes.bind_entry(),
            self.bla_meta.bind_entry(),
        ]
    }

    /// Orbit points that fit on the device: the shader must never be told
    /// about more (`Uniforms::ref_len`).
    fn points_limit(&self) -> usize {
        self.max_points
    }

    /// Grow the orbit arrays (by powers of two) to hold `needed` points,
    /// without keeping their contents. True if they were reallocated, so
    /// bind groups pointing at them need rebuilding.
    fn ensure_points(&mut self, device: &wgpu::Device, needed: usize) -> bool {
        let needed = needed.min(self.max_points);
        if needed <= self.orbit.capacity {
            return false;
        }
        let capacity = needed.next_power_of_two().min(self.max_points);
        self.orbit = DataArray::new(device, self.path, DataKind::Orbit, capacity);
        self.exps = DataArray::new(device, self.path, DataKind::Exps, capacity);
        true
    }

    /// Upload the orbit's first points (as many as fit; the caller grew the
    /// arrays with `ensure_points`).
    fn write_orbit(&self, queue: &wgpu::Queue, reference: &RefOrbit) {
        let count = reference.len().min(self.orbit.capacity);
        if count > 0 {
            self.orbit
                .write(queue, bytemuck::cast_slice(&reference[..count]));
            self.exps
                .write(queue, bytemuck::cast_slice(&reference.exps[..count]));
        }
    }

    /// Upload `table`, growing the BLA arrays (by powers of two) if needed.
    /// A table too large for the device is replaced by the empty one (no
    /// jumps). True if the arrays were reallocated.
    fn upload_bla(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, table: &BlaTable) -> bool {
        let empty;
        let table = if table.nodes.len() <= self.max_nodes && table.meta.len() <= self.max_meta {
            table
        } else {
            log::warn!(
                "BLA table ({} nodes) too large for this device; skipping BLA",
                table.nodes.len()
            );
            empty = BlaTable::empty();
            &empty
        };
        let grow = table.nodes.len() > self.bla_nodes.capacity
            || table.meta.len() > self.bla_meta.capacity;
        if grow {
            let nodes = table.nodes.len().next_power_of_two().min(self.max_nodes);
            let meta = table.meta.len().max(64).min(self.max_meta);
            self.bla_nodes = DataArray::new(device, self.path, DataKind::BlaNodes, nodes);
            self.bla_meta = DataArray::new(device, self.path, DataKind::BlaMeta, meta);
        }
        self.bla_nodes
            .write(queue, bytemuck::cast_slice(&table.nodes));
        self.bla_meta
            .write(queue, bytemuck::cast_slice(&table.meta));
        grow
    }
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
    /// Reference orbit + BLA table (bindings 1, 3-5), as `GpuPath` reads them.
    data: RefData,
    /// Generation of the BLA table currently in `bla`.
    uploaded_bla_generation: u64,
    lights_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    target_format: wgpu::TextureFormat,
    /// Generation of the reference orbit currently uploaded to `data`.
    uploaded_generation: u64,
    /// Contents of `lights_buffer`, so it's only re-uploaded on change.
    uploaded_lights: Option<[GpuLight; MAX_LIGHT_COUNT]>,

    /// Colourise pass: data texture → colour texture (palette mapping).
    colorize_pipeline: wgpu::RenderPipeline,
    colorize_bind_group_layout: wgpu::BindGroupLayout,
    /// Distance-field passes for shadow / 3D Complex Multibrot (see
    /// [`wants_envelope`]).
    lipschitz: Lipschitz,
    /// Whether the cache's envelope matches the current data texture. Not
    /// implied by iteration: switching shadow → 3D doesn't re-iterate.
    envelope_valid: bool,

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
    /// Auto colour scale's `ci` histogram + readback.
    ci_stats: CiStats,
    /// Measured iterate / refine cost, for sizing their bands.
    timing: PassTiming,
}

impl FractalRenderer {
    fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        self.bind_group = iterate_bind_group(
            device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.lights_buffer,
            &self.data,
        );
    }

    /// How this renderer's iteration shader reads its data.
    pub fn gpu_path(&self) -> GpuPath {
        self.data.path
    }

    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let path = GpuPath::for_device(device);
        log::debug!("iteration data path: {path:?}");
        let shader = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("mandelbrot"),
                    source: wgpu::ShaderSource::Wgsl(path.mandelbrot_source().into()),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fractal uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let data = RefData::new(device, path, INITIAL_REF_POINTS, 1 << 14, 64);

        let lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lights parameters"),
            size: std::mem::size_of::<[GpuLight; MAX_LIGHT_COUNT]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = iterate_bind_group_layout(device, path);
        let bind_group = iterate_bind_group(
            device,
            &bind_group_layout,
            &uniform_buffer,
            &lights_buffer,
            &data,
        );

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
        let colorize_shader = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("colorize"),
                    source: wgpu::ShaderSource::Wgsl(
                        concat!(
                            include_str!("../shaders/common.wgsl"),
                            include_str!("../shaders/iterate_uniforms.wgsl"),
                            include_str!("../shaders/colorize.wgsl"),
                        )
                        .into(),
                    ),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };
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
        let blit_shader = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("blit"),
                    source: wgpu::ShaderSource::Wgsl(
                        concat!(
                            include_str!("../shaders/common.wgsl"),
                            include_str!("../shaders/blit.wgsl"),
                        )
                        .into(),
                    ),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };

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
            data,
            uploaded_bla_generation: u64::MAX,
            lights_buffer,
            bind_group,
            target_format,
            uploaded_generation: u64::MAX,
            uploaded_lights: None,
            colorize_pipeline,
            colorize_bind_group_layout,
            lipschitz: Lipschitz::new(device),
            envelope_valid: false,
            blit_pipeline,
            blit_bind_group_layout,
            blit_sampler,
            cache: None,
            iterated: None,
            colored: None,
            ci_stats: CiStats::new(CiHistogram::new(device, path), device),
            timing: PassTiming::default(),
        }
    }

    /// The on-screen `ci` range from the latest auto-colour histogram, once
    /// its readback has landed (`None` before that, or if nothing escaped).
    pub fn take_ci_range(&mut self, device: &wgpu::Device) -> Option<(f32, f32)> {
        let CiStatsState::Mapping(done) = &self.ci_stats.state else {
            return None;
        };
        let _ = device.poll(wgpu::PollType::Poll);
        if !done.load(std::sync::atomic::Ordering::Acquire) {
            return None;
        }
        let staging = &self.ci_stats.staging;
        let range = {
            let data = staging.slice(..).get_mapped_range().ok()?;
            ci_range(&self.ci_stats.histogram.read(&data))
        };
        staging.unmap();
        self.ci_stats.state = CiStatsState::Idle;
        range
    }

    /// A histogram is on its way back: keep repainting to pick it up.
    pub fn ci_stats_pending(&self) -> bool {
        !matches!(self.ci_stats.state, CiStatsState::Idle)
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
            let envelope = old.envelope.iter().flat_map(|e| &e.0.textures);
            for t in old.textures.iter().chain(envelope) {
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
            colorize_bind_group(
                device,
                &self.colorize_bind_group_layout,
                &self.uniform_buffer,
                &self.lights_buffer,
                data,
            )
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
            envelope: None,
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
    /// texture to march over and `fs_color` has no 3D path. Likewise for
    /// the distance field ([`wants_envelope`]), which needs the whole image.
    /// With `ci_probe`, also what [`ExportRender::ci_range_blocking`] needs
    /// (auto colour scale).
    pub fn export_handles(
        &self,
        device: &wgpu::Device,
        uniforms: &Uniforms,
        ci_probe: bool,
    ) -> ExportHandles {
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
        let chain = uniforms.rendering_mode == 2 || wants_envelope(uniforms);
        let raymarch = chain.then(|| RaymarchHandles {
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
            lipschitz: wants_envelope(uniforms).then(|| self.lipschitz.clone()),
        });
        let probe = ci_probe.then(|| {
            let iterate = raymarch.as_ref().map_or_else(
                || {
                    fullscreen_pipeline(
                        device,
                        "fractal export ci probe pipeline",
                        &self.shader,
                        &self.pipeline_layout,
                        "fs_data",
                        DATA_FORMAT,
                        &constants,
                    )
                },
                |rm| rm.iterate.clone(),
            );
            (iterate, self.ci_stats.histogram.clone())
        });
        ExportHandles {
            pipeline,
            bind_group_layout: self.bind_group_layout.clone(),
            path: self.data.path,
            format: self.target_format,
            raymarch,
            probe,
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
    /// How the pipelines read the reference orbit and BLA table.
    path: GpuPath,
    format: wgpu::TextureFormat,
    /// The two-pass chain, for 3D mode and the distance field only.
    raymarch: Option<RaymarchHandles>,
    /// 1-spp iterate pipeline (`fs_data`) + `ci` histogram, for the auto
    /// colour scale's prepass.
    probe: Option<(wgpu::RenderPipeline, CiHistogram)>,
}

/// The interactive two-pass pipelines, for a 3D or distance-field export.
#[derive(Clone)]
struct RaymarchHandles {
    iterate: wgpu::RenderPipeline,
    refine: wgpu::RenderPipeline,
    colorize: wgpu::RenderPipeline,
    refine_bind_group_layout: wgpu::BindGroupLayout,
    colorize_bind_group_layout: wgpu::BindGroupLayout,
    /// The distance-field passes, when [`wants_envelope`].
    lipschitz: Option<Lipschitz>,
}

/// A 3D (or distance-field shadow) export's own data textures and the passes
/// that fill them: the tiles iterate into `data_view`, then one refine (if
/// AA), the distance-field rebuild (if wanted) and a colourise pass render
/// the finished height field into the export target.
struct RaymarchExport {
    iterate: wgpu::RenderPipeline,
    /// Refine pipeline, output view and input bind group, when AA is on.
    refine: Option<(wgpu::RenderPipeline, wgpu::TextureView, wgpu::BindGroup)>,
    envelope: Option<(Lipschitz, Envelope)>,
    colorize: wgpu::RenderPipeline,
    colorize_bind_group: wgpu::BindGroup,
    data_view: wgpu::TextureView,
}

/// A self-contained render of one export image. It owns its own uniform and
/// reference buffers (a snapshot of the view at export time), so it is unaffected
/// by panning/zooming on the main thread, and can run on a background thread.
/// The image is rendered in horizontal bands (sized by [`BandSizer`]) so
/// progress can be reported as the GPU works through it.
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
    /// Height of the first band, before timings refine it.
    first_band_rows: u32,
    pub swap_rb: bool,
    /// 3D mode or the distance field: tiles fill a data texture instead of
    /// the target.
    raymarch: Option<RaymarchExport>,
    // Only for headless auto colour (`ci_range_blocking`, `set_uniforms`).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    uniform_buffer: wgpu::Buffer,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    uniforms: Uniforms,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    probe: Option<(wgpu::RenderPipeline, CiHistogram)>,
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
        bla: &BlaTable,
        lights: &[Light],
    ) -> Self {
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("export uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let data = RefData::with_contents(device, queue, handles.path, reference, bla);
        let mut uniforms = uniforms;
        uniforms.ref_len = uniforms.ref_len.min(data.points_limit() as u32);
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

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
        let bind_group = iterate_bind_group(
            device,
            &handles.bind_group_layout,
            &uniform_buffer,
            &lights_buffer,
            &data,
        );

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
        // many submissions; more at high iteration counts so no single
        // submission runs long enough to trip a GPU reset (see
        // `WORK_PER_SUBMIT`). Export supersamples every pixel (×3 in shadow
        // mode, which also iterates two neighbours).
        let aa = uniforms.aa_level.max(1);
        let samples = aa * aa * if uniforms.rendering_mode != 0 { 3 } else { 1 };
        let tiles = (height / 128)
            .clamp(8, 64)
            .max(band_count(width, height, uniforms.max_iter, samples))
            .min(height.max(1));

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
            // The distance field reads the refined texture when AA is on,
            // and colourise reads the distance field, else the same texture.
            let refined = refine.as_ref().map_or(&data_view, |(_, v, _)| v);
            let envelope = rm.lipschitz.as_ref().map(|lp| {
                let env = Envelope::new(device, lp, refined, width, height);
                (lp.clone(), env)
            });
            let colorize_input = envelope.as_ref().map_or(refined, |(_, e)| e.output());
            let colorize_bind_group = colorize_bind_group(
                device,
                &rm.colorize_bind_group_layout,
                &uniform_buffer,
                &lights_buffer,
                colorize_input,
            );
            RaymarchExport {
                iterate: rm.iterate.clone(),
                refine,
                envelope,
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
            first_band_rows: height.div_ceil(tiles),
            swap_rb,
            raymarch,
            uniform_buffer,
            uniforms,
            probe: handles.probe.clone(),
        }
    }

    /// Replace the uploaded uniforms (e.g. with a refitted colour scale).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_uniforms(&mut self, queue: &wgpu::Queue, uniforms: Uniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        self.uniforms = uniforms;
    }

    /// Auto colour scale for an export: iterate a downscaled 1-spp prepass,
    /// histogram its `ci` and return the range to fit (`None` if nothing
    /// escaped). Needs handles built with `ci_probe`. Blocks on the GPU.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn ci_range_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<(f32, f32)> {
        ci_range(&self.probe_histogram_blocking(device, queue)?)
    }

    /// Upper bound on the step at which the last pixel escaped, from the same
    /// downscaled prepass as [`ci_range_blocking`](Self::ci_range_blocking)
    /// (`ci` = sqrt of the smooth count; `Some(0)` if nothing escaped). A
    /// pixel that ran off the end of a cut reference counts as escaping
    /// there (or reads as a huge `ci`), so this also says whether a longer
    /// orbit could change the image. `None` if the readback failed.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn max_escape_blocking(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<f64> {
        let hist = self.probe_histogram_blocking(device, queue)?;
        let Some(top) = hist.iter().rposition(|&n| n != 0) else {
            return Some(0.0);
        };
        if top + 1 >= CI_BINS {
            return Some(f64::INFINITY); // clamped: off the histogram's range
        }
        let ci = ((top + 1) as f64 * (CI_LOG2_MAX as f64 / CI_BINS as f64)).exp2() - 1.0;
        Some(ci * ci + 1.0)
    }

    /// Iterate the downscaled 1-spp prepass and read back its `ci`
    /// histogram (`ci_stats.wgsl`).
    #[cfg(not(target_arch = "wasm32"))]
    fn probe_histogram_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<Vec<u32>> {
        let (iterate, histogram) = self
            .probe
            .as_ref()
            .expect("export handles built without ci_probe");
        // The uniforms carry no resolution, so a smaller image of the same view
        // just samples it more coarsely; plenty for percentiles.
        let s = (CI_PROBE_MAX_DIM as f64 / self.width.max(self.height) as f64).min(1.0);
        let width = ((self.width as f64 * s).round() as u32).max(1);
        let height = ((self.height as f64 * s).round() as u32).max(1);
        let data = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("export ci probe"),
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
            .create_view(&wgpu::TextureViewDescriptor::default());
        let (hist, staging) = histogram.buffers(device);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("export ci probe"),
        });
        let samples = if self.uniforms.rendering_mode != 0 {
            3
        } else {
            1
        };
        // Timed bands, as the export's: the worst-case bound alone cut deep
        // probes (~1M `max_iter`) into one-row submissions, costlier than the
        // full-size render.
        let mut sizer = BandSizer::new(height.div_ceil(band_count(
            width,
            height,
            self.uniforms.max_iter,
            samples,
        )));
        let mut y0 = 0;
        while y0 < height {
            let y1 = (y0 + sizer.rows()).min(height);
            let start = std::time::Instant::now();
            let mut band_encoder = device.create_command_encoder(&Default::default());
            band_pass(
                &mut band_encoder,
                "export ci probe pass",
                &data,
                iterate,
                &[&self.bind_group],
                Some((width, y0, y1)),
                y0 == 0,
            );
            queue.submit([band_encoder.finish()]);
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            sizer.observe(y1 - y0, start.elapsed().as_secs_f64());
            y0 = y1;
        }
        histogram.record(
            device,
            &mut encoder,
            &data,
            width,
            height,
            hist.as_ref(),
            &staging,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv().ok()?.ok()?;
        let hist = histogram.read(&staging.slice(..).get_mapped_range().ok()?);
        staging.unmap();
        Some(hist)
    }

    /// Band sizing for rendering this export, starting at the worst-case
    /// height; feed it each band's GPU time ([`BandSizer::observe`]).
    pub fn band_sizer(&self) -> BandSizer {
        BandSizer::new(self.first_band_rows)
    }

    /// Render rows `[y0, y1)` into the export texture and submit them. The
    /// band at row 0 clears the whole attachment; later ones preserve earlier
    /// ones. In 3D mode (or with the distance field) the bands iterate into
    /// the data texture instead, and the one reaching the bottom also runs the
    /// whole-image refine, distance-field and colourise passes.
    pub fn render_band(&self, device: &wgpu::Device, queue: &wgpu::Queue, y0: u32, y1: u32) {
        let y1 = y1.min(self.height);
        if y1 <= y0 {
            return;
        }
        let load = if y0 == 0 {
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
            if let Some((lp, env)) = &rm.envelope {
                env.record(&mut encoder, lp);
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

/// Render every band of `er`, blocking on the GPU after each one to time it
/// (see [`BandSizer`]) and report the fraction of rows done.
#[cfg(not(target_arch = "wasm32"))]
fn render_bands_blocking(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    er: &ExportRender,
    mut on_progress: impl FnMut(f32),
) {
    let mut sizer = er.band_sizer();
    let mut y0 = 0;
    while y0 < er.height {
        let y1 = (y0 + sizer.rows()).min(er.height);
        let start = std::time::Instant::now();
        er.render_band(device, queue, y0, y1);
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        sizer.observe(y1 - y0, start.elapsed().as_secs_f64());
        y0 = y1;
        on_progress(y0 as f32 / er.height as f32);
    }
}

/// Render `er` band by band (blocking on the GPU after each band so progress
/// reflects real work), read it back, and encode the result as PNG bytes.
/// Blocks the calling thread throughout, so it's only for native targets:
/// the UI export path runs it on a background thread, headless rendering
/// runs it directly since it has no frame loop to share a thread with.
#[cfg(not(target_arch = "wasm32"))]
pub fn export_to_png_blocking(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    er: &ExportRender,
    compression: png::Compression,
    mut on_progress: impl FnMut(&'static str, f32),
) -> Vec<u8> {
    // Progress budget: rendering fills [0, RENDER_END], encoding the rest.
    const RENDER_END: f32 = 0.6;

    render_bands_blocking(device, queue, er, |done| {
        on_progress("Rendering", RENDER_END * done)
    });
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
        encode_png_with_progress(
            &data,
            er.width,
            er.height,
            er.padded_bpr,
            er.swap_rb,
            compression,
            |f| on_progress("Encoding", RENDER_END + (0.97 - RENDER_END) * f),
        )
    };
    er.readback().unmap();
    png
}

/// Render every band of `er`, read it back, and return a copy of the padded
/// readback bytes (`er.padded_bpr` per row) for [`encode_png`]. Used by the
/// headless animation pipeline, which encodes on other threads.
#[cfg(not(target_arch = "wasm32"))]
pub fn render_readback_blocking(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    er: &ExportRender,
) -> Vec<u8> {
    render_bands_blocking(device, queue, er, |_| {});
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

/// Strip a readback's row padding and convert it to tightly-packed RGBA8
/// (`width * height * 4` bytes, rows top to bottom).
#[cfg(not(target_arch = "wasm32"))]
pub fn unpad_rgba(
    padded: &[u8],
    width: u32,
    height: u32,
    padded_bpr: u32,
    swap_rb: bool,
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
    pixels
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
    let pixels = unpad_rgba(padded, width, height, padded_bpr, swap_rb);

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
    compression: png::Compression,
    mut on_progress: impl FnMut(f32),
) -> Vec<u8> {
    use std::io::Write as _;

    let row = (width * 4) as usize;
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(compression);
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

/// Layout of group 0 of the iterate/refine/export pipelines: uniforms,
/// lights and the [`RefData`] arrays as `path` binds them.
fn iterate_bind_group_layout(device: &wgpu::Device, path: GpuPath) -> wgpu::BindGroupLayout {
    let uniform = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut entries = vec![
        uniform(0),
        // Only read by the export pipeline's shadow branch (`fs_color` with
        // the custom-lights palette); the iterate pipeline (`fs_data`)
        // ignores it, but both pipelines share this layout.
        uniform(2),
    ];
    entries.extend(RefData::layout_entries(path));
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fractal bind group layout"),
        entries: &entries,
    })
}

/// Group 0 of the iterate/refine pipelines: uniforms, lights and the
/// reference orbit / BLA arrays.
fn iterate_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    lights_buffer: &wgpu::Buffer,
    data: &RefData,
) -> wgpu::BindGroup {
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: lights_buffer.as_entire_binding(),
        },
    ];
    entries.extend(data.bind_entries());
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fractal bind group"),
        layout,
        entries: &entries,
    })
}

/// Colourise pass input: the uniforms, the data texture `data` to colour and
/// the lights.
fn colorize_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniforms: &wgpu::Buffer,
    lights: &wgpu::Buffer,
    data: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("colorize bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(data),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: lights.as_entire_binding(),
            },
        ],
    })
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
    band_pass(encoder, label, target, pipeline, bind_groups, None, true);
}

/// [`data_pass`] restricted to `rows` (`[y0, y1)` of a `width`-wide target),
/// clearing the whole attachment first only if `clear`.
fn band_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_groups: &[&wgpu::BindGroup],
    rows: Option<(u32, u32, u32)>,
    clear: bool,
) {
    let load = if clear {
        wgpu::LoadOp::Clear(wgpu::Color::BLACK)
    } else {
        wgpu::LoadOp::Load
    };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
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
    if let Some((width, y0, y1)) = rows {
        // Full-viewport triangle (so the pixel→plane mapping is unchanged),
        // scissored to the band.
        pass.set_scissor_rect(0, y0, width, y1 - y0);
    }
    pass.set_pipeline(pipeline);
    for (i, bg) in bind_groups.iter().enumerate() {
        pass.set_bind_group(i as u32, *bg, &[]);
    }
    pass.draw(0..3, 0..1);
}

/// Worst-case pixel·iteration work allowed in one GPU submission. Drivers
/// reset the GPU when a single draw runs too long (i915 on integrated Intel
/// gives up after ~640 ms when it can't preempt, and a fullscreen draw can't
/// be preempted mid-way), which loses the device. At deep zooms the iteration
/// count reaches millions, so iteration passes are split into row bands, each
/// its own submission, so the driver can schedule other work in between.
/// 2^30 is a few tens of ms at worst on an integrated GPU.
const WORK_PER_SUBMIT: f64 = (1u64 << 30) as f64;

/// Number of row bands a `width`×`height` pass of up to `samples` ×
/// `max_iter` iterations per pixel needs to stay within [`WORK_PER_SUBMIT`]
/// each (at most one band per row).
fn band_count(width: u32, height: u32, max_iter: u32, samples: u32) -> u32 {
    let work = width as f64 * height as f64 * max_iter.max(1) as f64 * samples.max(1) as f64;
    ((work / WORK_PER_SUBMIT).ceil() as u32).clamp(1, height.max(1))
}

/// GPU time aimed for per band once real timings replace [`band_count`]'s
/// bound. That bound assumes every sample runs `max_iter` iterations, but BLA
/// and periodicity detection make most pixels far cheaper: at 1e-200 (~180k
/// iterations) it cut a 1080p AA export into 1080 one-row submissions, too
/// few pixels each to fill the GPU, ~14× slower than needed. Well under the
/// ~640 ms i915 reset.
#[cfg(not(target_arch = "wasm32"))]
const BAND_TARGET_SECS: f64 = 0.05;

/// Band height for a banded render that times its bands as it goes (export).
/// Starts at [`band_count`]'s safe height, then follows the measured cost per
/// row, growing at most 2× per band since density varies across the image.
/// Never below the safe height: that already bounds the worst case, and
/// thinner bands just underfill the GPU (1-row bands made a `--no-bla`
/// 1e-200 export ~4× slower).
#[derive(Clone, Copy, Debug)]
pub struct BandSizer {
    rows: u32,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    min_rows: u32,
}

impl BandSizer {
    /// Starting (and minimum) band height: the worst-case safe one.
    fn new(rows: u32) -> Self {
        let rows = rows.max(1);
        Self {
            rows,
            min_rows: rows,
        }
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Feed back that the last band, of `rows` rows, took `secs` on the GPU.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn observe(&mut self, rows: u32, secs: f64) {
        let ideal = rows as f64 * BAND_TARGET_SECS / secs.max(1e-5);
        self.rows = (ideal as u32).clamp(
            self.min_rows,
            self.rows.saturating_mul(2).max(self.min_rows),
        );
    }
}

/// Measured GPU cost per pixel of the interactive iterate / refine passes,
/// so their bands are sized from the previous render instead of the
/// worst-case bound. Timed natively through `on_submitted_work_done` (an
/// overestimate when the queue was busy, which only errs towards more
/// bands); on the web there's no `Instant` and bands stay worst-case.
#[derive(Default)]
struct PassTiming {
    /// Pipeline the costs were measured with; a different one resets them.
    key: Option<PipelineKey>,
    /// Seconds per pixel of the [iterate, refine] passes.
    #[cfg(not(target_arch = "wasm32"))]
    secs_per_px: [Option<f64>; 2],
    /// Completion times of the last timed render, filled in by callbacks:
    /// [start, iterate done, refine done], and its pixel count.
    #[cfg(not(target_arch = "wasm32"))]
    pending: Arc<std::sync::Mutex<PendingTiming>>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct PendingTiming {
    start: Option<std::time::Instant>,
    iterate: Option<std::time::Instant>,
    refine: Option<std::time::Instant>,
    pixels: f64,
    refined: bool,
}

impl PassTiming {
    /// Take in the last render's timings, if they've all arrived.
    #[cfg(not(target_arch = "wasm32"))]
    fn collect(&mut self) {
        let Ok(mut p) = self.pending.try_lock() else {
            return;
        };
        let (Some(start), Some(it)) = (p.start, p.iterate) else {
            return;
        };
        if p.refined && p.refine.is_none() {
            return;
        }
        let px = p.pixels.max(1.0);
        self.secs_per_px[0] = Some((it - start).as_secs_f64() / px);
        if let Some(r) = p.refine {
            self.secs_per_px[1] = Some((r - it).as_secs_f64() / px);
        }
        *p = PendingTiming::default();
    }

    /// Bands for pass `pass` (0 iterate, 1 refine): from the measured cost
    /// when there is one, never more than the worst-case `bound`.
    fn bands(&self, pass: usize, width: u32, height: u32, bound: u32) -> u32 {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(c) = self.secs_per_px[pass] {
            let secs = c * width as f64 * height as f64;
            return ((secs / BAND_TARGET_SECS).ceil() as u32).clamp(1, bound);
        }
        let _ = (pass, width, height);
        bound
    }
}

/// [`data_pass`] split into `bands` row bands. One band is recorded into
/// `encoder` as usual; more are each submitted on their own right away (so
/// they run before `encoder`, which is submitted later and reads the result).
#[allow(clippy::too_many_arguments)]
fn banded_data_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_groups: &[&wgpu::BindGroup],
    [width, height]: [u32; 2],
    bands: u32,
) {
    if bands <= 1 {
        data_pass(encoder, label, target, pipeline, bind_groups);
        return;
    }
    let band = height.div_ceil(bands);
    for y0 in (0..height).step_by(band as usize) {
        let y1 = (y0 + band).min(height);
        let mut band_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        band_pass(
            &mut band_encoder,
            label,
            target,
            pipeline,
            bind_groups,
            Some((width, y0, y1)),
            y0 == 0,
        );
        queue.submit([band_encoder.finish()]);
    }
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
    /// BLA table for `reference` and this view's `dc` range, and its
    /// generation (bumped on every rebuild).
    pub bla: Arc<BlaTable>,
    pub bla_generation: u64,
    /// Widget size in physical pixels — the cache texture resolution.
    pub size_px: [u32; 2],
    /// Histogram each new iteration for the auto colour scale
    /// ([`FractalRenderer::take_ci_range`]).
    pub auto_color: bool,
}

#[cfg(feature = "gui")]
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
                || r.bla_generation != self.bla_generation
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

        // Last frame's histogram copy has been submitted by now: map it.
        renderer.ci_stats.start_map();
        if iter_dirty {
            renderer.ci_stats.stale = true;
        }
        // Histogram the data texture when auto colour needs one: after this
        // frame's iteration (recorded below), or now if the texture is
        // already current (auto just turned on, or a readback was in flight
        // when it last changed).
        let wants_stats = self.auto_color
            && renderer.ci_stats.stale
            && matches!(renderer.ci_stats.state, CiStatsState::Idle);
        if wants_stats
            && !iter_dirty
            && let Some(cache) = &renderer.cache
        {
            let src = cache.aa.as_ref().map_or(&cache.data_view, |(v, _)| v);
            renderer
                .ci_stats
                .record(device, egui_encoder, src, cache.width, cache.height);
        }

        if !color_dirty {
            return Vec::new(); // cache still valid; paint() just blits it
        }

        if iter_dirty
            && renderer.uploaded_generation != self.generation
            && !self.reference.is_empty()
        {
            if renderer.data.ensure_points(device, self.reference.len()) {
                renderer.rebuild_bind_group(device);
            }
            renderer.data.write_orbit(queue, &self.reference);
            renderer.uploaded_generation = self.generation;
        }
        if iter_dirty && renderer.uploaded_bla_generation != self.bla_generation {
            if renderer.data.upload_bla(device, queue, &self.bla) {
                renderer.rebuild_bind_group(device);
            }
            renderer.uploaded_bla_generation = self.bla_generation;
        }

        // Every pass reads the uniform buffer; refresh it once. The shader
        // must not read past the orbit the device could hold (only ever
        // shorter than the reference on small WebGL2 devices).
        let mut uniforms = self.uniforms;
        uniforms.ref_len = uniforms.ref_len.min(renderer.data.points_limit() as u32);
        queue.write_buffer(&renderer.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
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
        let envelope = wants_envelope(&self.uniforms);
        if envelope
            && let Some(cache) = &renderer.cache
            && cache.envelope.is_none()
        {
            let src = cache.aa.as_ref().map_or(&cache.data_view, |(v, _)| v);
            let env = Envelope::new(device, &renderer.lipschitz, src, cache.width, cache.height);
            let bind_group = colorize_bind_group(
                device,
                &renderer.colorize_bind_group_layout,
                &renderer.uniform_buffer,
                &renderer.lights_buffer,
                env.output(),
            );
            if let Some(cache) = &mut renderer.cache {
                cache.envelope = Some((env, bind_group));
            }
        }
        let mut envelope_ran = false;
        let pipelines = &renderer.pipelines[&PipelineKey::from_uniforms(&self.uniforms)];
        if let Some(cache) = &renderer.cache {
            if iter_dirty {
                let max_iter = self.uniforms.max_iter;
                let key = PipelineKey::from_uniforms(&self.uniforms);
                let timing = &mut renderer.timing;
                if timing.key != Some(key) {
                    *timing = PassTiming::default();
                    timing.key = Some(key);
                }
                #[cfg(not(target_arch = "wasm32"))]
                let timed = {
                    timing.collect();
                    // Time this render unless the last one is still in flight.
                    let mut p = timing.pending.lock().unwrap();
                    let idle = p.start.is_none();
                    if idle {
                        p.pixels = width as f64 * height as f64;
                        p.refined = cache.aa.is_some();
                        p.start = Some(std::time::Instant::now());
                    }
                    idle
                };
                // Iteration pass: 1-spp perturbation iterate → data texture.
                // Timed passes always submit their own bands, so the
                // completion callback can follow them.
                let bands = timing.bands(0, width, height, band_count(width, height, max_iter, 1));
                #[cfg(not(target_arch = "wasm32"))]
                let bands = if timed { bands.max(2) } else { bands };
                banded_data_pass(
                    device,
                    queue,
                    egui_encoder,
                    "fractal iterate pass",
                    &cache.data_view,
                    &pipelines.iterate,
                    &[&renderer.bind_group],
                    [width, height],
                    bands,
                );
                #[cfg(not(target_arch = "wasm32"))]
                if timed {
                    let p = Arc::clone(&timing.pending);
                    queue.on_submitted_work_done(move || {
                        p.lock().unwrap().iterate = Some(std::time::Instant::now());
                    });
                }
                if let Some((data_aa_view, _)) = &cache.aa {
                    // Adaptive AA: supersample only the non-smooth pixels.
                    let aa = self.uniforms.aa_level;
                    let bound = band_count(width, height, max_iter, aa * aa);
                    let bands = timing.bands(1, width, height, bound);
                    #[cfg(not(target_arch = "wasm32"))]
                    let bands = if timed { bands.max(2) } else { bands };
                    banded_data_pass(
                        device,
                        queue,
                        egui_encoder,
                        "fractal AA refine pass",
                        data_aa_view,
                        &pipelines.refine,
                        &[&renderer.bind_group, &cache.refine_bind_group],
                        [width, height],
                        bands,
                    );
                    #[cfg(not(target_arch = "wasm32"))]
                    if timed {
                        let p = Arc::clone(&timing.pending);
                        queue.on_submitted_work_done(move || {
                            p.lock().unwrap().refine = Some(std::time::Instant::now());
                        });
                    }
                }
                if wants_stats {
                    let src = cache.aa.as_ref().map_or(&cache.data_view, |(v, _)| v);
                    renderer
                        .ci_stats
                        .record(device, egui_encoder, src, width, height);
                }
            }

            // Shadow / 3D Complex Multibrot: rebuild the DE as a distance field.
            let env = cache.envelope.as_ref().filter(|_| envelope);
            if let Some((env, _)) = env
                && (iter_dirty || !renderer.envelope_valid)
            {
                env.record(egui_encoder, &renderer.lipschitz);
                envelope_ran = true;
            }

            // Colourise pass: data texture → colour texture.
            let colorize_bind_group = match env {
                Some((_, bg)) => bg,
                None => cache
                    .aa
                    .as_ref()
                    .map_or(&cache.colorize_bind_group, |(_, bg)| bg),
            };
            data_pass(
                egui_encoder,
                "fractal colorize pass",
                &cache.color_view,
                &renderer.colorize_pipeline,
                &[colorize_bind_group],
            );
        }

        renderer.envelope_valid = envelope_ran || (renderer.envelope_valid && !iter_dirty);
        if iter_dirty {
            renderer.iterated = Some(IterState {
                uniforms: self.uniforms,
                generation: self.generation,
                bla_generation: self.bla_generation,
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
