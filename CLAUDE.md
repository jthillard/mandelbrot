# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A deep-zoom fractal explorer (Rust + wgpu + egui + WGSL). It zooms past the
~10¹³× limit of plain `f64` using **perturbation theory**: one high-precision
reference orbit is computed on the CPU (arbitrary precision via `dashu-float`),
and every pixel is rendered on the GPU as a cheap `f32` delta from it, with
rebasing to avoid glitches. The `f32` GPU tier reaches roughly 10³⁰×. Runs
natively (Vulkan/Metal/DX12) and in the browser (WebGPU only — WebGL2 can't do
storage buffers, which the fragment shader needs for the reference orbit).

## Commands

```sh
cargo run --release          # native, run (release matters: fractal math is hot)
cargo test                   # reference-orbit math, share-link round-trip, WGSL validation
cargo test --test shader_valid   # just the WGSL parse/validate tests (naga, no GPU needed)
cargo clippy
cargo fmt                    # rustfmt.toml just pins edition = "2024"
```

Web build (WebGPU):

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.128   # must match the wasm-bindgen crate version
./build-web.sh                                     # -> ./dist
python3 -m http.server -d dist 8080
```

Native CLI flags (`src/cli.rs`, applied in `FractalApp::apply_cli`): `--kind`,
`--power`, `--julia re,im`, `--phoenix-p re,im`, `--lambda-l re,im`,
`--palette`, `--share <fragment>`,
`--view re,im,half_height[,iterations]`, `--de`, `--buddhabrot`,
`--buddha-palette`. `--headless` (`src/headless.rs`) skips the window
entirely: it builds the same view from the other flags, creates its own
offscreen wgpu device, and renders straight to a PNG (`--width`/`--height`,
default 1920×1080, `--export-path out.png`) without needing a GPU-backed
window/event loop. Not yet supported with `--buddhabrot`. Run
`mandelbrot --help` for the full list.

`--headless` also has an animation mode, for feeding into `ffmpeg`: add
`--to-view re,im,half_height[,iterations]` (or `--to-share <fragment>`, which
only pulls position/zoom/iterations out of the link) alongside a start view
(`--view`/`--share`/`--kind`/`--julia`), plus `--frames N` or
`--fps`/`--duration`. `--export-path` then names an output *directory* of
`frame-00001.png`, `frame-00002.png`, ... instead of a single file. Only the
camera (center + half-height) is animated — kind, colors, and per-kind
constants stay fixed at whatever the start flags set. `view::interpolate_view`
does the interpolation: half-height geometrically (log-linear, since zoom
spans many decades), center linearly through the complex plane at full
`Big` precision; `--linear` swaps the default smoothstep easing for constant
pacing. Iteration count auto-scales with zoom depth per frame (same
`auto_iteration_count` the interactive app uses while zooming), overriding
any iteration count from `--view`/`--share`/`--to-view`/`--to-share`.

There's no GPU in most sandboxes: `cargo check`/`cargo test --test shader_valid`
are the fast, headless way to validate a change. `cargo test` also runs but
doesn't touch the GPU — the reference-orbit tests are pure CPU math (see
below), and `shader_valid` parses/validates WGSL with `naga` statically instead
of creating a pipeline.

## Architecture

### The perturbation pipeline (the core mechanism, spans several files)

For a pixel at parameter `c = C_ref + dc`, its orbit is written as
`y_n = X_n + e_n`, where `X_n` is the (shared, high-precision) reference orbit
and `e_n` is a small `f32` delta. Whenever `|y_n| < |e_n|` (or the reference
runs out), rebase: `e ← y_n − X_0`, restart the reference index at 0. This is
what makes deep zoom cheap — one expensive high-precision orbit, then every
pixel is a handful of `f32` complex multiplies.

- `src/view.rs` — `ViewState`; center is arbitrary-precision `FBig` (`Big`
  type alias), pixel scale stays `f64` (still in-range at 10³⁰×). Precision
  (bits) scales with zoom depth (`precision_for`).
- `src/fractal/kind.rs` — the `FractalKind` enum (Mandelbrot, Burning Ship,
  Tricorn, Multibrot, Celtic, Perpendicular, Buffalo, Phoenix, Lambda,
  Complex Multibrot) plus everything that only needs to switch on it:
  `label`/`description`/`formula` (UI text), `share_tag`/`from_share_tag`
  (share-link encoding), `default_set_view` (per-kind starting view), and the
  `ALL` array used to enumerate every kind.
- `src/fractal/reference.rs` — `compute_reference`/`compute_set_reference`:
  iterate the chosen formula at high precision on the CPU, emitting `Z_n` as
  `f32` pairs — that's the reference orbit the GPU perturbs from. At
  precision ≤ `F64_MAX_PRECISION` (80 bits, i.e. shallow views) it takes a
  plain-`f64` fast path (`compute_reference_f64`), so each kind's formula
  exists twice in this file (f64 + `FBig`) and both must stay in sync;
  `f64_fast_path_matches_big` checks they agree. Requests are made with 1.5×
  iteration headroom (`reference_iterations` in `app.rs`), so auto-iterations
  creeping up during a zoom doesn't recompute the orbit every frame.
- `src/shaders/*.wgsl` — none of these are standalone WGSL modules; WGSL has
  no `#include`, so each is compiled by concatenating plain-text fragments
  with `concat!`/`include_str!` at the `create_shader_module` call site (see
  `renderer.rs`, `buddhabrot.rs`, and `tests/shader_valid.rs`, which must
  concatenate the same pieces to validate what actually gets built).
  `common.wgsl` (fullscreen-triangle vertex helper, `cmul`/`cpow`, `KIND_*`
  constants) is prepended to every shader. `iterate_uniforms.wgsl` (the
  perturbation-pipeline `Uniforms` struct + `palette()`) is additionally
  prepended to `mandelbrot.wgsl` and `colorize.wgsl`, which share that layout.
  Because there's no namespacing, a definition must live in exactly one file
  among those concatenated together for a given shader — don't redefine a
  `common.wgsl`/`iterate_uniforms.wgsl` symbol locally.
- `src/shaders/mandelbrot.wgsl` — the perturbation fragment shader. It is
  **specialized per pipeline** through WGSL `override` constants (`KIND`,
  `IS_JULIA`, `DE`), so the per-iteration kind/Julia/DE branches fold away at
  pipeline creation. Read those constants in the shader, never `u.kind` /
  `u.is_julia` / `u.de_coloring` (they're still uploaded for layout reasons).
  `renderer.rs` builds one pipeline set per `PipelineKey` lazily on first
  use, and `tests/shader_valid.rs` compiles every kind × Julia × DE variant to
  SPIR-V. So a new kind needs no pipeline-list change, only its `KIND_*`
  constant. `buddhabrot.wgsl` does the same with its own `override KIND`.
  Interior pixels exit early through **periodicity detection**. It uses
  Brent-style checkpoints plus two guards: the cycle's multiplier must be
  clearly attracting (`PERIOD_MAX_MULT2`), and the contracting return must
  repeat in `PERIOD_CONFIRMATIONS` consecutive windows. Both guards are
  needed: without them, exterior pixels at cusps and minibrot edges turned
  black. Retune them only against f64 ground truth on such views. Phoenix is
  excluded (two-term map).
  `advance_delta(z, e)` is the per-kind delta step (`z` = reference point,
  `e` = current delta); the caller adds `step_add` (= `dc`) afterward — this
  relies on `c` being additive in every current kind's formula (a kind where
  it isn't, e.g. a rational map with `c` in a denominator, would need its own
  step function that consumes `dc` internally instead, plus extra per-step
  reference data since the orbit point alone wouldn't be enough to recover an
  exact delta). `fprime(z)` is the derivative used for distance-estimation
  (DE) shading; exact for holomorphic kinds, an approximation (`~2Z`) for the
  abs-based ones. A `KIND_*` constant (from `common.wgsl`) must match the
  matching `FractalKind` variant's discriminant exactly.
- `src/fractal/renderer.rs` — `FractalRenderer` (wgpu pipelines, uniform +
  storage buffers, bind groups), `Uniforms` (repr(C) layout that must match
  the WGSL `Uniforms` struct field-for-field, including padding; it includes
  CPU-precomputed data: `cm_coef`, the Complex Multibrot binomial
  coefficients from `app.rs::complex_binomials`, and `light_count` for the
  packed `GpuLight` buffer from `lights.rs::gpu_lights`), and
  `FractalCallback` (the `egui_wgpu::CallbackTrait` impl: `prepare()` uploads
  changed buffers and decides whether to re-run the iterate pass, the cheap
  colourise pass, or just blit the cached texture). Also `ExportRender`, a
  self-contained tiled renderer used for PNG export off the UI thread.
- `src/worker.rs` — native background thread for reference-orbit computation
  (coalesces bursts of requests so a fast drag doesn't compute every
  intermediate view). The wasm32 build computes inline instead (see the
  `#[cfg(target_arch = "wasm32")]` branch in `app.rs::ensure_reference`) —
  **any signature change to `compute_reference`/`compute_set_reference` or
  `RefRequest`/`RefResult` must be applied to both call sites.**
- `src/app.rs` — `FractalApp` (the egui app + all UI). Key methods:
  `should_request`/`ensure_reference` (decide when the reference is stale and
  dispatch/collect it), `make_uniforms` (assemble the per-frame `Uniforms`),
  `tick_animations` (drives the "morph c/p/λ" and auto-zoom animations),
  `default_view_for` (wraps `FractalKind::default_set_view`, adding the
  kind-independent Julia case). `JULIA_PRESETS` and `SET_PRESETS` are sized as
  `[T; FractalKind::<last variant> as usize + 1]` — adding a new `FractalKind`
  means bumping both (and adding an empty `&[]` slot to each if the kind has
  none), plus adding it to `FractalKind::ALL` in `kind.rs`.
- `src/fractal/share.rs` — `ShareState`: encodes the full view (mode, kind,
  full-precision decimal center, zoom, iterations, per-kind constants,
  coloring) as a `#`-fragment URL for bookmarking/sharing deep-zoom locations.

### Adding a new `FractalKind`

Touches, in order: `kind.rs` (enum variant + `ALL` slot + `label`/
`description`/`formula`/`share_tag`/`from_share_tag`/`default_set_view`
arms), `reference.rs` (CPU iteration formula arm, and a test comparing
against a naive `f64` iteration), `common.wgsl` (matching `KIND_*` const),
`mandelbrot.wgsl` (matching `advance_delta`/`fprime` arms), `buddhabrot.wgsl`
(matching arm in `advance()`, if the kind makes sense as a Buddhabrot),
`renderer.rs` `Uniforms` (only if the kind needs a new per-kind constant,
e.g. Phoenix's `phoenix_p`), `app.rs` (`JULIA_PRESETS`/`SET_PRESETS` slot,
and optionally a UI control for its constant + an animation toggle,
following the Phoenix/Lambda pattern). If `c` doesn't enter the formula
additively (e.g. a rational map with `c` in a denominator), the
`advance_delta`/`step_add` split doesn't work — that needs its own step
function plus extra per-step reference data uploaded in a second GPU buffer
alongside the orbit.

### Buddhabrot is a separate pipeline

`src/fractal/buddhabrot.rs` + `src/shaders/buddhabrot.wgsl` implement the
Monte-Carlo orbit-density histogram. It does **not** use the perturbation/
reference-orbit machinery: a Buddhabrot sample's orbit scatters across the
whole image rather than staying in one pixel, so it's plain `f32` iteration
from the live view (no deep zoom) via a compute pass that accumulates into a
histogram buffer, tone-mapped by a fragment pass every frame. Its own
`KIND_*` iteration formulas in `advance()` must be kept in sync with
`reference.rs` by hand (there's no shared code path).

### Two-pass render + caching (`renderer.rs`)

The interactive path splits iteration (expensive, perturbation) from
colourising (cheap, palette remap) into separate offscreen textures, so
palette/color-scale/offset tweaks skip re-iteration entirely (`geom_differs`
vs `color_differs` in `renderer.rs` decide which pass reruns). A frame where
neither differs uploads and renders nothing and only blits. So any new
uniform field must go into one of those two functions (or the lights
comparison), or changing it won't redraw.

AA is **adaptive** on the interactive path. `fs_data` always iterates 1
sample per pixel. When AA is on, `fs_refine` reads that texture and runs the
2×2 grid only on pixels whose 4-neighbours differ (interior/exterior edge, or
`ci`/DE beyond `AA_CI_EPS`/`AA_DE_EPS`), copying the rest. Colourise then
reads the refined texture. PNG export (`fs_color`) still supersamples every
pixel.

While the user is actively panning/zooming, the app renders downscaled with
AA off (`INTERACT_DOWNSCALE`) and snaps back to full resolution once input
settles (`INTERACT_SETTLE`).
