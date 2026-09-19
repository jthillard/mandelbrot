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

Native debug env vars (see `src/app.rs`, near the top of `FractalApp::new`):
`MANDEL_KIND`, `MANDEL_POWER`, `MANDEL_JULIA="re,im"`, `MANDEL_SHARE="<fragment>"`,
`MANDEL_VIEW="re,im,half_height[,iterations]"`, `MANDEL_DE=1`,
`MANDEL_BUDDHABROT=1`, `MANDEL_EXPORT=1` (+ `MANDEL_EXPORT_PATH=out.png`).

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
- `src/fractal/reference.rs` — `FractalKind` enum (Mandelbrot, Burning Ship,
  Tricorn, Multibrot, Celtic, Perpendicular, Buffalo, Phoenix, Lambda) and
  `compute_reference`/`compute_set_reference`: iterate the chosen formula at
  high precision on the CPU, emitting `Z_n` as `f32` pairs — that's the
  reference orbit the GPU perturbs from.
- `src/shaders/mandelbrot.wgsl` — the perturbation fragment shader.
  `advance_delta(z, e)` is the per-kind delta step (`z` = reference point,
  `e` = current delta); the caller adds `step_add` (= `dc`) afterward — this
  relies on `c` being additive in every current kind's formula (a kind where
  it isn't, e.g. a rational map with `c` in a denominator, would need its own
  step function that consumes `dc` internally instead, plus extra per-step
  reference data since the orbit point alone wouldn't be enough to recover an
  exact delta). `fprime(z)` is the derivative used for distance-estimation
  (DE) shading; exact for holomorphic kinds, an approximation (`~2Z`) for the
  abs-based ones. A `KIND_*` constant here must match the matching
  `FractalKind` variant's discriminant exactly.
- `src/fractal/renderer.rs` — `FractalRenderer` (wgpu pipelines, uniform +
  storage buffers, bind groups), `Uniforms` (repr(C) layout that must match
  the WGSL `Uniforms` struct field-for-field, including padding), and
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
  `default_view_for` (per-kind starting view). `KINDS`, `JULIA_PRESETS`, and
  `SET_PRESETS` are sized as `[T; FractalKind::<last variant> as usize + 1]` —
  adding a new `FractalKind` means bumping all three (and adding an empty
  `&[]` slot to the two preset arrays if the kind has none).
- `src/fractal/share.rs` — `ShareState`: encodes the full view (mode, kind,
  full-precision decimal center, zoom, iterations, per-kind constants,
  coloring) as a `#`-fragment URL for bookmarking/sharing deep-zoom locations.

### Adding a new `FractalKind`

Touches, in order: `reference.rs` (enum variant + CPU iteration formula, and a
test comparing against a naive `f64` iteration), `mandelbrot.wgsl` (matching
`KIND_*` const + `advance_delta`/`fprime` arms), `buddhabrot.wgsl` (matching
arm in `advance()`, if the kind makes sense as a Buddhabrot), `renderer.rs`
`Uniforms` (only if the kind needs a new per-kind constant, e.g. Phoenix's
`phoenix_p`), `share.rs` (encode/decode string tag), `app.rs` (`KINDS` label,
`JULIA_PRESETS`/`SET_PRESETS` slot, `default_view_for` entry, and optionally a
UI control for its constant + an animation toggle, following the
Phoenix/Lambda pattern). If `c` doesn't enter the formula additively (e.g. a
rational map with `c` in a denominator), the `advance_delta`/`step_add` split
doesn't work — that needs its own step function plus extra per-step reference
data uploaded in a second GPU buffer alongside the orbit.

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
vs `color_differs` in `renderer.rs` decide which pass reruns). While the user
is actively panning/zooming, the app renders downscaled with AA off
(`INTERACT_DOWNSCALE`) and snaps back to full resolution once input settles
(`INTERACT_SETTLE`).
