# Fractal Explorer

A fast, interactive deep-zoom fractal explorer — Mandelbrot and Julia sets —
built with **Rust + wgpu + egui + WGSL**. It zooms far past the ~10¹³× limit of
plain `f64` using **perturbation theory**: one high-precision reference orbit is
computed on the CPU (arbitrary precision via `rug` natively, `malachite-float`
on the web), and every pixel is
rendered on the GPU as a cheap `f32` delta from it, with **rebasing** to avoid
glitches. Runs natively (Vulkan/Metal/DX12) and in the browser (WebGPU).

The `f32` GPU tier reaches roughly **10³⁰× magnification** with sharp detail.

## Features

- Mandelbrot and Julia sets (with Julia-constant editor + presets)
- Smooth continuous coloring, `sqrt`-compressed to stay clean at deep zoom
- Several palettes
- Drag to pan, scroll to zoom toward the cursor
- Arbitrary-precision center; per-view reference orbit computed on a background
  thread (native) so the UI stays responsive
- Reference reuse: small pans/zooms reuse the current reference (no recompute)
- PNG export at up to 4× the on-screen resolution
- Shareable/bookmarkable deep-zoom links (`#…` URL fragment, full precision)

## Build & run — native

```sh
cargo run --release
```

Native builds use `rug` (GMP/MPFR) for the reference orbit, which needs a C
toolchain and `m4` (on Windows, MSYS2).

## Build & run — web (WebGPU)

Requires the `wasm32-unknown-unknown` target and `wasm-bindgen-cli` (matching the
`wasm-bindgen` crate version, currently 0.2.x):

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.128   # once

./build-web.sh                       # outputs ./dist (index.html, .js, .wasm)
                                     # (builds with --features wasm: pure-Rust big floats)
python3 -m http.server -d dist 8080  # serve over http
```

Then open <http://localhost:8080> in a **WebGPU-capable browser** (recent
Chrome/Edge, Firefox, or Safari 26+). WebGPU needs a secure context; `localhost`
qualifies. Deploy by serving the `dist/` directory as static files.

> Note: `trunk` also works in principle, but on some systems its `libdeflate-sys`
> C dependency fails to compile; `build-web.sh` uses `wasm-bindgen-cli` directly
> to avoid that.

## Controls

- **Drag** — pan
- **Scroll** — zoom toward the cursor
- **iterations** — raise this as you zoom deeper (deep boundary pixels need many
  more iterations; too few shows solid black)
- **Copy link** — copies a URL that restores the exact view
- **Export PNG** — saves `fractal-<timestamp>.png` (native, in the working dir)
  or downloads it (web)

## How it works

- `src/view.rs` — view state. Center is arbitrary precision (`Big`, from
  `src/bignum/`); the pixel
  scale stays `f64` (even at 10³⁰× it is ~10⁻³³, within `f64` range).
- `src/fractal/reference.rs` — high-precision reference orbit `Z_{n+1}=Z_n²+C`.
- `src/shaders/mandelbrot.wgsl` — per-pixel perturbation `e_{n+1}=2·Z_n·e_n+e_n²+δc`
  with Zhuoran rebasing (`e ← z − Z₀` when the true value drops below the delta),
  a `dc_offset` so a reused/slightly-stale reference still maps correctly, and
  smooth coloring.
- `src/fractal/renderer.rs` — wgpu pipeline, storage buffer for the orbit, egui
  paint callback, and offscreen render-to-PNG.
- `src/worker.rs` — native background thread for the reference orbit (coalesces
  bursts of requests). The web build computes it inline.
- `src/fractal/share.rs` — URL-fragment encode/decode.

### Limits & possible extensions

The `f32` tier degrades past ~10³⁰×. Natural next steps (scaffolding is in
place): an emulated **double-float** GPU tier (~10⁶⁰×), **floatexp** rescaling
and **BLA** iteration-skipping for near-unlimited depth, and moving the web
reference computation to a Web Worker.

## Tests

```sh
cargo test
cargo test --features wasm   # same suite on the web build's big-float backend
```

Covers the reference orbit (vs. a naive `f64` iteration, Mandelbrot and Julia)
and share-link round-tripping.

## Debug/testing env vars (native)

- `MANDEL_VIEW="re,im,half_height[,iterations]"` — start at a specific view
- `MANDEL_JULIA="cre,cim"` — start in Julia mode with constant `c`
- `MANDEL_SHARE="<fragment>"` — restore a share fragment
- `MANDEL_EXPORT=1` (+ optional `MANDEL_EXPORT_PATH=out.png`) — export on the
  first frame, for scripted captures
