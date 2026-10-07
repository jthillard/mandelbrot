#!/usr/bin/env bash
# Build the WebAssembly bundle into ./dist using wasm-bindgen.
# Serve it over HTTP (e.g. `python3 -m http.server -d dist 8080`) and open in a
# WebGPU-capable browser (recent Chrome/Edge, Firefox, or Safari 26+); the
# `webgl` feature falls back to WebGL2 elsewhere (no Buddhabrot there).
set -euo pipefail

export PATH="$HOME/.cargo/bin:$PATH"
OUT="${1:-dist}"

echo "==> cargo build (wasm32, release)"
cargo build --release --target wasm32-unknown-unknown --features wasm,webgl

echo "==> wasm-bindgen -> $OUT"
mkdir -p "$OUT"
wasm-bindgen \
    --target web \
    --no-typescript \
    --out-dir "$OUT" \
    --out-name mandelbrot \
    target/wasm32-unknown-unknown/release/mandelbrot.wasm

cp index.html "$OUT/index.html"
cp favicon.ico "$OUT/favicon.ico"

echo "==> done: $OUT/ (index.html, mandelbrot.js, mandelbrot_bg.wasm)"
echo "    serve:  python3 -m http.server -d $OUT 8080"
