#!/usr/bin/env bash
# Render the lossless 4K60 test clips that sweep.sh encodes.
#
# Each clip goes through the same RGBA -> bt709 tv-range 10-bit conversion as
# the Makefile, then into lossless FFV1. The clips are therefore exactly what the
# encoders see, so VMAF against them measures the encoder alone.
#
#   tools/encode/make_clips.sh [--size WxH] [--seconds S] [--force] [clip...]
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out="$here/clips"
size=3840x2160
seconds=4
fps=60
force=0
only=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --size) size=$2; shift 2 ;;
        --seconds) seconds=$2; shift 2 ;;
        --force) force=1; shift ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) only+=("$1"); shift ;;
    esac
done

# Seahorse valley, minibrot at 3.7e-41 (tools/deep-zoom/LOCATIONS.md).
SH_RE=-0.77568376800905379745347652613832924487504096622022
SH_IM=0.13646736829469012473375311880735014411233827361594
# Seahorse valley, minibrot at 3.4e-21 (period 429, `find_deep.py seahorse 10`).
# Same look as the 1e-40 one for this purpose, but ~10x faster to render.
MB_RE=-0.775683767855941232535147681616
MB_IM=0.136467368300490586441979944945

# name | mandelbrot flags. Zoom speeds bracket a typical 60 s dive (~0.7 decade/s).
clips=(
    # Worst case: dense filaments, 1.5 decades/s of scaling motion.
    "filaments|--view=$SH_RE,$SH_IM,1e-8 --to-view=$SH_RE,$SH_IM,3e-15 --linear"
    # Smooth concentric bands closing in on the minibrot: banding/blocking risk.
    "bands|--view=$MB_RE,$MB_IM,1e-18 --to-view=$MB_RE,$MB_IM,1.1e-20 --linear"
    # Embedded Julia set, slow drift: fine static texture, psy/detail retention.
    "julia|--view=$SH_RE,$SH_IM,2e-30 --to-view=$SH_RE,$SH_IM,1e-30 --linear"
)

[[ -x $root/target/release/mandelbrot ]] || cargo build --release --manifest-path "$root/Cargo.toml"
mkdir -p "$out"
w=${size%x*}
h=${size#*x}

for entry in "${clips[@]}"; do
    name=${entry%%|*}
    flags=${entry#*|}
    if [[ ${#only[@]} -gt 0 && ! " ${only[*]} " =~ " $name " ]]; then continue; fi
    dst="$out/$name.mkv"
    if [[ -f $dst && $force = 0 ]]; then echo "skip $name (exists, --force to redo)"; continue; fi
    echo "render $name ($size, ${seconds}s @ ${fps}fps)"
    # shellcheck disable=SC2086
    "$root/target/release/mandelbrot" --headless --antialias $flags \
        --width "$w" --height "$h" --fps "$fps" --duration "$seconds" --export-path - |
        ffmpeg -hide_banner -loglevel error -y \
            -f rawvideo -pix_fmt rgba -s "$size" -framerate "$fps" -i - \
            -vf "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int+bitexact,format=yuv420p10le" \
            -c:v ffv1 -level 3 -slices 16 -g 1 \
            -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
            "$dst.tmp.mkv"
    mv "$dst.tmp.mkv" "$dst"
done
