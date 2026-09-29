#!/usr/bin/env bash
# Encode every test clip with every config, then score each encode.
#
#   tools/encode/sweep.sh [--clips a,b] [--keep] [--tag NAME] CONFIG_FILE...
#
# A config line is `name | ffmpeg output args` (blank lines and # comments are
# skipped). The args go between `-i clip.mkv` and the output file, e.g.
#   x265-crf18 | -c:v libx265 -preset medium -crf 18 -x265-params aq-mode=3
# Each row of results/<tag>.tsv has encode fps, bitrate, VMAF 4K (mean, 1st
# percentile, min), VMAF-NEG mean, CAMBI banding (mean, max) and PSNR-Y.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
clipdir="$here/clips"
model=/usr/share/model
export SVT_LOG=1  # errors only
clips=""
keep=0
tag=$(date +%Y%m%d-%H%M%S)
configs=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --clips) clips=$2; shift 2 ;;
        --keep) keep=1; shift ;;
        --tag) tag=$2; shift 2 ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) configs+=("$1"); shift ;;
    esac
done
[[ ${#configs[@]} -gt 0 ]] || { echo "usage: $0 [--clips a,b] [--keep] [--tag NAME] CONFIG_FILE..." >&2; exit 1; }

if [[ -z $clips ]]; then
    clips=$(cd "$clipdir" && ls ./*.mkv | sed 's|^\./||; s|\.mkv$||' | paste -sd,)
fi
[[ -n $clips ]] || { echo "no clips in $clipdir: run make_clips.sh first" >&2; exit 1; }

mkdir -p "$here/results" "$here/work"
tsv="$here/results/$tag.tsv"
[[ -f $tsv ]] || printf 'clip\tconfig\tfps\tmbps\tvmaf\tvmaf_p1\tvmaf_min\tvmaf_neg\tcambi\tcambi_max\tpsnr_y\n' > "$tsv"

IFS=, read -ra clip_list <<< "$clips"
for cfg in "${configs[@]}"; do
    while IFS= read -r line || [[ -n $line ]]; do
        [[ $line =~ ^[[:space:]]*(#|$) ]] && continue
        name=$(sed 's/[[:space:]]*|.*//' <<< "$line")
        args=$(sed 's/^[^|]*|[[:space:]]*//' <<< "$line")
        for clip in "${clip_list[@]}"; do
            ref="$clipdir/$clip.mkv"
            enc="$here/work/$clip--$name.mp4"
            log="$here/work/$clip--$name.json"
            frames=$(ffprobe -v error -count_packets -select_streams v:0 \
                -show_entries stream=nb_read_packets -of csv=p=0 "$ref")
            printf '%-10s %-40s ' "$clip" "$name"

            t0=$(date +%s.%N)
            # shellcheck disable=SC2086
            ffmpeg -hide_banner -loglevel error -y -i "$ref" $args -an "$enc" < /dev/null
            t1=$(date +%s.%N)

            # 4K model for both VMAF variants; CAMBI flags banding, which VMAF
            # barely sees and which is the main risk on smooth fractal gradients.
            # Both streams are retimed by frame index: MKV stores 60 fps in whole
            # ms, and pairing by PTS matched every third frame with its neighbour.
            ffmpeg -hide_banner -loglevel error -i "$enc" -i "$ref" -lavfi \
                "[0:v]settb=1/60,setpts=N[d];[1:v]settb=1/60,setpts=N[r];[d][r]libvmaf=log_fmt=json:log_path=$log:n_threads=$(nproc):n_subsample=2:model='path=$model/vmaf_4k_v0.6.1.json\\:name=vmaf|path=$model/vmaf_4k_v0.6.1neg.json\\:name=neg':feature='name=cambi|name=psnr'" \
                -f null - < /dev/null

            row=$(python3 - "$log" "$enc" "$frames" "$t0" "$t1" <<'PY'
import json, os, sys
log, enc, frames, t0, t1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
fr = json.load(open(log))["frames"]
col = lambda k: sorted(f["metrics"][k] for f in fr)
v = col("vmaf")
p1 = v[max(0, int(0.01 * len(v)) - 1)] if len(v) >= 100 else v[0]
mean = lambda xs: sum(xs) / len(xs)
cambi = col("cambi")
fps = frames / (t1 - t0)
mbps = os.path.getsize(enc) * 8 / (frames / 60) / 1e6
print(f"{fps:.2f}\t{mbps:.1f}\t{mean(v):.2f}\t{p1:.2f}\t{v[0]:.2f}\t{mean(col('neg')):.2f}\t{mean(cambi):.2f}\t{cambi[-1]:.2f}\t{mean(col('psnr_y')):.2f}")
PY
)
            printf '%s\t%s\t%s\n' "$clip" "$name" "$row" >> "$tsv"
            awk -F'\t' '{printf "%6s fps %7s Mb/s  vmaf %s p1 %s min %s neg %s  cambi %s/%s  psnr %s\n",$1,$2,$3,$4,$5,$6,$7,$8,$9}' <<< "$row"
            [[ $keep = 1 ]] || rm -f "$enc"
        done
    done < "$cfg"
done
echo "results: $tsv"
