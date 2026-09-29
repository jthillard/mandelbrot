# Encoder tuning

The x265 and SVT-AV1 settings in the `Makefile` come from this harness. It
encodes a few lossless 4K60 test clips with a list of configs, then scores
every encode against its source with VMAF (4K model) and CAMBI (banding).

```sh
tools/encode/make_clips.sh                       # render clips/ once (~20 min on a GTX 1650)
tools/encode/sweep.sh --tag x265 configs/x265-axes.txt
tools/encode/sweep.sh --clips julia --keep --tag try my-configs.txt
tools/encode/ladder.py results/x265-ladder.tsv   # CRF / Mb/s where each family meets the targets
```

Needs ffmpeg with libx265, libsvtav1 and libvmaf, the libvmaf models in
`/usr/share/model` (`vmaf_4k_v0.6.1.json`, `vmaf_4k_v0.6.1neg.json`), and
python3. `clips/`, `work/` and `results/` are git-ignored.

## Clips

`make_clips.sh` pipes `mandelbrot --headless --antialias` through the same
RGBA → bt709 tv-range 10-bit conversion as the `Makefile` into lossless FFV1,
so the clips are exactly what the encoders see and VMAF measures the encoder
alone. Each is 4 s of 3840×2160 at 60 fps (`--size`/`--seconds` for quick tests).

| clip | content | what it stresses |
|---|---|---|
| `filaments` | seahorse valley, 1e-8 → 3e-15 | dense detail under fast scaling motion |
| `bands` | minibrot at 3.4e-21, 1e-18 → 1.1e-20 | smooth concentric gradients: banding, blocking |
| `julia` | embedded Julia set at 2e-30, slow zoom | low-contrast fine texture on a bright background |

`julia` turned out to be the hardest by far. At a CRF where the other two
already score VMAF 99, both encoders smear its faint speckles, so it decides
every tier's CRF.

## Configs and results

A config line is `name | ffmpeg output args`. `sweep.sh` encodes each clip,
times it, and appends one row per encode to `results/<tag>.tsv`:

| column | meaning |
|---|---|
| `fps` | encode speed, wall clock, all cores |
| `mbps` | bitrate |
| `vmaf`, `vmaf_p1`, `vmaf_min` | VMAF 4K mean, 1st percentile, min |
| `vmaf_neg` | VMAF NEG: like VMAF, but doesn't reward sharpening |
| `cambi`, `cambi_max` | banding, mean and worst frame (0 = none, above ~5 is visible) |
| `psnr_y` | luma PSNR |

VMAF uses every other frame (`n_subsample=2`). With 4 s clips that's 120
frames, so `vmaf_p1` is the worst sampled frame. Both inputs are retimed by
frame index before scoring. The MKV clips store 60 fps timestamps in whole
milliseconds, and pairing by timestamp compared every third frame with its
neighbour.

`ladder.py` groups configs into families by stripping a trailing `-CRF`
(`x265-rect-16` → `x265-rect`). It interpolates, per clip, the CRF and bitrate
where each target is met. A tier has one CRF for all content, so the lowest of
those CRFs binds it.

The configs files are in the order they were run:

- `speed.txt`: preset speed pass (which presets fit the time budget).
- `crf-probe.txt`: coarse CRF ranges, to see where VMAF 95–97 land.
- `x265-axes.txt`, `av1-axes.txt`: one option at a time from a baseline.
- `*-ladder.txt`, `*-ladder2.txt`: the promising combinations at several CRFs,
  to compare them at matched bitrate rather than at a fixed CRF.

## Results (Ryzen 7 3700X, 8C/16T; ffmpeg 9.0, x265 4.3, SVT-AV1 4.2.0)

Tier targets: **delivery** = VMAF mean ≈ 95 and worst frame ≥ 90 on every
clip, smallest file. **Master** = worst frame ≥ 97 on every clip. `julia` binds
every tier. "Interp" rows are interpolated between the two ladder rungs around
that CRF (log bitrate, linear VMAF). Everything else was measured.

| target | CRF | clip | Mb/s | fps | VMAF mean | p1 = min | CAMBI mean / max | |
|---|---|---|---|---|---|---|---|---|
| `x265` | 17 | bands | 279 | 2.4 | 99.93 | 98.81 | 1.53 / 6.00 | interp 16–20 |
| | | filaments | 241 | 3.2 | 99.95 | 98.63 | 2.54 / 8.92 | interp 16–20 |
| | | julia | 68 | 6.1 | 94.81 | 92.75 | 0 / 0 | interp 16–20 |
| `x265-master` | 12 | bands | 466 | 1.8 | 100.00 | 100.00 | 1.58 / 6.19 | |
| | | filaments | 342 | 2.8 | 100.00 | 99.99 | 2.65 / 9.49 | |
| | | julia | 171 | 3.6 | 98.36 | 97.58 | 0 / 0 | |
| `av1` | 32 | bands | 180 | 6.0 | 99.95 | 97.07 | 1.47 / 4.70 | |
| | | filaments | 192 | 7.7 | 99.90 | 97.04 | 2.28 / 7.14 | |
| | | julia | 50 | 7.4 | 95.03 | 93.39 | 0 / 0 | |
| `av1-master` | 17 | bands | 451 | 6.9 | 100.00 | 99.49 | 1.23 / 4.87 | interp 14–20 |
| | | filaments | 380 | 8.6 | 100.00 | 99.50 | 2.40 / 7.67 | interp 14–20 |
| | | julia | 152 | 9.0 | 98.09 | 97.12 | 0 / 0 | interp 14–20 |

SVT-AV1 wins both tiers: the delivery files are 20–35% smaller than
x265's, and it encodes 1.2–2.5× faster. CAMBI max is the same at every CRF on
`filaments` (6–9), so it comes from the content, not the encoder.

What the sweeps showed (details as comments in the `Makefile`):

- **SVT-AV1:** `tune=0` + `enable-variance-boost=1` + `enable-qm=1` is the
  combination that fixes faint texture. With the defaults, `julia` scores a
  worst frame of 85 at CRF 26. The temporal filter (`enable-tf`) was the
  suspect for smeared filaments, but turning it off or weakening it cost
  35–50% more bits for lower VMAF and more banding. `qm-min=0` caps quality.
  `sharpness`, `ac-bias` and `enable-overlays` did nothing measurable.
- **x265:** `rect`, `merange=92` and `rdoq-level=2:psy-rdoq=1` help. SAO hurts.
  `aq-mode=4`, `deblock=-1`, `psy-rd=1`, `me=star`, `rc-lookahead=60` and
  `bframes=8` didn't help at matched bitrate or cost too much speed.
- The worst SVT frames without variance boost sit in the clip's last
  mini-GOP, so part of that p1 is an artefact of a 4 s clip.

**Speed budget.** It was 3–5 fps on an i7-1165G7 (4C/8T). Pinning this machine
to 4C/8T (`taskset -c 0-3,8-11`) as a proxy gives 1.5× (SVT) to 1.8× (x265)
for the full 16 threads, so here the budget is about 5–9 fps. SVT preset 6
fits at both tiers. x265 `medium` with these options doesn't: at the low CRFs
`julia` forces, it runs at 2–6 fps. This SVT-AV1 package also reports
`(debug)`: it's built without `NDEBUG`, but it's optimised and it's what
ffmpeg links.

**Not yet measured:** x265 `-preset fast` with these options (the speed fix
for x265) and `aq-strength=1.2` (`configs/x265-ladder2.txt`). Also a
confirming encode at `av1-master`'s interpolated CRF 17.

## Tips

- Keep the machine idle while sweeping: `fps` is wall clock time.
- A config that changes bitrate at a fixed CRF can't be judged from one row.
  Put it on a ladder and compare with `ladder.py`.
- `--keep` keeps the encodes in `work/`, e.g. to extract frames to compare by eye:
  `ffmpeg -i work/julia--x.mp4 -vf "select=eq(n\,120),crop=480:270:1680:945" -frames:v 1 x.png`.
