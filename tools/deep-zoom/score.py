#!/usr/bin/env python3
"""Score a deep zoom by rendering thumbnails along it.

One `--headless` animation call streams the whole zoom (full set down to
the target, evenly spaced in log-depth) as raw RGBA thumbnails. Per frame:

- detail: mean gradient magnitude at half resolution, so flat bands and
  empty voids score low and aliasing noise doesn't score high;
- a Fourier-Mellin descriptor (|FFT| resampled to log-polar, then |FFT| of
  that), invariant to translation, rotation and scale. A Misiurewicz spiral
  looks the same at every turn, so its frames all get the same descriptor.

A frame's novelty is its mean descriptor distance to the 3 frames before
it. boring = fraction of frames with novelty below BORING (at the default
32 frames, spiral zooms sit around 0.17, morphing chains around 0.28). score = mean detail x
(1 - boring) x (1 - empty fraction).

Usage:
  score.py --view=RE,IM,HALF_HEIGHT [--frames 32] [--debug] [--sheet out.png]

Needs numpy and a release build (cargo build --release).
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY = os.path.join(ROOT, "target", "release", "mandelbrot")
START_VIEW = "-0.75,0,1.5"

BORING = 0.21  # novelty below which a frame repeats the ones before it
LAGS = (1, 2, 3)
EMPTY = 0.02  # detail below which a frame counts as empty


def render_path(view, frames=32, width=192, height=108, start=START_VIEW):
    """Frames from the full set down to `view` (RE,IM,HALF_HEIGHT), shape (n, h, w, 3)."""
    view = view.removeprefix("--view=")
    cmd = [BINARY, "--headless", f"--view={start}", f"--to-view={view}",
           "--frames", str(frames), "--linear", "--width", str(width),
           "--height", str(height), "--export-path", "-"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         check=True).stdout
    rgba = np.frombuffer(out, np.uint8).reshape(frames, height, width, 4)
    return rgba[..., :3]


def edges(frame):
    """RGB gradient magnitude: filaments and band slopes, whatever the palette phase.

    The cosine palettes offset the channels' phases, so the three channel
    gradients never vanish together the way one brightness gradient does.
    """
    f = frame.astype(np.float32) / 255.0
    gy, gx = np.gradient(f, axis=(0, 1))
    return np.sqrt((gx * gx + gy * gy).sum(axis=2))


def detail(frame):
    """Edge strength after a 2x2 box average.

    Thumbnails are rendered without AA, so a deep frame's sub-pixel filaments
    alias into noise. The average smooths that noise away but keeps real
    filaments, so a noisy frame doesn't win on detail.
    """
    h, w = frame.shape[0] // 2 * 2, frame.shape[1] // 2 * 2
    f = frame[:h, :w].astype(np.float32)
    f = (f[0::2, 0::2] + f[1::2, 0::2] + f[0::2, 1::2] + f[1::2, 1::2]) / 4
    return float(edges(f.astype(np.uint8)).mean())


def structure(e):
    """Rank-normalised edge map: which pixels are filaments, not how bright they are."""
    r = np.empty(e.size, np.float32)
    r[np.argsort(e, axis=None, kind="stable")] = np.linspace(0, 1, e.size)
    return r.reshape(e.shape)


def _log_polar_grid(h, w, n_r=32, n_a=64):
    cy, cx = h / 2, w / 2
    r = np.exp(np.linspace(np.log(2), np.log(min(cy, cx) - 1), n_r))
    a = np.linspace(0, np.pi, n_a, endpoint=False)  # |FFT| is point-symmetric
    ys = (cy + r[:, None] * np.sin(a)[None, :]).astype(int)
    xs = (cx + r[:, None] * np.cos(a)[None, :]).astype(int)
    return ys, xs


def descriptor(g):
    """Translation-, rotation- and scale-invariant signature of a frame."""
    g = (g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    ys, xs = _log_polar_grid(*g.shape)
    lp = np.log1p(mag[ys, xs])
    lp -= lp.mean()
    # Rotation and scale are shifts along the log-polar axes: drop them.
    d = np.abs(np.fft.fft2(lp))[:8, :16].ravel()
    return d / (np.linalg.norm(d) + 1e-12)


@dataclass
class PathScore:
    score: float
    detail: float
    boring: float
    empty: float
    per_frame: list  # (detail, novelty)


def score_frames(frames):
    maps = [edges(f) for f in frames]
    det = np.array([detail(f) for f in frames])
    desc = [descriptor(structure(e)) for e in maps]
    novelty = np.array([
        np.mean([np.linalg.norm(desc[i] - desc[i - lag]) for lag in LAGS])
        for i in range(max(LAGS), len(frames))
    ])
    # The last frames close in on the minibrot, which is meant to look alike.
    boring = float((novelty[:-2] < BORING).mean())
    empty = float((det < EMPTY).mean())
    per_frame = list(zip(det.tolist(), [0.0] * max(LAGS) + novelty.tolist()))
    return PathScore(float(det.mean()) * (1 - boring) * (1 - empty),
                     float(det.mean()), boring, empty, per_frame)


def save_sheet(frames, path, picks=(0.25, 0.5, 0.75, 1.0), columns=4):
    """A grid of the frames at the given fractions of the zoom, row by row."""
    from PIL import Image  # optional: only needed for sheets

    idx = [min(len(frames) - 1, round(p * (len(frames) - 1))) for p in picks]
    rows = [np.concatenate([frames[i] for i in idx[r:r + columns]], axis=1)
            for r in range(0, len(idx), columns)]
    Image.fromarray(np.ascontiguousarray(np.concatenate(rows, axis=0))).save(path)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--view", required=True, help="RE,IM,HALF_HEIGHT of the zoom target")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--debug", action="store_true", help="print per-frame metrics")
    ap.add_argument("--sheet", help="write a contact sheet PNG (needs Pillow)")
    a = ap.parse_args(argv)

    frames = render_path(a.view, a.frames)
    s = score_frames(frames)
    if a.debug:
        for i, (d, nov) in enumerate(s.per_frame):
            print(f"{i:3} detail={d:.3f} novelty={nov:.3f}")
    print(f"score={s.score:.4f} detail={s.detail:.3f} boring={s.boring:.2f} empty={s.empty:.2f}")
    if a.sheet:
        save_sheet(frames, a.sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
