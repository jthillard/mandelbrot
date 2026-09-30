#!/usr/bin/env python3
"""Random search for deep zooms that keep changing shape all the way down.

Builds many Julia-morphing chains (morph.py) from different seeds on every
core, renders each zoom's thumbnails on the GPU (score.py), and prints the
best ones with their views and a contact sheet (8 frames) in out/.

Zooms are scored down to the last embedded Julia set, not the minibrot:
the last quarter of the way to any minibrot is rings closing in on it (as
self-similar as a spiral) and looks the same in every chain.

Usage:
  search.py REGION [--depth 300] [--candidates 24] [--seed 1] [--jobs N]

Needs mpmath, numpy, Pillow and a release build (cargo build --release).
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from morph import build_chain, describe, julia_view, minibrot_view
from score import render_path, save_sheet, score_frames

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def chain_job(region, stages, depth, d0, seed, max_period):
    chain = build_chain(region, stages, depth, d0, seed, max_period)
    if not chain:
        return seed, None
    last = chain[-1]
    return seed, {
        "stages": describe(chain),
        "depth": last.depth,
        "period": last.period,
        "julia": julia_view(last),
        "minibrot": minibrot_view(last),
    }


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("region", help="find_deep.py region name, or re,im,k,p")
    ap.add_argument("--stages", type=int, default=14)
    ap.add_argument("--depth", type=float, default=300, help="stop past size 10^-DEPTH")
    ap.add_argument("--d0", type=float, default=4, help="stage 0 distance 10^-D0 from the anchor")
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1, help="first seed; candidates use seed, seed+1, ...")
    ap.add_argument("--max-period", type=int, default=200_000)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    results = []
    with ProcessPoolExecutor(a.jobs) as pool:
        futures = [pool.submit(chain_job, a.region, a.stages, a.depth, a.d0, s, a.max_period)
                   for s in range(a.seed, a.seed + a.candidates)]
        # Chains finish in any order; render each as it arrives (one at a time on the GPU).
        for f in as_completed(futures):
            seed, r = f.result()
            if r is None:
                print(f"seed {seed}: no chain", file=sys.stderr)
                continue
            try:
                frames = render_path(r["julia"])
            except (OSError, subprocess.CalledProcessError) as e:
                print(f"seed {seed}: render failed: {e}", file=sys.stderr)
                continue
            s = score_frames(frames)
            sheet = os.path.join(OUT, f"{a.region.split(',')[0]}-{seed}.png")
            save_sheet(frames, sheet, picks=[i / 7 for i in range(8)], columns=4)
            print(f"seed {seed}: score={s.score:.4f} boring={s.boring:.2f} "
                  f"depth=1e-{r['depth']:.0f} period={r['period']}", file=sys.stderr)
            results.append((s, seed, r, sheet))

    results.sort(key=lambda x: -x[0].score)
    for s, seed, r, sheet in results[:a.top]:
        print(f"\n## seed {seed}: score={s.score:.4f} detail={s.detail:.3f} "
              f"boring={s.boring:.2f} empty={s.empty:.2f}")
        print(f"# contact sheet: {sheet}")
        print("\n".join(r["stages"]))
        print(f"# last embedded Julia set (end here to keep changing shape all the way):\n{r['julia']}")
        print(f"# minibrot (the last quarter of the zoom is rings closing in):\n{r['minibrot']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
