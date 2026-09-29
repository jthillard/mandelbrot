#!/usr/bin/env python3
"""Where does each config family meet a VMAF target? Reads sweep.sh TSVs.

    tools/encode/ladder.py results/x265-ladder.tsv [--target vmaf:95 --target p1:90 ...]

A family is a config name minus its trailing -CRF (x265-rect-16 -> x265-rect;
x265-crf16 -> x265).
For each family and clip, log(Mb/s) and CRF are interpolated linearly in the
chosen metric between the two ladder rungs that bracket the target. A tier
uses one CRF for everything, so the clip needing the lowest CRF binds it; that
CRF is printed last as "all". A dash means the ladder doesn't reach the target.
"""
import argparse
import csv
import math
import re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("tsv", nargs="+")
ap.add_argument("--target", action="append",
                help="metric:value, metric one of vmaf, p1, min, neg (default vmaf:95, p1:90, p1:97)")
args = ap.parse_args()
targets = [(m, float(v)) for m, v in (t.split(":") for t in (args.target or ["vmaf:95", "p1:90", "p1:97"]))]
col = {"vmaf": "vmaf", "p1": "vmaf_p1", "min": "vmaf_min", "neg": "vmaf_neg"}

fam = defaultdict(lambda: defaultdict(list))  # family -> clip -> [(crf, row)]
for path in args.tsv:
    for r in csv.DictReader(open(path), delimiter="\t"):
        m = re.fullmatch(r"(.*)-(?:crf)?(\d+)", r["config"])
        if m:
            fam[m[1]][r["clip"]].append((int(m[2]), r))


def solve(pts, key, target):
    """(crf, mbps, fps) where metric `key` crosses `target`, or None."""
    pts = sorted(pts, key=lambda p: p[0])  # quality falls as CRF rises
    for (c0, a), (c1, b) in zip(pts, pts[1:]):
        q0, q1 = float(a[key]), float(b[key])
        if q0 >= target >= q1 and q0 != q1:
            t = (q0 - target) / (q0 - q1)
            lr = math.log(float(a["mbps"])) * (1 - t) + math.log(float(b["mbps"])) * t
            fps = float(a["fps"]) * (1 - t) + float(b["fps"]) * t
            return c0 + t * (c1 - c0), math.exp(lr), fps
    return None


for metric, target in targets:
    print(f"\n## {metric} >= {target:g}   (CRF @ Mb/s, fps)")
    clips = sorted({c for f in fam.values() for c in f})
    print(f"{'family':24}" + "".join(f"{c:>22}" for c in clips) + f"{'all (binding clip)':>24}")
    for f, per in fam.items():
        cells, worst = [], None
        for c in clips:
            s = solve(per.get(c, []), col[metric], target)
            cells.append(f"{s[0]:5.1f} @ {s[1]:6.1f}, {s[2]:4.1f}" if s else "-")
            if s is None:
                worst = "missing"
            elif worst != "missing" and (worst is None or s[0] < worst[0]):
                worst = (s[0], c)
        w = "-" if worst in (None, "missing") else f"CRF {worst[0]:4.1f} ({worst[1]})"
        print(f"{f:24}" + "".join(f"{x:>22}" for x in cells) + f"{w:>24}")
