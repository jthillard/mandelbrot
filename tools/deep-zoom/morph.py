#!/usr/bin/env python3
"""Chain minibrots by Julia morphing, so a deep zoom keeps changing shape.

find_deep.py puts a minibrot of size r^2 at distance r from a Misiurewicz
point, so the first half of the zoom (in log scale) is the same spiral over
and over. Here the Misiurewicz point is only used shallowly (stage 0), and
each later minibrot is picked inside the previous one's embedded Julia set,
off its centre. Zooming past a minibrot's neighbourhood folds the picture
(locally z -> z^2), so every stage shows the previous shape doubled instead
of a repeat.

Usage:
  morph.py REGION [--stages N] [--depth D] [--d0 D0] [--seed S]

REGION is a name from find_deep.REGIONS or "re,im,k,p". Prints each stage's
period, size and the views to look at. search.py tries many seeds and ranks
the resulting zooms by rendering them.

Needs mpmath.
"""

import argparse
import math
import random
import sys
from dataclasses import dataclass

from mpmath import exp, log10, mp, mpc, mpf, sqrt

from find_deep import REGIONS, misiurewicz, nucleus, period_in_ball, size

# The shader treats an exhausted reference as escaped, and the reference is
# requested with 1.5x headroom (app.rs), so periods past this can't render.
MAX_ITERATIONS = 1 << 24


@dataclass
class Stage:
    nucleus: mpc
    period: int
    size: mpf
    reach: mpf  # distance from the previous stage's centre (stage 0: the disk radius)
    rho: float = 0.0  # where the target was picked, in units of the previous julia scale
    theta: float = 0.0

    @property
    def julia(self):
        """Half-height of the embedded Julia set: geometric midpoint of reach and size."""
        return sqrt(self.size * self.reach)

    @property
    def depth(self):
        return float(-log10(self.size))


def set_precision(depth):
    mp.dps = int(2 * depth) + 60


def anchor_point(region):
    if region in REGIONS:
        (re, im), k, p = REGIONS[region]
    else:
        re, im, k, p = region.split(",")
        k, p = int(k), int(p)
    return mpc(re, im), k, p


def find_minibrot(center, radius, min_period):
    """Nucleus of the lowest-period minibrot whose atom covers the disk, or None.

    Rejects periods <= min_period (an ancestor, not something new next to
    the target) and Newton landing far from the disk (another nucleus of the
    same period).
    """
    p = period_in_ball(center, radius)
    if p is None or p <= min_period:
        return None
    n = nucleus(center, p)
    if abs(n - center) > 4 * radius:
        return None
    s = size(n, p)
    if not s > 0:
        return None
    return n, p, s


def stage0(region, d0, rng):
    """Minibrot near the Misiurewicz point at distance 10^-d0: a short spiral."""
    c, k, p = anchor_point(region)
    set_precision(2 * d0)
    m = misiurewicz(c, k, p)
    r = mpf(10) ** (-d0)
    for _ in range(20):
        theta = rng.uniform(0, 2 * math.pi)
        off = r * exp(1j * mpf(theta))
        found = find_minibrot(m + off, r, 0)
        # Size ~r^2; far below means Newton didn't converge (as in find_deep.py).
        if found and r * r * 1e-10 < found[2] < r * r * 1e6:
            n, per, s = found
            return Stage(n, per, s, abs(n - m), 0.0, theta)
    return None


def next_stage(prev, rng, max_depth, beta=0.05, tries=12):
    """A minibrot inside prev's embedded Julia set, off its centre."""
    for _ in range(tries):
        rho = rng.uniform(0.05, 0.6)
        theta = rng.uniform(0, 2 * math.pi)
        # Enough digits to place the target and find the period; the nucleus
        # is polished at the new depth's precision below, which is the costly
        # part (a few Newton steps instead of all of them).
        mp.dps = int(-log10(prev.julia)) + 60
        d = rho * prev.julia
        t = prev.nucleus + d * exp(1j * mpf(theta))
        found = find_minibrot(t, beta * d, prev.period)
        if not found:
            continue
        n, p, s = found
        # Newton didn't converge (implausibly small), no real progress, or
        # past the target depth (renders get slow and views huge).
        if s < prev.size ** 2.5 or s > prev.size ** 1.2 or -log10(s) > max_depth:
            continue
        set_precision(float(-log10(s)))
        n = nucleus(n, p)  # polish at the precision this depth needs
        s2 = size(n, p)
        if not s / 10 < s2 < s * 10:  # the polish wandered off to another nucleus
            continue
        return Stage(n, p, s2, abs(n - prev.nucleus), rho, theta)
    return None


def build_chain(region, stages, depth, d0, seed, max_period=200_000):
    """Stage list, deepest last, ending between depth and 1.5x depth if it can.

    Stops early at max_period or a dead end (no stage within reach).
    """
    rng = random.Random(seed)
    s = stage0(region, d0, rng)
    if s is None:
        return []
    chain = [s]
    while len(chain) < stages and chain[-1].depth < depth:
        s = next_stage(chain[-1], rng, 1.5 * depth)
        if s is None or s.period > min(max_period, MAX_ITERATIONS // 2):
            break
        chain.append(s)
    return chain


def view_args(stage, half_height):
    set_precision(stage.depth)
    digits = int(stage.depth) + 10
    re = mp.nstr(stage.nucleus.real, digits, strip_zeros=False)
    im = mp.nstr(stage.nucleus.imag, digits, strip_zeros=False)
    return f"--view={re},{im},{mp.nstr(half_height, 2)}"


def minibrot_view(stage):
    return view_args(stage, 3 * stage.size)


def julia_view(stage):
    return view_args(stage, stage.julia)


def describe(chain):
    lines = []
    for i, s in enumerate(chain):
        lines.append(f"# stage {i}: period={s.period} size={mp.nstr(s.size, 3)} "
                     f"julia={mp.nstr(s.julia, 3)} rho={s.rho:.2f} theta={s.theta:.2f}")
    return lines


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("region")
    ap.add_argument("--stages", type=int, default=6)
    ap.add_argument("--depth", type=float, default=300, help="stop past size 10^-DEPTH")
    ap.add_argument("--d0", type=float, default=4, help="stage 0 distance 10^-D0 from the anchor")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-period", type=int, default=200_000)
    ap.add_argument("--all-stages", action="store_true", help="print every stage's views")
    a = ap.parse_args(argv)

    chain = build_chain(a.region, a.stages, a.depth, a.d0, a.seed, a.max_period)
    if not chain:
        print("no minibrot found near the anchor, try another seed", file=sys.stderr)
        return 1
    for line in describe(chain):
        print(line)
    for i, s in enumerate(chain if a.all_stages else chain[-1:]):
        i = i if a.all_stages else len(chain) - 1
        print(f"# stage {i} embedded Julia set:\n{julia_view(s)}")
        print(f"# stage {i} minibrot:\n{minibrot_view(s)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
