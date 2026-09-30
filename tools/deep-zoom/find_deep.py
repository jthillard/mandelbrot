#!/usr/bin/env python3
"""Find deep Mandelbrot minibrots near a Misiurewicz point.

Misiurewicz points (preperiodic c, f^(k+p)(0) = f^k(0)) have self-similar
spirals at every depth. A disk of radius r = 10^-D around one contains a
minibrot of size ~r^2, found by:

  1. the ball-period method: the first n where the disk's image covers 0
     gives the period p of a nucleus inside it;
  2. Newton's method on f^p(0) = 0 at high precision for the nucleus;
  3. the standard size estimate (Heiland-Allen) for its scale.

Zooming at the nucleus shows spirals, then an embedded Julia set (around
size^0.75), then the minibrot (half-height ~3x size). The spirals take
the first half of the zoom: for zooms that keep changing shape, use
morph.py / search.py instead.

Usage:
  find_deep.py REGION DEPTH [OFFSET_RE,OFFSET_IM]
  find_deep.py --list

REGION is a name from REGIONS or "re,im,k,p" for any Misiurewicz point.
OFFSET (default 0.6,0.3, in units of r) picks a different minibrot at the
same depth. If the printed size is far below 10^-(2*DEPTH), Newton didn't
converge: try another offset.

Needs mpmath (pip install mpmath).
"""

import sys

from mpmath import log10, mp, mpc, mpf

# name: (approximate seed, preperiod k, period p). The exact point is
# refined by Newton at the requested precision.
REGIONS = {
    "seahorse": (("-0.77568377", "0.13646737"), 24, 2),
    "elephant": (("0.2925", "0.0149"), 32, 3),
    "antenna": (("-0.1011", "0.9563"), 38, 4),
}


def converged(d):
    return abs(d) < mpf(10) ** (-mp.dps + 20)


def misiurewicz(c, k, p, steps=200):
    """Newton on f^(k+p)(0) - f^k(0) = 0."""
    for _ in range(steps):
        z = dz = mpc(0)
        zk = dzk = None
        for i in range(1, k + p + 1):
            dz = 2 * z * dz + 1
            z = z * z + c
            if i == k:
                zk, dzk = z, dz
        d = (z - zk) / (dz - dzk)
        c -= d
        if converged(d):
            break
    return c


def period_in_ball(c, r, maxit=1_000_000):
    """First n where the disk of radius r around c maps onto a disk containing 0."""
    z = dz = mpc(0)
    for n in range(1, maxit):
        dz = 2 * z * dz + 1
        z = z * z + c
        if abs(z) < abs(dz) * r:
            return n
        if abs(z) > 4:
            return None
    return None


def nucleus(c, p, steps=200):
    """Newton on f^p(0) = 0."""
    for _ in range(steps):
        z = dz = mpc(0)
        for _ in range(p):
            dz = 2 * z * dz + 1
            z = z * z + c
        d = z / dz
        c -= d
        if converged(d):
            break
    return c


def size(c, p):
    """Approximate size of the minibrot with nucleus c and period p."""
    z = mpc(0)
    l = b = mpc(1)
    for _ in range(1, p):
        z = z * z + c
        l = 2 * z * l
        b = b + 1 / l
    return abs(1 / (b * l * l))


def main(argv):
    if len(argv) >= 1 and argv[0] == "--list":
        for name, ((re, im), k, p) in REGIONS.items():
            print(f"{name:10} ~{re}{'+' if not im.startswith('-') else ''}{im}i  M({k},{p})")
        return 0
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2

    region, depth = argv[0], int(argv[1])
    if region in REGIONS:
        (re, im), k, p = REGIONS[region]
    else:
        re, im, k, p = region.split(",")
        k, p = int(k), int(p)
    off_re, off_im = argv[2].split(",") if len(argv) == 3 else ("0.6", "0.3")

    mp.dps = 2 * depth + 60
    m = misiurewicz(mpc(re, im), k, p)
    r = mpf(10) ** (-depth)
    c0 = m + r * mpc(off_re, off_im)
    period = period_in_ball(c0, r)
    if period is None:
        print("no period found (the disk escapes)", file=sys.stderr)
        return 1
    n = nucleus(c0, period)
    s = size(n, period)
    if s < mpf(10) ** (-2 * depth - 10):
        print(f"size {mp.nstr(s, 3)} is implausibly small: Newton didn't converge, "
              "try another offset", file=sys.stderr)
        return 1

    digits = int(-log10(s)) + 10
    log_s = float(log10(s))
    re_s = mp.nstr(n.real, digits, strip_zeros=False)
    im_s = mp.nstr(n.imag, digits, strip_zeros=False)
    print(f"# {region} depth={depth} period={period} size={mp.nstr(s, 3)}")
    print("# minibrot:")
    print(f"--view={re_s},{im_s},{mp.nstr(3 * s, 2)}")
    print(f"# embedded Julia set on the way down:")
    print(f"--view={re_s},{im_s},1e{round(0.75 * log_s)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
