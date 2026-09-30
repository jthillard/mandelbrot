//! Bivariate linear approximation (BLA): skipping many perturbation steps at
//! once.
//!
//! Near the reference, a Mandelbrot delta step `e ← 2·X_m·e + e² + dc` is
//! linear in `(e, dc)` as long as `e²` is negligible next to `2·X_m·e`, i.e.
//! `|e| < R = ε·|X_m|`. Composing such steps gives `e_{m+k} = A·e_m + B·dc`
//! for whole runs of steps. This module builds a binary table of those
//! compositions from the reference orbit (level `l` merges `2^l` steps). The
//! shader (`bla_lookup` in `mandelbrot.wgsl`) then jumps each pixel over the
//! longest run whose validity radius its delta is inside. At deep zoom every
//! pixel follows the reference for tens of thousands of iterations before
//! its orbit diverges, so most of that turns into a few table lookups.
//!
//! Merging steps `x` then `y`:
//! `A = A_y·A_x`, `B = A_y·B_x + B_y`,
//! `R = min(R_x, max(0, (R_y − |B_x|·dc_max) / |A_x|))`.
//! `R` depends on the largest pixel offset `dc_max`, so a table is only valid
//! for views whose `|dc|` stays below the one it was built with (see
//! [`dc_max_log2`]). Coefficients span far more than f64's exponent range at
//! deep zoom (`|A|` ~ 1/pixel size), so they're built as [`Fx`] (f64 mantissa
//! with its own exponent) and uploaded as f32 mantissa + i32 exponent.
//!
//! Only the plain Mandelbrot map (set plane and Julia) uses it for now.

use super::FractalKind;
use super::reference::RefOrbit;
use super::renderer::Uniforms;

/// Relative size of the dropped `e²` term that a step may have. f32's own
/// precision (2^-24) already renders indistinguishably from plain stepping,
/// but a chaotic seahorse pixel then escaped 7 iterations late against an
/// f64 emulation (`escape_counts_match`). 2^-28 fixes that at the same
/// render speed on deep views.
const EPSILON_LOG2: i32 = -28;

/// Lowest level uploaded. Lower levels are still merged, just not stored:
/// a 1..4-step jump doesn't pay for its lookup, and skipping them keeps the
/// table at 8 bytes per orbit point.
const MIN_LEVEL: u32 = 3;

/// Upload budget (bytes): the WebGPU default storage binding size limit.
const MAX_TABLE_BYTES: usize = 128 << 20;

/// One table node, as the shader's `Bla` struct: `A = a·2^a_exp`,
/// `B = b·2^b_exp`, valid while `log2|e| < r_log2`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBla {
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub a_exp: i32,
    pub b_exp: i32,
    pub r_log2: f32,
    pub _pad: u32,
}

/// `r_log2` of a node that's never valid (f32 `-inf` is fine in WGSL, but a
/// finite sentinel keeps comparisons obvious).
const NEVER: f32 = -3.0e38;

/// A BLA table ready for upload.
///
/// `meta` is `[min_level, level_count, off_0, …, off_{level_count}]`: level
/// `min_level + k` occupies `nodes[off_k..off_{k+1}]`, and its node `i`
/// covers reference steps `1 + i·2^l .. 1 + (i+1)·2^l`.
#[derive(Clone, Debug, PartialEq)]
pub struct BlaTable {
    pub nodes: Vec<GpuBla>,
    pub meta: Vec<u32>,
}

impl BlaTable {
    /// A table with no levels: the shader never jumps.
    pub fn empty() -> Self {
        Self {
            nodes: vec![GpuBla::default()],
            meta: vec![MIN_LEVEL, 0, 0],
        }
    }

    #[cfg(test)]
    fn levels(&self) -> u32 {
        self.meta[1]
    }
}

impl Default for BlaTable {
    fn default() -> Self {
        Self::empty()
    }
}

/// Whether the shader's BLA path applies to these uniforms' kind / morph
/// (the `BLA` pipeline override, see `PipelineKey`).
pub fn applies(u: &Uniforms) -> bool {
    u.kind == FractalKind::Mandelbrot as u32 && u.morph_w <= 0.0
}

/// The table for rendering `u` from `orbit` (empty when BLA doesn't apply
/// or is switched off).
pub fn for_uniforms(orbit: &RefOrbit, u: &Uniforms, enabled: bool) -> BlaTable {
    if !enabled || !applies(u) {
        return BlaTable::empty();
    }
    build(
        orbit,
        u.ref_len as usize,
        dc_max_log2(u),
        u.is_julia != 0,
        u.bailout_sq as f64,
    )
}

/// Largest `log2|dc|` a pixel sample of this view can have, rounded up with
/// one binade of headroom so the table survives small pans and zoom-outs.
/// `span` / `dc_offset` are in units of `2^scale_exp` (see `make_uniforms`);
/// AA jitter and `fs_color`'s shadow neighbours stay within `span` plus a
/// pixel, which the headroom covers.
pub fn dc_max_log2(u: &Uniforms) -> i32 {
    let dc = (u.dc_offset[0] as f64).hypot(u.dc_offset[1] as f64);
    let span = (u.span[0] as f64).hypot(u.span[1] as f64);
    let m = dc + span;
    if m <= 0.0 {
        return i32::MIN / 2;
    }
    m.log2().ceil() as i32 + 1 + u.scale_exp
}

/// Extended-range complex number `(re + i·im)·2^e`, the larger component
/// normalized to `[0.5, 1)` (or exactly zero, with `e = 0`). Scalars use
/// `im = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Fx {
    re: f64,
    im: f64,
    e: i32,
}

/// `frexp`-style exponent of a finite non-zero f64 (`|x| = f·2^k`, `f` in
/// `[0.5, 1)`).
fn exponent(x: f64) -> i32 {
    let bits = x.abs().to_bits();
    let raw = ((bits >> 52) & 0x7ff) as i32;
    if raw == 0 {
        // Subnormal: normalize through a multiply (exact).
        return exponent(x * 2f64.powi(64)) - 64;
    }
    raw - 1022
}

/// `x·2^k` for `|k|` well inside f64's exponent range (mantissas only).
fn scale(x: f64, k: i32) -> f64 {
    x * 2f64.powi(k)
}

impl Fx {
    const ZERO: Fx = Fx {
        re: 0.0,
        im: 0.0,
        e: 0,
    };

    fn new(re: f64, im: f64, e: i32) -> Fx {
        let a = re.abs().max(im.abs());
        if a == 0.0 || !a.is_finite() {
            return Fx::ZERO;
        }
        let k = exponent(a);
        Fx {
            re: scale(re, -k),
            im: scale(im, -k),
            e: e + k,
        }
    }

    fn scalar(x: f64, e: i32) -> Fx {
        Fx::new(x, 0.0, e)
    }

    fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }

    fn mul(self, o: Fx) -> Fx {
        if self.is_zero() || o.is_zero() {
            return Fx::ZERO;
        }
        Fx::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
            self.e + o.e,
        )
    }

    fn add(self, o: Fx) -> Fx {
        if self.is_zero() {
            return o;
        }
        if o.is_zero() {
            return self;
        }
        let e = self.e.max(o.e);
        let (da, db) = (self.e - e, o.e - e);
        if da < -80 {
            return o;
        }
        if db < -80 {
            return self;
        }
        Fx::new(
            scale(self.re, da) + scale(o.re, db),
            scale(self.im, da) + scale(o.im, db),
            e,
        )
    }

    fn neg(self) -> Fx {
        Fx {
            re: -self.re,
            im: -self.im,
            e: self.e,
        }
    }

    /// `|self|` as a scalar.
    fn abs(self) -> Fx {
        Fx::scalar(self.re.hypot(self.im), self.e)
    }

    /// Scalar division (both scalars, `o` non-zero).
    fn div(self, o: Fx) -> Fx {
        Fx::scalar(self.re / o.re, self.e - o.e)
    }

    /// Scalar comparison (both scalars).
    fn lt(self, o: Fx) -> bool {
        self.log2() < o.log2()
    }

    /// `log2|self|` (`-inf` for zero).
    fn log2(self) -> f64 {
        if self.is_zero() {
            return f64::NEG_INFINITY;
        }
        self.re.hypot(self.im).log2() + self.e as f64
    }
}

#[derive(Clone, Copy)]
struct Node {
    a: Fx,
    b: Fx,
    r: Fx,
}

impl Node {
    /// Step at reference point `x`: `A = 2x`, `B = 1` (0 for Julia, where no
    /// dc is added), `R = ε·|x|`.
    fn step(x: Fx, julia: bool) -> Node {
        Node {
            a: Fx { e: x.e + 1, ..x },
            b: if julia { Fx::ZERO } else { Fx::scalar(1.0, 0) },
            r: Fx {
                e: x.e + EPSILON_LOG2,
                ..x.abs()
            },
        }
    }

    /// `self` then `y`.
    fn then(self, y: Node, dc_max: Fx) -> Node {
        let a = y.a.mul(self.a);
        let b = y.a.mul(self.b).add(y.b);
        let r = if self.r.is_zero() || self.a.is_zero() {
            Fx::ZERO
        } else {
            let rem = y.r.add(self.b.abs().mul(dc_max).neg());
            if rem.re <= 0.0 {
                Fx::ZERO
            } else {
                let ry = rem.div(self.a.abs());
                if ry.lt(self.r) { ry } else { self.r }
            }
        };
        Node { a, b, r }
    }

    fn gpu(self) -> GpuBla {
        let r_log2 = self.r.log2();
        GpuBla {
            a: [self.a.re as f32, self.a.im as f32],
            b: [self.b.re as f32, self.b.im as f32],
            a_exp: self.a.e,
            b_exp: self.b.e,
            r_log2: if r_log2.is_finite() {
                r_log2 as f32
            } else {
                NEVER
            },
            _pad: 0,
        }
    }
}

/// Build the table for the first `len` points of `orbit`. `dc_max_log2` is
/// [`dc_max_log2`] of the view (ignored for Julia, where dc is 0).
///
/// A jump must not skip a pixel's escape: the shader only tests the bailout
/// between jumps, and inside a run a pixel's value is the reference's to
/// within ε. The reference itself is iterated past the pixel bailout
/// (`REFERENCE_ESCAPE_SQ`), so the table stops at its first point beyond
/// `bailout_sq`: runs may land there, never pass it.
pub fn build(
    orbit: &RefOrbit,
    len: usize,
    dc_max_log2: i32,
    julia: bool,
    bailout_sq: f64,
) -> BlaTable {
    let len = len.min(orbit.len());
    // Steps 1..=len-2 (step m reads X_m and lands on X_{m+1}, which must
    // exist). Step 0 (X_0 = 0, e ← e² + dc) is never linear.
    let limit = bailout_sq * (1.0 - 1e-3);
    let escape = (1..len).find(|&m| {
        let [re, im] = orbit.points[m];
        orbit.exps[m] == 0 && (re as f64).powi(2) + (im as f64).powi(2) >= limit
    });
    let steps = match escape {
        Some(m) => m - 1,
        None => len.saturating_sub(2),
    };
    let mut min_level = MIN_LEVEL;
    // Level l has steps >> l nodes, so levels >= min hold < 2·(steps >> min).
    while (steps >> min_level) * 2 * size_of::<GpuBla>() > MAX_TABLE_BYTES {
        min_level += 1;
    }
    if steps >> min_level == 0 {
        return BlaTable::empty();
    }
    let dc_max = if julia {
        Fx::ZERO
    } else {
        Fx::scalar(1.0, dc_max_log2)
    };
    let point = |m: usize| {
        let [re, im] = orbit.points[m];
        Fx::new(re as f64, im as f64, orbit.exps[m])
    };

    // First stored level, folded straight from single steps.
    let run = 1usize << min_level;
    let mut level: Vec<Node> = (0..steps >> min_level)
        .map(|i| {
            let start = 1 + i * run;
            (1..run).fold(Node::step(point(start), julia), |acc, k| {
                acc.then(Node::step(point(start + k), julia), dc_max)
            })
        })
        .collect();

    let mut nodes = Vec::new();
    let mut meta = vec![min_level, 0];
    while !level.is_empty() {
        meta.push(nodes.len() as u32);
        nodes.extend(level.iter().map(|n| n.gpu()));
        level = level
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[x, y]| x.then(*y, dc_max))
            .collect();
    }
    meta[1] = (meta.len() - 2) as u32;
    meta.push(nodes.len() as u32);
    BlaTable { nodes, meta }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::compute_set_reference;
    use crate::view::Big;

    fn orbit(cr: f64, ci: f64, iters: u32) -> RefOrbit {
        compute_set_reference(
            &Big::from_f64(cr, 128),
            &Big::from_f64(ci, 128),
            iters,
            128,
            FractalKind::Mandelbrot,
            2,
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            None,
        )
    }

    fn fx_to_f64(v: [f32; 2], e: i32) -> (f64, f64) {
        let s = 2f64.powi(e);
        (v[0] as f64 * s, v[1] as f64 * s)
    }

    /// Replay the shader's lookup: the longest valid aligned node at `m`.
    fn lookup(t: &BlaTable, m: usize, e_log2: f64) -> Option<(usize, usize)> {
        let (lmin, levels) = (t.meta[0], t.meta[1]);
        if m == 0 || levels == 0 {
            return None;
        }
        let j = m - 1;
        let tz = if j == 0 { 32 } else { j.trailing_zeros() };
        if tz < lmin {
            return None;
        }
        let mut best = None;
        for k in 0..=(tz - lmin).min(levels - 1) {
            let l = lmin + k;
            let base = t.meta[2 + k as usize] as usize;
            let count = t.meta[3 + k as usize] as usize - base;
            let local = j >> l;
            if local >= count {
                break;
            }
            let node = &t.nodes[base + local];
            if e_log2 >= node.r_log2 as f64 {
                break;
            }
            best = Some((base + local, 1usize << l));
        }
        best
    }

    /// Jumping with the table must track the plain perturbation iteration.
    #[test]
    fn jumps_match_stepping() {
        // A point inside the main cardioid: the orbit never escapes, so the
        // reference is long and smooth, and the deltas stay small.
        let orb = orbit(-0.1, 0.2, 5000);
        assert_eq!(orb.len(), 5001);
        let dc_max_log2 = -40;
        let t = build(&orb, orb.len(), dc_max_log2, false, 1e6);
        assert!(t.levels() > 3);
        let x = |m: usize| (orb.points[m][0] as f64, orb.points[m][1] as f64);
        let mut jumped = 0;
        for &(dr, di) in &[(0.7, 0.1), (-0.3, 0.9), (0.0, -1.0)] {
            let dc = (dr * 2f64.powi(dc_max_log2), di * 2f64.powi(dc_max_log2));
            // Reference: plain perturbation steps.
            let mut e_ref = (0.0, 0.0);
            let mut plain = vec![e_ref];
            for m in 0..orb.len() - 1 {
                let (xr, xi) = x(m);
                let (er, ei) = e_ref;
                e_ref = (
                    2.0 * (xr * er - xi * ei) + er * er - ei * ei + dc.0,
                    2.0 * (xr * ei + xi * er) + 2.0 * er * ei + dc.1,
                );
                plain.push(e_ref);
            }
            // With jumps.
            let (mut m, mut e) = (0usize, (0.0f64, 0.0f64));
            while m < orb.len() - 1 {
                let e_log2 = e.0.hypot(e.1).log2();
                if let Some((idx, n)) = lookup(&t, m, e_log2) {
                    let nd = &t.nodes[idx];
                    let (ar, ai) = fx_to_f64(nd.a, nd.a_exp);
                    let (br, bi) = fx_to_f64(nd.b, nd.b_exp);
                    e = (
                        ar * e.0 - ai * e.1 + br * dc.0 - bi * dc.1,
                        ar * e.1 + ai * e.0 + br * dc.1 + bi * dc.0,
                    );
                    m += n;
                    jumped += n;
                } else {
                    let (xr, xi) = x(m);
                    let (er, ei) = e;
                    e = (
                        2.0 * (xr * er - xi * ei) + er * er - ei * ei + dc.0,
                        2.0 * (xr * ei + xi * er) + 2.0 * er * ei + dc.1,
                    );
                    m += 1;
                }
                let want = plain[m];
                let err = (e.0 - want.0).hypot(e.1 - want.1);
                let mag = want.0.hypot(want.1);
                assert!(err <= 1e-4 * mag + 1e-300, "m={m}: {e:?} vs {want:?}");
            }
        }
        assert!(jumped > (orb.len() - 1) / 2, "only {jumped} steps jumped");
    }

    /// The shader's f32 loop (rebasing, escape, smooth count) with and
    /// without jumps, on a chaotic seahorse view: escape iteration counts and
    /// smooth values must agree.
    #[test]
    fn escape_counts_match() {
        let (cr, ci) = (-0.743_643_887_037_151, 0.131_825_904_205_330);
        let orb = orbit(cr, ci, 20_000);
        let half = 1e-9f64;
        let dc_max_log2 = (2.0 * half).log2().ceil() as i32 + 1;
        let t = build(&orb, orb.len(), dc_max_log2, false, 1e6);
        let x = |m: usize| (orb.points[m][0] as f64, orb.points[m][1] as f64);
        let run = |dc: (f64, f64), bla: bool| -> (u32, f64) {
            let (mut m, mut n, mut e) = (0usize, 0u32, (0.0f64, 0.0f64));
            loop {
                let xm = x(m);
                let z = (xm.0 + e.0, xm.1 + e.1);
                let z2 = z.0 * z.0 + z.1 * z.1;
                if z2 > 1e6 || n >= 20_000 {
                    return (n, z2);
                }
                let hit = if bla && m != 0 {
                    lookup(&t, m, e.0.hypot(e.1).log2())
                } else {
                    None
                };
                if let Some((idx, k)) = hit {
                    let nd = &t.nodes[idx];
                    let (ar, ai) = fx_to_f64(nd.a, nd.a_exp);
                    let (br, bi) = fx_to_f64(nd.b, nd.b_exp);
                    e = (
                        ar * e.0 - ai * e.1 + br * dc.0 - bi * dc.1,
                        ar * e.1 + ai * e.0 + br * dc.1 + bi * dc.0,
                    );
                    m += k;
                    n += k as u32;
                } else {
                    let (er, ei) = e;
                    e = (
                        2.0 * (xm.0 * er - xm.1 * ei) + er * er - ei * ei + dc.0,
                        2.0 * (xm.0 * ei + xm.1 * er) + 2.0 * er * ei + dc.1,
                    );
                    m += 1;
                    n += 1;
                }
                if m >= orb.len() {
                    return (u32::MAX, 0.0);
                }
                let xm = x(m);
                let z = (xm.0 + e.0, xm.1 + e.1);
                if z.0 * z.0 + z.1 * z.1 < e.0 * e.0 + e.1 * e.1 {
                    e = z;
                    m = 0;
                }
            }
        };
        let mut jumped_any = false;
        for i in 0..40 {
            let a = i as f64 * 0.37;
            let dc = (half * a.cos(), half * (1.3 * a).sin());
            let (n0, z0) = run(dc, false);
            let (n1, z1) = run(dc, true);
            jumped_any |= n0 > 1000;
            let smooth = |n: u32, z2: f64| n as f64 - (0.5 * z2.ln()).log2();
            assert!(
                (smooth(n0, z0) - smooth(n1, z1)).abs() < 0.05,
                "dc {dc:?}: plain {n0} ({z0}), bla {n1} ({z1})"
            );
        }
        assert!(jumped_any);
    }

    /// A table built for a small dc_max must reject deltas it can't cover.
    #[test]
    fn radius_shrinks_with_dc_max() {
        let orb = orbit(-0.1, 0.2, 2000);
        let small = build(&orb, orb.len(), -40, false, 1e6);
        let big = build(&orb, orb.len(), -4, false, 1e6);
        let top = |t: &BlaTable| t.nodes[t.meta[2 + t.levels() as usize - 1] as usize].r_log2;
        assert!(top(&big) < top(&small));
    }

    #[test]
    fn layout() {
        let orb = orbit(-0.1, 0.2, 1000);
        let t = build(&orb, orb.len(), -30, false, 1e6);
        let steps = orb.len() - 2;
        assert_eq!(t.meta[0], MIN_LEVEL);
        for k in 0..t.levels() as usize {
            let count = t.meta[3 + k] - t.meta[2 + k];
            assert_eq!(count as usize, steps >> (MIN_LEVEL as usize + k));
        }
        assert_eq!(*t.meta.last().unwrap() as usize, t.nodes.len());
        let tiny = orbit(-3.0, 0.0, 1000); // escapes at once
        assert_eq!(build(&tiny, tiny.len(), -30, false, 1e6).levels(), 0);
    }
}
