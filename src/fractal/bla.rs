//! Bivariate linear approximation (BLA): skipping many perturbation steps at
//! once.
//!
//! Near the reference, a delta step `e ← f(X_m + e) − f(X_m) + dc` is linear
//! in `(e, dc)` as long as its non-linear part is negligible:
//! `e' ≈ M_m·e + dc`, with `M_m` the real 2×2 Jacobian of the map at `X_m`.
//! Composing such steps gives `e_{m+k} = M·e_m + N·dc` for whole runs of
//! steps. This module builds a binary table of those compositions from the
//! reference orbit (level `l` merges `2^l` steps). The shader (`bla_lookup`
//! in `mandelbrot.wgsl`) then jumps each pixel over the longest run whose
//! validity radius `R` its delta is inside. At deep zoom every pixel follows
//! the reference for tens of thousands of iterations before its orbit
//! diverges, so most of that turns into a few table lookups.
//!
//! Real matrices rather than complex coefficients, so one shader path covers
//! every kind: `M` is complex multiplication by `f'(X)` for the holomorphic
//! kinds, and a sign fold of `C(2X)` for the abs / conjugate ones (see
//! [`Node::step`], which also holds each kind's radius). The folds are exact
//! while the delta doesn't cross the fold line, so their radius also keeps
//! it on one side. Phoenix's two-term map would need a 4×4 state matrix and
//! isn't covered; neither are kind-switch morphs.
//!
//! Merging steps `x` then `y`:
//! `M = M_y·M_x`, `N = M_y·N_x + N_y`,
//! `R = min(R_x, max(0, (R_y − ‖N_x‖·dc_max) / ‖M_x‖))`, with `‖·‖` the
//! spectral norm (exact for 2×2: the Frobenius norm would overestimate
//! conformal maps by √2, and shrink R by that at every level).
//! `R` depends on the largest pixel offset `dc_max`, so a table is only valid
//! for views whose `|dc|` stays below the one it was built with (see
//! [`dc_max_log2`]). Coefficients span far more than f64's exponent range at
//! deep zoom (`‖M‖` ~ 1/pixel size), so they're built as floatexp ([`Fx`],
//! [`Mx`]: f64 mantissas with their own exponent) and uploaded as f32
//! mantissas + i32 exponent.

use super::FractalKind;
use super::reference::RefOrbit;
use super::renderer::Uniforms;

/// Relative size of the dropped non-linear term that a step may have. f32's
/// own precision (2^-24) already renders indistinguishably from plain
/// stepping, but a chaotic seahorse pixel then escaped 7 iterations late
/// against an f64 emulation (`escape_counts_match`). 2^-28 fixes that at the
/// same render speed on deep views.
const EPSILON_LOG2: i32 = -28;

/// Lowest level uploaded. Lower levels are still merged, just not stored:
/// a 1..4-step jump doesn't pay for its lookup, and skipping them keeps the
/// table at 12 bytes per orbit point.
const MIN_LEVEL: u32 = 3;

/// A step whose own radius is below `2^SEG_BAD_LOG2` (`|X| < 2^-12` for
/// Mandelbrot: a near-zero pass of the reference, or a fold line for the abs
/// kinds) ends a segment (see [`build`]).
const SEG_BAD_LOG2: f64 = EPSILON_LOG2 as f64 - 12.0;

/// Shortest segment worth a node (the levels cover shorter runs in a
/// lookup or two). Also bounds the segment count to `steps / SEG_MIN_STEPS`.
const SEG_MIN_STEPS: usize = 64;

/// Steps folded per parallel task when building segments.
const SEG_CHUNK: usize = 1024;

/// Upload budget (bytes): the WebGPU default storage binding size limit.
const MAX_TABLE_BYTES: usize = 128 << 20;

/// One table node, as the shader's `Bla` struct: `M = m·2^m_exp`,
/// `N = n·2^n_exp` (row-major 2×2: `[m00, m01, m10, m11]`), valid while
/// `log2|e| < r_log2`. `steps` is the run length of a segment node (0 in
/// the levels, where it follows from the level).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBla {
    pub m: [f32; 4],
    pub n: [f32; 4],
    pub m_exp: i32,
    pub n_exp: i32,
    pub r_log2: f32,
    pub steps: u32,
}

/// `r_log2` of a node that's never valid (f32 `-inf` is fine in WGSL, but a
/// finite sentinel keeps comparisons obvious).
const NEVER: f32 = -3.0e38;

/// A BLA table ready for upload.
///
/// `meta` is `[min_level, level_count, off_0, …, off_{level_count},
/// seg_count, seg_start_0, …]`: level `min_level + k` occupies
/// `nodes[off_k..off_{k+1}]`, and its node `i` covers reference steps
/// `1 + i·2^l .. 1 + (i+1)·2^l`. Segment `i` is `nodes[off_{level_count} + i]`
/// and covers `steps` steps from `seg_start_i` (ascending); see [`build`].
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
            meta: vec![MIN_LEVEL, 0, 0, 0],
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
/// (the `BLA` pipeline override, see `PipelineKey`): every kind but Phoenix.
pub fn applies(u: &Uniforms) -> bool {
    u.kind != FractalKind::Phoenix as u32 && u.morph_w <= 0.0
}

/// The iteration map a table linearizes, with the same (f32) constants the
/// shader's delta step uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepMap {
    pub kind: FractalKind,
    /// Multibrot exponent, clamped like the shader's.
    pub power: u32,
    pub lambda: (f64, f64),
    pub complex_power: (f64, f64),
    /// Julia plane: c is fixed, so no `dc` is added per step (`N = 0`).
    pub julia: bool,
}

impl StepMap {
    pub fn from_uniforms(u: &Uniforms) -> Option<Self> {
        let kind = *FractalKind::ALL.get(u.kind as usize)?;
        Some(Self {
            kind,
            power: u.power.clamp(2, 20),
            lambda: (u.lambda_l[0] as f64, u.lambda_l[1] as f64),
            complex_power: (u.complex_power[0] as f64, u.complex_power[1] as f64),
            julia: u.is_julia != 0,
        })
    }
}

/// The table for rendering `u` from `orbit` (empty when BLA doesn't apply
/// or is switched off).
pub fn for_uniforms(orbit: &RefOrbit, u: &Uniforms, enabled: bool) -> BlaTable {
    match StepMap::from_uniforms(u) {
        Some(map) if enabled && applies(u) => build(
            orbit,
            u.ref_len as usize,
            dc_max_log2(u),
            u.bailout_sq as f64,
            &map,
        ),
        _ => BlaTable::empty(),
    }
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

    /// `self·2^k` (exact).
    fn shl(self, k: i32) -> Fx {
        if self.is_zero() {
            return self;
        }
        Fx {
            e: self.e + k,
            ..self
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

    /// Scalar minimum (both scalars, non-negative).
    fn min(self, o: Fx) -> Fx {
        if self.log2() < o.log2() { self } else { o }
    }

    /// `log2|self|` (`-inf` for zero).
    fn log2(self) -> f64 {
        if self.is_zero() {
            return f64::NEG_INFINITY;
        }
        self.re.hypot(self.im).log2() + self.e as f64
    }

    /// `self^k` for an integer `k >= 0`.
    fn powi(self, k: u32) -> Fx {
        (0..k).fold(Fx::scalar(1.0, 0), |acc, _| acc.mul(self))
    }

    /// Principal `self^p` for a complex `p` (the branch `cpow` and the
    /// reference use: arg in (-π, π]). 0 for 0, like `cpow`.
    fn cpow(self, p: (f64, f64)) -> Fx {
        if self.is_zero() {
            return Fx::ZERO;
        }
        let ln_r = self.re.hypot(self.im).ln() + self.e as f64 * std::f64::consts::LN_2;
        let theta = self.im.atan2(self.re);
        let lr = p.0 * ln_r - p.1 * theta;
        let li = p.0 * theta + p.1 * ln_r;
        let k = (lr / std::f64::consts::LN_2).floor();
        let mag = (lr - k * std::f64::consts::LN_2).exp();
        let (sin, cos) = li.sin_cos();
        Fx::new(mag * cos, mag * sin, k as i32)
    }
}

/// Extended-range real 2×2 matrix `[[a, b], [c, d]]·2^e` (row-major `m`),
/// normalized so its largest entry is in `[0.5, 1)` (or all zero, `e = 0`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Mx {
    m: [f64; 4],
    e: i32,
}

impl Mx {
    const ZERO: Mx = Mx { m: [0.0; 4], e: 0 };
    const IDENTITY: Mx = Mx {
        m: [0.5, 0.0, 0.0, 0.5],
        e: 1,
    };

    fn new(m: [f64; 4], e: i32) -> Mx {
        let a = m.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if a == 0.0 || !a.is_finite() {
            return Mx::ZERO;
        }
        let k = exponent(a);
        Mx {
            m: m.map(|v| scale(v, -k)),
            e: e + k,
        }
    }

    /// Complex multiplication by `a`.
    fn complex(a: Fx) -> Mx {
        Mx::new([a.re, -a.im, a.im, a.re], a.e)
    }

    /// `diag(s0, s1)·self`: flip the sign of rows (the abs folds).
    fn fold(self, s0: f64, s1: f64) -> Mx {
        let [a, b, c, d] = self.m;
        Mx {
            m: [s0 * a, s0 * b, s1 * c, s1 * d],
            e: self.e,
        }
    }

    fn is_zero(self) -> bool {
        self.m == [0.0; 4]
    }

    /// `self·o`.
    fn mul(self, o: Mx) -> Mx {
        if self.is_zero() || o.is_zero() {
            return Mx::ZERO;
        }
        let [a, b, c, d] = self.m;
        let [p, q, r, s] = o.m;
        Mx::new(
            [a * p + b * r, a * q + b * s, c * p + d * r, c * q + d * s],
            self.e + o.e,
        )
    }

    fn add(self, o: Mx) -> Mx {
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
        let mut m = [0.0; 4];
        for (k, v) in m.iter_mut().enumerate() {
            *v = scale(self.m[k], da) + scale(o.m[k], db);
        }
        Mx::new(m, e)
    }

    /// Largest singular value (the operator norm), as a scalar.
    fn norm(self) -> Fx {
        let [a, b, c, d] = self.m;
        let f2 = a * a + b * b + c * c + d * d;
        let det = a * d - b * c;
        let disc = (f2 * f2 - 4.0 * det * det).max(0.0).sqrt();
        Fx::scalar(((f2 + disc) / 2.0).sqrt(), self.e)
    }
}

#[derive(Clone, Copy)]
struct Node {
    m: Mx,
    n: Mx,
    r: Fx,
}

/// Sign of a mantissa, as the fold factor (+1 at 0: the radius is 0 there
/// anyway, so the node is never used).
fn sign(x: f64) -> f64 {
    if x < 0.0 { -1.0 } else { 1.0 }
}

impl Node {
    /// One step of `map` at reference point `x`: its Jacobian `M`, `N = I`
    /// (0 for Julia, where no dc is added) and the radius within which the
    /// step is linear to ε. Must match `advance_delta_kind` in the shader
    /// (the same maps as `reference.rs::step_f64`).
    ///
    /// ε bounds: the dropped second-order term relative to the linear one
    /// (|e|/(2|X|) for z², (p−1)|e|/(2|X|) for z^p, |e|/|1−2X| for Lambda)
    /// is kept at ε/2. The folds (abs kinds) are isometries, so their M
    /// has both singular values 2|X|, like Mandelbrot's; their fold bounds
    /// keep the delta from crossing the line where the abs switches branch
    /// (with a factor ½ for the dropped e² there). Complex Multibrot must also
    /// not cross the principal branch cut (the negative real axis), like
    /// `complex_multibrot_delta`.
    fn step(map: &StepMap, x: Fx) -> Node {
        let eps = |r: Fx| r.shl(EPSILON_LOG2);
        let c2x = Mx::complex(x.shl(1));
        let (re, im) = (x.re, x.im); // x's components, in units of 2^x.e
        let hyp = re.hypot(im);
        // Fold bounds, as scalars at x's scale:
        // |xy|/(|x|+|y|) (sign of xy), |x²−y²|/(2|X|) (sign of x²−y²), |y|.
        let fold_xy = || Fx::scalar(0.5 * (re * im).abs() / (re.abs() + im.abs()), x.e);
        let fold_sq = || Fx::scalar(0.5 * (re * re - im * im).abs() / (2.0 * hyp), x.e);
        let (m, r) = if x.is_zero() {
            (Mx::ZERO, Fx::ZERO)
        } else {
            match map.kind {
                FractalKind::Mandelbrot => (c2x, eps(x.abs())),
                FractalKind::Tricorn => (c2x.fold(1.0, -1.0), eps(x.abs())),
                FractalKind::BurningShip => {
                    (c2x.fold(1.0, sign(re * im)), eps(x.abs()).min(fold_xy()))
                }
                FractalKind::Celtic => (
                    c2x.fold(sign(re * re - im * im), 1.0),
                    eps(x.abs()).min(fold_sq()),
                ),
                FractalKind::Buffalo => (
                    c2x.fold(sign(re * re - im * im), -sign(re * im)),
                    eps(x.abs()).min(fold_xy()).min(fold_sq()),
                ),
                FractalKind::Perpendicular => (
                    c2x.fold(1.0, -sign(im)),
                    eps(x.abs()).min(Fx::scalar(0.5 * im.abs(), x.e)),
                ),
                FractalKind::Multibrot => {
                    let p = map.power;
                    let a = x.powi(p - 1).mul(Fx::scalar(p as f64, 0));
                    (
                        Mx::complex(a),
                        eps(x.abs().div(Fx::scalar((p - 1) as f64, 0))),
                    )
                }
                FractalKind::Lambda => {
                    let (lr, li) = map.lambda;
                    let one_2x = Fx::scalar(1.0, 0).add(x.shl(1).neg());
                    let a = Fx::new(lr, li, 0).mul(one_2x);
                    (Mx::complex(a), eps(one_2x.abs().shl(-1)))
                }
                FractalKind::ComplexMultibrot => {
                    let (pr, pi) = map.complex_power;
                    let a = x.cpow((pr - 1.0, pi)).mul(Fx::new(pr, pi, 0));
                    let cut = if re < 0.0 {
                        Fx::scalar(0.5 * im.abs(), x.e)
                    } else {
                        Fx::scalar(0.5 * hyp, x.e)
                    };
                    let q = (pr - 1.0).hypot(pi);
                    let r = if q > 1e-12 {
                        eps(x.abs().div(Fx::scalar(q, 0))).min(cut)
                    } else {
                        cut // p = 1: the map is linear
                    };
                    (Mx::complex(a), r)
                }
                // Not linearized (see `applies`).
                FractalKind::Phoenix => (Mx::ZERO, Fx::ZERO),
            }
        };
        Node {
            m,
            n: if map.julia { Mx::ZERO } else { Mx::IDENTITY },
            r,
        }
    }

    /// `self` then `y`.
    fn then(self, y: Node, dc_max: Fx) -> Node {
        let m = y.m.mul(self.m);
        let n = y.m.mul(self.n).add(y.n);
        let r = if self.r.is_zero() || self.m.is_zero() {
            Fx::ZERO
        } else {
            let rem = y.r.add(self.n.norm().mul(dc_max).neg());
            if rem.re <= 0.0 {
                Fx::ZERO
            } else {
                rem.div(self.m.norm()).min(self.r)
            }
        };
        Node { m, n, r }
    }

    fn gpu(self) -> GpuBla {
        let r_log2 = self.r.log2();
        GpuBla {
            m: self.m.m.map(|v| v as f32),
            n: self.n.m.map(|v| v as f32),
            m_exp: self.m.e,
            n_exp: self.n.e,
            r_log2: if r_log2.is_finite() {
                r_log2 as f32
            } else {
                NEVER
            },
            steps: 0,
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
    bailout_sq: f64,
    map: &StepMap,
) -> BlaTable {
    let len = len.min(orbit.len());
    // Steps 1..=len-2 (step m reads X_m and lands on X_{m+1}, which must
    // exist). Step 0 (X_0 = 0 in the set plane, where e ← e² + dc) is never
    // linear for the z² kinds, so it's never approximated.
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
    while ((steps >> min_level) * 2 + steps / SEG_MIN_STEPS) * size_of::<GpuBla>() > MAX_TABLE_BYTES
    {
        min_level += 1;
    }
    if steps >> min_level == 0 {
        return BlaTable::empty();
    }
    let dc_max = if map.julia {
        Fx::ZERO
    } else {
        Fx::scalar(1.0, dc_max_log2)
    };
    let point = |m: usize| {
        let [re, im] = orbit.points[m];
        Fx::new(re as f64, im as f64, orbit.exps[m])
    };

    // First stored level, folded straight from single steps. The same pass
    // notes the "bad" steps (tiny radius) that end segments (below).
    let run = 1usize << min_level;
    let is_bad = |nd: &Node| nd.r.log2() < SEG_BAD_LOG2;
    let first: Vec<(Node, Vec<usize>)> = par_map(steps >> min_level, 4096, |i| {
        let start = 1 + i * run;
        let mut bad = Vec::new();
        let mut acc = Node::step(map, point(start));
        if is_bad(&acc) {
            bad.push(start);
        }
        for m in start + 1..start + run {
            let nd = Node::step(map, point(m));
            if is_bad(&nd) {
                bad.push(m);
            }
            acc = acc.then(nd, dc_max);
        }
        (acc, bad)
    });
    let base: Vec<Node> = first.iter().map(|f| f.0).collect();
    let tail = 1 + base.len() * run..steps + 1; // past the last whole run
    let bad: Vec<usize> = first
        .into_iter()
        .flat_map(|f| f.1)
        .chain(tail.filter(|&m| is_bad(&Node::step(map, point(m)))))
        .collect();

    let mut nodes = Vec::new();
    let mut meta = vec![min_level, 0];
    let mut level = base.clone();
    while !level.is_empty() {
        meta.push(nodes.len() as u32);
        nodes.extend(level.iter().map(|n| n.gpu()));
        level = par_map(level.len() / 2, 4096, |i| {
            level[2 * i].then(level[2 * i + 1], dc_max)
        });
    }
    meta[1] = (meta.len() - 2) as u32;
    meta.push(nodes.len() as u32);

    // Segments: one node per run between bad steps, from step 1 and from
    // just after each bad step, where a pixel lands after stepping over it
    // or rebasing. Deep minibrot views pass near 0 once per period, so the
    // aligned levels cut every period into ~10 jumps plus up to 2^min_level
    // single steps; a segment crosses it in one.
    let mut segs = Vec::new();
    let mut start = 1;
    for b in bad.into_iter().chain([steps + 1]) {
        if b >= start + SEG_MIN_STEPS {
            segs.push((start, b));
        }
        start = b + 1;
    }
    // Folded in parallel pieces cut at multiples of `chunk` (aligned to the
    // first level), each from first-level nodes plus single steps at its
    // ragged ends, then merged in order.
    let chunk = SEG_CHUNK.max(run);
    let pieces: Vec<(usize, usize)> = segs
        .iter()
        .flat_map(|&(a, b)| {
            let cuts = (a - 1) / chunk + 1..(b - 2) / chunk + 1;
            std::iter::once(a)
                .chain(cuts.map(move |k| 1 + k * chunk))
                .chain([b])
                .collect::<Vec<_>>()
                .windows(2)
                .map(|w| (w[0], w[1]))
                .collect::<Vec<_>>()
        })
        .collect();
    let folded = par_map(pieces.len(), 4, |i| {
        let (a, b) = pieces[i];
        let mut acc: Option<Node> = None;
        let mut m = a;
        while m < b {
            let aligned = (m - 1) % run == 0 && m + run <= b && (m - 1) / run < base.len();
            let (nd, k) = if aligned {
                (base[(m - 1) / run], run)
            } else {
                (Node::step(map, point(m)), 1)
            };
            acc = Some(acc.map_or(nd, |x| x.then(nd, dc_max)));
            m += k;
        }
        acc.unwrap()
    });
    let mut k = 0;
    for &(a, b) in &segs {
        let mut seg = folded[k];
        k += 1;
        while k < pieces.len() && pieces[k].0 < b && pieces[k].0 > a {
            seg = seg.then(folded[k], dc_max);
            k += 1;
        }
        nodes.push(GpuBla {
            steps: (b - a) as u32,
            ..seg.gpu()
        });
    }
    meta.push(segs.len() as u32);
    meta.extend(segs.iter().map(|&(a, _)| a as u32));
    BlaTable { nodes, meta }
}

/// `(0..n).map(f).collect()`, split across cores when there are at least
/// `min` items, enough work to pay for the threads (a 900k-step table took
/// ~60 ms on one core).
fn par_map<T: Send>(n: usize, min: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    #[cfg(not(target_arch = "wasm32"))]
    if n >= min {
        let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
        let chunk = n.div_ceil(threads);
        let f = &f;
        return std::thread::scope(|scope| {
            let parts: Vec<_> = (0..n)
                .step_by(chunk)
                .map(|a| scope.spawn(move || (a..(a + chunk).min(n)).map(f).collect::<Vec<T>>()))
                .collect();
            parts.into_iter().flat_map(|p| p.join().unwrap()).collect()
        });
    }
    (0..n).map(f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractal::{compute_reference, compute_set_reference};
    use crate::view::Big;

    const LAMBDA: (f64, f64) = (1.0, 0.0);
    const CPOW: (f64, f64) = (2.5, 0.3);

    fn map(kind: FractalKind, julia: bool) -> StepMap {
        StepMap {
            kind,
            power: 3,
            lambda: LAMBDA,
            complex_power: CPOW,
            julia,
        }
    }

    /// Reference orbit from `z0` with parameter `c` (`z0 = 0`: set plane).
    fn orbit(map: &StepMap, z0: (f64, f64), c: (f64, f64), iters: u32) -> RefOrbit {
        let b = |v: f64| Big::from_f64(v, 128);
        compute_reference(
            &b(z0.0),
            &b(z0.1),
            &b(c.0),
            &b(c.1),
            iters,
            128,
            map.kind,
            map.power,
            (0.0, 0.0),
            map.lambda,
            map.complex_power,
            None,
        )
    }

    fn mandel_orbit(cr: f64, ci: f64, iters: u32) -> RefOrbit {
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

    type C = (f64, f64);

    fn cmul(a: C, b: C) -> C {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    fn add(a: C, b: C) -> C {
        (a.0 + b.0, a.1 + b.1)
    }

    fn mag2(a: C) -> f64 {
        a.0 * a.0 + a.1 * a.1
    }

    /// `M·e + N·dc` for a node, in f64.
    fn apply(nd: &GpuBla, e: C, dc: C) -> C {
        let mv = |m: [f32; 4], k: i32, v: C| {
            let s = 2f64.powi(k);
            (
                (m[0] as f64 * v.0 + m[1] as f64 * v.1) * s,
                (m[2] as f64 * v.0 + m[3] as f64 * v.1) * s,
            )
        };
        add(mv(nd.m, nd.m_exp, e), mv(nd.n, nd.n_exp, dc))
    }

    /// The shader's `diffabs`: |c + d| − |c| without cancellation.
    fn diffabs(c: f64, d: f64) -> f64 {
        let cd = c + d;
        if c >= 0.0 {
            if cd >= 0.0 { d } else { -(2.0 * c + d) }
        } else if cd > 0.0 {
            2.0 * c + d
        } else {
            -d
        }
    }

    fn cpow(z: C, p: C) -> C {
        if z == (0.0, 0.0) {
            return (0.0, 0.0);
        }
        let ln_r = 0.5 * mag2(z).ln();
        let th = z.1.atan2(z.0);
        let mag = (p.0 * ln_r - p.1 * th).exp();
        let (s, c) = (p.0 * th + p.1 * ln_r).sin_cos();
        (mag * c, mag * s)
    }

    /// f64 port of the shader's `advance_delta_kind`: f(X + e) − f(X).
    fn delta(map: &StepMap, z: C, e: C) -> C {
        let sq = add(cmul((2.0 * z.0, 2.0 * z.1), e), cmul(e, e));
        match map.kind {
            FractalKind::Mandelbrot => sq,
            FractalKind::Tricorn => {
                let (cz, ce) = ((z.0, -z.1), (e.0, -e.1));
                add(cmul((2.0 * cz.0, 2.0 * cz.1), ce), cmul(ce, ce))
            }
            FractalKind::BurningShip => {
                let dp = z.0 * e.1 + z.1 * e.0 + e.0 * e.1;
                (sq.0, 2.0 * diffabs(z.0 * z.1, dp))
            }
            FractalKind::Celtic => (diffabs(z.0 * z.0 - z.1 * z.1, sq.0), sq.1),
            FractalKind::Buffalo => (
                diffabs(z.0 * z.0 - z.1 * z.1, sq.0),
                -diffabs(2.0 * z.0 * z.1, sq.1),
            ),
            FractalKind::Perpendicular => {
                let da = diffabs(z.1, e.1);
                let ay = z.1.abs() + da;
                (sq.0, -2.0 * (z.0 * da + e.0 * ay))
            }
            FractalKind::Multibrot => {
                // (Z+e)^p − Z^p = e·Σ_k (Z+e)^k Z^(p−1−k), Horner-style.
                let y = add(z, e);
                let (mut s, mut zj) = ((1.0, 0.0), (1.0, 0.0));
                for _ in 1..map.power {
                    zj = cmul(zj, z);
                    s = add(cmul(s, y), zj);
                }
                cmul(e, s)
            }
            FractalKind::Lambda => {
                let t = (1.0 - 2.0 * z.0 - e.0, -2.0 * z.1 - e.1);
                cmul(map.lambda, cmul(e, t))
            }
            FractalKind::ComplexMultibrot => {
                let p = map.complex_power;
                let y = add(z, e);
                let crosses = z.0 < 0.0 && ((z.1 < 0.0) != (y.1 < 0.0));
                let w2 = mag2(e) / mag2(z);
                if w2 < 0.25 && !crosses {
                    let zz = mag2(z);
                    let w = cmul(e, (z.0 / zz, -z.1 / zz));
                    let (mut wk, mut acc, mut coef) = (w, (0.0, 0.0), (1.0, 0.0));
                    for k in 1..=40 {
                        coef = cmul(coef, (p.0 - (k - 1) as f64, p.1));
                        coef = (coef.0 / k as f64, coef.1 / k as f64);
                        acc = add(acc, cmul(coef, wk));
                        wk = cmul(wk, w);
                        if mag2(wk) < 1e-36 * mag2(acc) {
                            break;
                        }
                    }
                    cmul(cpow(z, p), acc)
                } else {
                    let (a, b) = (cpow(y, p), cpow(z, p));
                    (a.0 - b.0, a.1 - b.1)
                }
            }
            FractalKind::Phoenix => unreachable!(),
        }
    }

    /// Replay the shader's lookup: the segment starting at `m` if it's
    /// valid, else the longest valid aligned node at `m`, of at most
    /// `budget` steps.
    fn lookup(t: &BlaTable, m: usize, e_log2: f64, budget: usize) -> Option<(usize, usize)> {
        let (lmin, levels) = (t.meta[0], t.meta[1]);
        if m == 0 || levels == 0 {
            return None;
        }
        let sc = 3 + levels as usize; // seg_count
        let starts = &t.meta[sc + 1..sc + 1 + t.meta[sc] as usize];
        if let Ok(i) = starts.binary_search(&(m as u32)) {
            let idx = t.meta[sc - 1] as usize + i;
            let nd = &t.nodes[idx];
            if e_log2 < nd.r_log2 as f64 && nd.steps as usize <= budget {
                return Some((idx, nd.steps as usize));
            }
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
            if (1usize << l) > budget || local >= count {
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

    struct Escape {
        smooth: f64,
        jumped: usize,
    }

    /// The shader's f32 loop (rebasing, bailout, smooth count), in f64, with
    /// or without the table. `off` is the pixel's dc (set plane) or z
    /// offset (Julia). `None` when the pixel didn't escape. `round_f32`
    /// rounds the delta to f32 after every plain step, like the GPU does.
    fn emulate(
        orb: &RefOrbit,
        t: Option<&BlaTable>,
        map: &StepMap,
        off: C,
        max_iter: usize,
        round_f32: bool,
    ) -> Option<Escape> {
        let x = |m: usize| (orb.points[m][0] as f64, orb.points[m][1] as f64);
        let (step_add, mut e) = if map.julia {
            ((0.0, 0.0), off)
        } else {
            (off, (0.0, 0.0))
        };
        let (mut m, mut n, mut jumped) = (0usize, 0usize, 0usize);
        let x0 = x(0);
        loop {
            let z = add(x(m), e);
            if mag2(z) > 1e6 {
                let smooth = n as f64 - (0.5 * mag2(z).ln()).log2();
                return Some(Escape { smooth, jumped });
            }
            if n >= max_iter {
                return None;
            }
            let hit = t.filter(|_| m != 0).and_then(|t| {
                let budget = (max_iter - n).min(orb.len() - 1 - m);
                lookup(t, m, 0.5 * mag2(e).log2(), budget)
            });
            if let Some((idx, k)) = hit {
                e = apply(&t.unwrap().nodes[idx], e, step_add);
                m += k;
                n += k;
                jumped += k;
            } else {
                e = add(delta(map, x(m), e), step_add);
                if round_f32 {
                    e = (e.0 as f32 as f64, e.1 as f32 as f64);
                }
                m += 1;
                n += 1;
            }
            if m >= orb.len() {
                return Some(Escape {
                    smooth: n as f64,
                    jumped,
                });
            }
            let z = add(x(m), e);
            if mag2(z) < mag2(e) {
                e = (z.0 - x0.0, z.1 - x0.1);
                m = 0;
            }
        }
    }

    /// Escape iteration of `c` from `z0` iterated directly (in f64).
    fn escape_time(map: &StepMap, z0: C, c: C, max: usize) -> usize {
        let mut z = z0;
        for n in 0..max {
            if mag2(z) > 1e6 {
                return n;
            }
            z = add(delta(map, (0.0, 0.0), z), c); // f(0) = 0 for every kind
        }
        max
    }

    /// A boundary point of `map`'s default view (escaping late, i.e.
    /// chaotic nearby): the most iterations under `max / 2`.
    fn boundary_point(map: &StepMap, julia_c: Option<C>, max: usize) -> C {
        let (cx, cy, hh) = map.kind.default_set_view();
        let (cx, cy, hh) = if julia_c.is_some() {
            (0.0, 0.0, 1.5)
        } else {
            (cx, cy, hh)
        };
        let mut best = ((cx, cy), 0);
        let k = 97;
        for i in 0..k {
            for j in 0..k {
                let p = (
                    cx + hh * (2.0 * i as f64 / k as f64 - 1.0) * 1.3,
                    cy + hh * (2.0 * j as f64 / k as f64 - 1.0),
                );
                let n = match julia_c {
                    Some(c) => escape_time(map, p, c, max),
                    None => escape_time(map, (0.0, 0.0), p, max),
                };
                if n < max / 2 && n > best.1 {
                    best = (p, n);
                }
            }
        }
        assert!(best.1 > 50, "{:?}: no boundary point found", map.kind);
        best.0
    }

    /// Compare jumping against plain f64 stepping on 40 pixels around the
    /// reference. Chaotic pixels amplify any error, so a few escape counts
    /// differ whatever the method; BLA passes if it mismatches no more
    /// pixels than plain stepping with the GPU's f32 rounding does (and
    /// actually jumps). Returns the steps jumped.
    fn check_pixels(orb: &RefOrbit, t: &BlaTable, m: &StepMap, half: f64, max: usize) -> usize {
        let (mut jumped, mut bad_bla, mut bad_f32) = (0, 0, 0);
        let mut worst = String::new();
        for i in 0..40 {
            let a = i as f64 * 0.37;
            let off = (half * a.cos(), half * (1.3 * a).sin());
            let plain = emulate(orb, None, m, off, max, false).map(|e| e.smooth);
            let f32ish = emulate(orb, None, m, off, max, true).map(|e| e.smooth);
            let fast = emulate(orb, Some(t), m, off, max, false);
            if let Some(f) = &fast {
                jumped += f.jumped;
            }
            let fast = fast.map(|e| e.smooth);
            let differs = |x: Option<f64>| match (plain, x) {
                (Some(a), Some(b)) => (a - b).abs() >= 0.05,
                (a, b) => a.is_some() != b.is_some(),
            };
            if differs(fast) {
                bad_bla += 1;
                worst = format!("off {off:?}: plain {plain:?}, bla {fast:?}");
            }
            bad_f32 += differs(f32ish) as usize;
        }
        assert!(
            bad_bla <= bad_f32.max(1),
            "{:?} julia={}: {bad_bla} pixels differ with BLA, {bad_f32} with f32 \
             rounding (last: {worst})",
            m.kind,
            m.julia
        );
        jumped
    }

    /// For every kind BLA covers: jumping must reproduce the plain
    /// perturbation loop's smooth escape counts on a chaotic view, and must
    /// actually jump.
    #[test]
    fn escape_counts_match_all_kinds() {
        let max = 4000;
        let mut cases: Vec<(StepMap, Option<C>)> = FractalKind::ALL
            .iter()
            .filter(|&&k| k != FractalKind::Phoenix)
            .map(|&k| (map(k, false), None))
            .collect();
        cases.push((map(FractalKind::Mandelbrot, true), Some((-0.8, 0.156))));
        cases.push((map(FractalKind::BurningShip, true), Some((-0.6, -0.9))));
        for (m, julia_c) in cases {
            let p = boundary_point(&m, julia_c, max);
            let orb = match julia_c {
                Some(c) => orbit(&m, p, c, max as u32),
                None => orbit(&m, (0.0, 0.0), p, max as u32),
            };
            let half = 1e-15f64;
            let t = build(
                &orb,
                orb.len(),
                (2.0 * half).log2().ceil() as i32 + 1,
                1e6,
                &m,
            );
            let jumped = check_pixels(&orb, &t, &m, half, max);
            assert!(jumped > 0, "{:?} julia={}: never jumped", m.kind, m.julia);
        }
    }

    /// Jumping with the table must track the plain perturbation iteration.
    #[test]
    fn jumps_match_stepping() {
        // A point inside the main cardioid: the orbit never escapes, so the
        // reference is long and smooth, and the deltas stay small.
        let m = map(FractalKind::Mandelbrot, false);
        let orb = mandel_orbit(-0.1, 0.2, 5000);
        assert_eq!(orb.len(), 5001);
        let dc_max_log2 = -40;
        let t = build(&orb, orb.len(), dc_max_log2, 1e6, &m);
        assert!(t.levels() > 3);
        let x = |k: usize| (orb.points[k][0] as f64, orb.points[k][1] as f64);
        let mut jumped = 0;
        for &(dr, di) in &[(0.7, 0.1), (-0.3, 0.9), (0.0, -1.0)] {
            let dc = (dr * 2f64.powi(dc_max_log2), di * 2f64.powi(dc_max_log2));
            let mut e_ref = (0.0, 0.0);
            let mut plain = vec![e_ref];
            for k in 0..orb.len() - 1 {
                e_ref = add(delta(&m, x(k), e_ref), dc);
                plain.push(e_ref);
            }
            let (mut k, mut e) = (0usize, (0.0f64, 0.0f64));
            while k < orb.len() - 1 {
                let budget = orb.len() - 1 - k;
                if let Some((idx, n)) = lookup(&t, k, 0.5 * mag2(e).log2(), budget) {
                    e = apply(&t.nodes[idx], e, dc);
                    k += n;
                    jumped += n;
                } else {
                    e = add(delta(&m, x(k), e), dc);
                    k += 1;
                }
                let want = plain[k];
                let err = mag2((e.0 - want.0, e.1 - want.1)).sqrt();
                assert!(
                    err <= 1e-4 * mag2(want).sqrt() + 1e-300,
                    "k={k}: {e:?} vs {want:?}"
                );
            }
        }
        assert!(jumped > (orb.len() - 1) / 2, "only {jumped} steps jumped");
    }

    /// The seahorse case that set ε: the smooth counts with and without
    /// jumps must agree.
    #[test]
    fn escape_counts_match() {
        let m = map(FractalKind::Mandelbrot, false);
        let orb = mandel_orbit(-0.743_643_887_037_151, 0.131_825_904_205_330, 20_000);
        let half = 1e-15f64;
        let t = build(
            &orb,
            orb.len(),
            (2.0 * half).log2().ceil() as i32 + 1,
            1e6,
            &m,
        );
        let jumped = check_pixels(&orb, &t, &m, half, 20_000);
        assert!(jumped > 0);
    }

    /// The spectral norm is exact on conformal maps and folds of them.
    #[test]
    fn norm_is_exact() {
        let a = Fx::new(0.3, -0.8, 5);
        let want = a.abs().log2();
        for m in [
            Mx::complex(a),
            Mx::complex(a).fold(1.0, -1.0),
            Mx::complex(a).fold(-1.0, 1.0),
        ] {
            assert!((m.norm().log2() - want).abs() < 1e-12);
        }
        let diag = Mx::new([4.0, 0.0, 0.0, 0.25], 0);
        assert!((diag.norm().log2() - 2.0).abs() < 1e-12);
    }

    /// A table built for a small dc_max must reject deltas it can't cover.
    #[test]
    fn radius_shrinks_with_dc_max() {
        let m = map(FractalKind::Mandelbrot, false);
        let orb = mandel_orbit(-0.1, 0.2, 2000);
        let small = build(&orb, orb.len(), -60, 1e6, &m);
        let big = build(&orb, orb.len(), -30, 1e6, &m); // dc above the ~2^-30.5 step radius
        assert!(big.nodes[0].r_log2 < small.nodes[0].r_log2);
    }

    #[test]
    fn layout() {
        let m = map(FractalKind::Mandelbrot, false);
        let orb = mandel_orbit(-0.1, 0.2, 1000);
        let t = build(&orb, orb.len(), -30, 1e6, &m);
        let steps = orb.len() - 2;
        assert_eq!(t.meta[0], MIN_LEVEL);
        for k in 0..t.levels() as usize {
            let count = t.meta[3 + k] - t.meta[2 + k];
            assert_eq!(count as usize, steps >> (MIN_LEVEL as usize + k));
        }
        let (levels, sc) = (t.levels() as usize, 3 + t.levels() as usize);
        let seg_count = t.meta[sc] as usize;
        assert_eq!(t.meta.len(), sc + 1 + seg_count);
        assert_eq!(t.meta[2 + levels] as usize + seg_count, t.nodes.len());
        // No bad step in the cardioid: one segment over every step.
        assert_eq!(seg_count, 1);
        assert_eq!(
            (t.meta[sc + 1], t.nodes.last().unwrap().steps),
            (1, steps as u32)
        );
        let tiny = mandel_orbit(-3.0, 0.0, 1000); // escapes at once
        assert_eq!(build(&tiny, tiny.len(), -30, 1e6, &m).levels(), 0);
    }
}
