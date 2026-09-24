// Buddhabrot / Nebulabrot rendering: a Monte-Carlo density histogram of
// escaping orbits, accumulated progressively across frames by a compute pass,
// then tone-mapped to colour by a fragment pass every frame.
//
// This does NOT use the deep-zoom perturbation/reference-orbit machinery in
// mandelbrot.wgsl: Buddhabrot's structure is a global Monte-Carlo property of
// the whole basin (a random sample's orbit scatters across the *whole* image,
// not just its own pixel), so the "gather" per-pixel model doesn't apply, and
// deep zoom isn't meaningful for it the way it is for the escape-time set.
// Samples are iterated directly in f32 from the current view's bounds.
//
// Sampling convention: for KIND_LAMBDA the formula z -> l*z*(1-z) has no `c`
// term at all (l is a fixed distortion constant, not a per-sample parameter),
// so the randomly sampled point instead seeds z0 (a "Julia-Buddhabrot" over
// z0 with l fixed). Every other kind samples c with z0 = 0, matching its
// ordinary parameter plane.
//
// A sample's orbit is only plotted if it escapes within b_cap iterations (the
// classic Buddhabrot rule: only escaping orbits are drawn). Its points are
// then splat into up to three histogram channels by cap (r_cap <= g_cap <=
// b_cap): fast-escaping (common) orbits light all three channels (bright),
// slow-escaping (rare) orbits only light the b_cap channel — the classic
// Nebulabrot false-colour split.
//
// Two-pass iteration avoids needing a per-thread orbit buffer sized to
// max_iter: the first pass just finds the escape iteration (if any); the
// second replays the same orbit from scratch, splatting each point.

struct Uniforms {
    center: vec2<f32>,
    half_height: f32,
    aspect: f32,
    phoenix_p: vec2<f32>,
    lambda_l: vec2<f32>,
    bailout_sq: f32,
    kind: u32,
    power: u32,
    r_cap: u32,
    g_cap: u32,
    b_cap: u32,
    seed: u32,
    samples_this_dispatch: u32,
    exposure: f32,
    width: u32,
    height: u32,
    total_samples: f32,
    // Tonemap colour style: 0 = classic (R/G/B = raw caps), 1 = nebula
    // (yellow core, blue halo), 2 = grayscale.
    palette: u32,
    // Padding so `complex_power` (a vec2, 8-byte aligned) starts on an
    // 8-byte boundary. NOT vec3<u32> — that type aligns to 16 bytes in WGSL
    // (unlike Rust's `[u32; 3]`, which aligns to 4), which silently added 32
    // bytes instead of 16 and mismatched the Rust struct's size (a wgpu
    // validation error at dispatch time: "size 96 where the shader expects
    // 112").
    _pad0: u32,
    // Complex exponent for the Complex Multibrot kind; unused by other kinds.
    complex_power: vec2<f32>,
};

// Fractal kind, as a pipeline-overridable constant (set per compute pipeline
// from `u.kind`, see `BuddhabrotRenderer::compute_pipeline`): every kind
// branch in the iteration loop folds away at pipeline creation. Read this,
// never `u.kind`.
override KIND: u32 = 0u;

const PALETTE_NEBULA: u32 = 0u;
const PALETTE_YELLOW: u32 = 1u;
const PALETTE_GRAYSCALE: u32 = 2u;

@group(0) @binding(0) var<uniform> u: Uniforms;
// Compute pass: read-write atomic histogram (3 planes of width*height, R/G/B).
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>>;
// Tonemap pass: read-only plain view of the same buffer.
@group(0) @binding(2) var<storage, read> tm_histogram: array<u32>;

// --- RNG: a small, fast integer hash (WGSL has no native RNG). ---
fn hash_u32(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}
fn rand01(seed: u32) -> f32 {
    return f32(hash_u32(seed)) * (1.0 / 4294967295.0);
}

fn complex_pow(z: vec2<f32>, p: u32) -> vec2<f32> {
    var r = vec2<f32>(1.0, 0.0);
    for (var i: u32 = 0u; i < p; i = i + 1u) {
        r = cmul(r, z);
    }
    return r;
}

// One iteration step z_n -> z_{n+1} for the current kind. `zp` is the
// previous iterate (z_{n-1}), used only by the Phoenix two-term recurrence.
// Must match `FractalKind` in reference.rs (the direct, non-perturbative form
// of the same formulas).
fn advance(z: vec2<f32>, zp: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
    if KIND == KIND_BURNING_SHIP {
        return vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * abs(z.x * z.y)) + c;
    } else if KIND == KIND_TRICORN {
        return vec2<f32>(z.x * z.x - z.y * z.y, -2.0 * z.x * z.y) + c;
    } else if KIND == KIND_MULTIBROT {
        return complex_pow(z, clamp(u.power, 2u, 8u)) + c;
    } else if KIND == KIND_CELTIC {
        return vec2<f32>(abs(z.x * z.x - z.y * z.y), 2.0 * z.x * z.y) + c;
    } else if KIND == KIND_PERPENDICULAR {
        return vec2<f32>(z.x * z.x - z.y * z.y, -2.0 * z.x * abs(z.y)) + c;
    } else if KIND == KIND_BUFFALO {
        return vec2<f32>(abs(z.x * z.x - z.y * z.y), -abs(2.0 * z.x * z.y)) + c;
    } else if KIND == KIND_PHOENIX {
        let sq = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
        return sq + c + cmul(u.phoenix_p, zp);
    } else if KIND == KIND_LAMBDA {
        // l * z * (1 - z); c is unused (see file doc comment above).
        return cmul(u.lambda_l, cmul(z, vec2<f32>(1.0 - z.x, -z.y)));
    } else if KIND == KIND_COMPLEX_MULTIBROT {
        return cpow(z, u.complex_power) + c;
    }
    return vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c; // Mandelbrot
}

// Map a complex-plane point to a flat pixel index, or -1 if outside the
// current viewport (the sampling region and the display region are the same).
//
// This must be the exact inverse of how `view.rs::pan_pixels`/`zoom_at_pixel`
// relate screen pixels to world points (those are the confirmed-correct,
// user-tested ground truth — NOT the shader-comment-derived convention tried
// here previously, which was wrong: dragging/zooming treat +y screen exactly
// like +x, no flip, so screen-down means im *increasing*, not decreasing).
fn pixel_index(p: vec2<f32>) -> i32 {
    let half_w = u.half_height * u.aspect;
    let uu = (p.x - u.center.x) / half_w * 0.5 + 0.5;
    let vv = 0.5 + (p.y - u.center.y) / u.half_height * 0.5;
    if uu < 0.0 || uu >= 1.0 || vv < 0.0 || vv >= 1.0 {
        return -1;
    }
    let px = i32(uu * f32(u.width));
    let py = i32(vv * f32(u.height));
    return py * i32(u.width) + px;
}

// Splat one visited orbit point into the R/G/B histogram planes it qualifies
// for by the orbit's total escape iteration `n` (nested caps: a fast escape
// lights all three; only a slow, rare one lights just the blue plane).
fn splat(p: vec2<f32>, n: u32) {
    let idx = pixel_index(p);
    if idx < 0 {
        return;
    }
    let plane = i32(u.width) * i32(u.height);
    if n <= u.b_cap {
        atomicAdd(&histogram[idx + 2 * plane], 1u);
    }
    if n <= u.g_cap {
        atomicAdd(&histogram[idx + plane], 1u);
    }
    if n <= u.r_cap {
        atomicAdd(&histogram[idx], 1u);
    }
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.samples_this_dispatch {
        return;
    }

    let base = hash_u32(gid.x ^ (u.seed * 0x9e3779b9u));
    let rx = rand01(base);
    let ry = rand01(hash_u32(base ^ 0x68bc21ebu));
    let half_w = u.half_height * u.aspect;
    let sample = vec2<f32>(
        u.center.x + (rx * 2.0 - 1.0) * half_w,
        u.center.y + (ry * 2.0 - 1.0) * u.half_height,
    );

    var c = sample;
    var z0 = vec2<f32>(0.0, 0.0);
    if KIND == KIND_LAMBDA {
        c = vec2<f32>(0.0, 0.0); // unused by the Lambda step
        z0 = sample;
    }

    // First pass: just find the escape iteration (if any).
    var zp = vec2<f32>(0.0, 0.0);
    var z = z0;
    var n: u32 = 0u;
    var escaped = false;
    loop {
        if dot(z, z) > u.bailout_sq {
            escaped = true;
            break;
        }
        if n >= u.b_cap {
            break;
        }
        let next = advance(z, zp, c);
        zp = z;
        z = next;
        n = n + 1u;
    }
    if !escaped || n == 0u {
        return;
    }

    // Second pass: replay the same orbit, splatting each visited point.
    // z0 itself is not splat: it's the same fixed point (0,0), or the sample
    // itself for Lambda, for every orbit — plotting it would just spike the
    // origin instead of showing the orbit's actual shape.
    zp = vec2<f32>(0.0, 0.0);
    z = z0;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let next = advance(z, zp, c);
        zp = z;
        z = next;
        splat(z, n);
    }
}

// --- Tonemap: histogram counts -> colour, drawn as a fullscreen triangle. ---

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(fullscreen_triangle_pos(idx), 0.0, 1.0);
}

@fragment
fn fs_tonemap(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let x = i32(pos.x);
    let y = i32(pos.y);
    if x < 0 || y < 0 || x >= i32(u.width) || y >= i32(u.height) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let idx = y * i32(u.width) + x;
    let plane = i32(u.width) * i32(u.height);
    let r = f32(tm_histogram[idx]);
    let g = f32(tm_histogram[idx + plane]);
    let b = f32(tm_histogram[idx + 2 * plane]);

    // Normalize by the *average* density (total samples / pixel count) rather
    // than total samples alone, so the scale stays sane across widget sizes
    // and sample-dispatch rates. Buddhabrot density is extremely peaked (the
    // brightest pixels run tens of times the average), so the compressive
    // exponential tonemap only needs a small fraction of the average to reach
    // full brightness at those peaks; 0.05 is a hand-tuned starting point,
    // the exposure slider covers the rest.
    let avg_density = max(u.total_samples / f32(u.width * u.height), 1.0e-6);
    let scale = u.exposure * 0.05 / avg_density;
    // Per-cap brightness, each already compressed to [0,1]. Nested caps mean
    // r <= g <= b pointwise (every orbit counted in a smaller cap is also
    // counted in every larger one), so fb alone is the full escaping-orbit
    // density and fr picks out just the common, fast-escaping ones.
    let fr = 1.0 - exp(-r * scale);
    let fg = 1.0 - exp(-g * scale);
    let fb = 1.0 - exp(-b * scale);

    var col: vec3<f32>;
    if u.palette == PALETTE_YELLOW {
        // fr is *not* a good stand-alone brightness signal: with c sampled
        // uniformly over the whole viewport, nearly every sample outside the
        // set escapes within a handful of iterations and splats a couple of
        // points near itself, so fr is a near-uniform wash across the entire
        // image (not concentrated near the boundary the way fb is) — adding
        // it directly (tried first, both raw and gamma-lifted) drags that
        // wash up to full brightness and floods the background with solid
        // colour. Instead use it as a *multiplicative* warm (yellow) tint on
        // top of fb's brightness, so it only shows up where fb is already
        // bright (i.e. real near-boundary density) and stays near-zero across
        // the background (fb ≈ 0 there, so warmth * fb ≈ 0 regardless of fr).
        col = vec3<f32>(
            fb + fb * fr * 1.3,
            fb + fb * fr * 0.6,
            fb,
        );
    } else if u.palette == PALETTE_GRAYSCALE {
        // fb is the full escaping-orbit density (the cumulative superset);
        // reuse it directly as a single luminance channel.
        col = vec3<f32>(fb, fb, fb);
    } else {
        col = vec3<f32>(fr, fg, fb); // classic: raw per-cap R/G/B
    }
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
