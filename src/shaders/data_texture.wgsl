// Reference orbit + BLA table as textures, and bit-exact replacements for
// float builtins: the data fragment of `mandelbrot.wgsl` on devices without
// fragment-stage storage buffers (WebGL2, `GpuPath::Texture` in
// renderer.rs). Must define the same functions as `data_storage.wgsl`.
//
// Every array is laid out row-major in a texture DATA_TEX_WIDTH texels wide
// (element i at (i % W, i / W); `RefData` in renderer.rs). BLA nodes are 3
// `Rgba32Uint` texels each (m, n, then m_exp/n_exp/r_log2/steps), floats as
// raw bits: a float texture could flush the small i32 exponents, whose bit
// patterns are subnormal floats.

@group(0) @binding(1) var ref_orbit_tex: texture_2d<f32>;
@group(0) @binding(3) var ref_exp_tex: texture_2d<i32>;
@group(0) @binding(4) var bla_nodes_tex: texture_2d<u32>;
@group(0) @binding(5) var bla_meta_tex: texture_2d<u32>;

// Must match `DATA_TEX_WIDTH` in renderer.rs (2048, WebGL2's guaranteed
// minimum texture size).
const DATA_TEX_LOG2_W: u32 = 11u;

fn data_texel(i: u32) -> vec2<i32> {
    return vec2<i32>(i32(i & ((1u << DATA_TEX_LOG2_W) - 1u)), i32(i >> DATA_TEX_LOG2_W));
}

fn ref_point(m: u32) -> vec2<f32> {
    return textureLoad(ref_orbit_tex, data_texel(m), 0).xy;
}

fn ref_exp_at(m: u32) -> i32 {
    return textureLoad(ref_exp_tex, data_texel(m), 0).x;
}

fn bla_node(i: u32) -> Bla {
    let a = textureLoad(bla_nodes_tex, data_texel(3u * i), 0);
    let b = textureLoad(bla_nodes_tex, data_texel(3u * i + 1u), 0);
    let c = textureLoad(bla_nodes_tex, data_texel(3u * i + 2u), 0);
    return Bla(
        bitcast<vec4<f32>>(a),
        bitcast<vec4<f32>>(b),
        bitcast<i32>(c.x),
        bitcast<i32>(c.y),
        bitcast<f32>(c.z),
        c.w,
    );
}

fn bla_r_log2(i: u32) -> f32 {
    return bitcast<f32>(textureLoad(bla_nodes_tex, data_texel(3u * i + 2u), 0).z);
}

fn bla_steps(i: u32) -> u32 {
    return textureLoad(bla_nodes_tex, data_texel(3u * i + 2u), 0).w;
}

fn bla_meta_at(i: u32) -> u32 {
    return textureLoad(bla_meta_tex, data_texel(i), 0).x;
}

// 2^e for e in [-126, 127], built from its bits.
fn pow2_f32(e: i32) -> f32 {
    return bitcast<f32>(u32(e + 127) << 23u);
}

// WGSL `frexp` from the exponent bits. naga's GLSL ES 3.00 polyfill goes
// through log2, which is wrong for negative x and off by one near powers of
// two. Subnormals are scaled up by 2^64 first (0 if the GPU flushes them).
fn frexp_f32(x: f32) -> Frexp {
    var bits = bitcast<u32>(x);
    var bias = 0;
    if ((bits >> 23u) & 0xffu) == 0u {
        if (bits & 0x7fffffffu) == 0u {
            return Frexp(x, 0);
        }
        bits = bitcast<u32>(x * 18446744073709551616.0); // 2^64
        bias = -64;
        if ((bits >> 23u) & 0xffu) == 0u {
            return Frexp(0.0, 0);
        }
    }
    let biased = i32((bits >> 23u) & 0xffu);
    if biased == 255 {
        return Frexp(x, 0); // inf / NaN
    }
    return Frexp(bitcast<f32>((bits & 0x807fffffu) | 0x3f000000u), biased - 126 + bias);
}

// WGSL `ldexp` (GLSL ES 3.10+) as two exact power-of-two multiplies, valid
// for |k| <= 252; with both halves the same sign, the intermediate is
// normal whenever the result is.
fn ldexp_f32(x: f32, k: i32) -> f32 {
    let k1 = clamp(k / 2, -126, 127);
    let k2 = clamp(k - k1, -126, 127);
    return x * pow2_f32(k1) * pow2_f32(k2);
}

// WGSL `countTrailingZeros` (`findLSB` is GLSL ES 3.10+): the exponent of
// the lowest set bit, which converts to f32 exactly.
fn ctz_u32(x: u32) -> u32 {
    if x == 0u {
        return 32u;
    }
    let low = x & (~x + 1u);
    return (bitcast<u32>(f32(low)) >> 23u) - 127u;
}
