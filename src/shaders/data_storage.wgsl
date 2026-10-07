// Reference orbit + BLA table as storage buffers, and the native float
// builtins: the data fragment of `mandelbrot.wgsl` on every device with
// fragment-stage storage buffers (WebGPU, Vulkan, Metal, DX12). Each
// accessor is a plain index that the shader compiler inlines. Must define
// the same functions as `data_texture.wgsl` (the WebGL2 fallback).

@group(0) @binding(1) var<storage, read> ref_orbit: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> ref_exp: array<i32>;
@group(0) @binding(4) var<storage, read> bla_nodes: array<Bla>;
@group(0) @binding(5) var<storage, read> bla_meta: array<u32>;

fn ref_point(m: u32) -> vec2<f32> {
    return ref_orbit[m];
}

fn ref_exp_at(m: u32) -> i32 {
    return ref_exp[m];
}

fn bla_node(i: u32) -> Bla {
    return bla_nodes[i];
}

fn bla_r_log2(i: u32) -> f32 {
    return bla_nodes[i].r_log2;
}

fn bla_steps(i: u32) -> u32 {
    return bla_nodes[i].steps;
}

fn bla_meta_at(i: u32) -> u32 {
    return bla_meta[i];
}

fn frexp_f32(x: f32) -> Frexp {
    let f = frexp(x);
    return Frexp(f.fract, f.exp);
}

fn ldexp_f32(x: f32, k: i32) -> f32 {
    return ldexp(x, k);
}

fn ctz_u32(x: u32) -> u32 {
    return countTrailingZeros(x);
}
