//! Static validation of the WGSL shaders. `cargo build` does not type-check
//! WGSL (that happens at pipeline creation), so this parses and validates each
//! shader with the same `naga` version wgpu uses — catching shader errors
//! without needing a GPU or a display.

fn validate(name: &str, src: &str) -> (naga::Module, naga::valid::ModuleInfo) {
    let module = match naga::front::wgsl::parse_str(src) {
        Ok(m) => m,
        Err(e) => panic!("{name}: WGSL parse error:\n{}", e.emit_to_string(src)),
    };
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    match validator.validate(&module) {
        Ok(info) => (module, info),
        Err(e) => panic!("{name}: WGSL validation error:\n{}", e.emit_to_string(src)),
    }
}

/// Number of fractal kinds, i.e. the `const KIND_*` declarations in
/// common.wgsl (one per `FractalKind` variant, values 0..N).
fn kind_count() -> u32 {
    let n = include_str!("../src/shaders/common.wgsl")
        .lines()
        .filter(|l| l.starts_with("const KIND_"))
        .count() as u32;
    assert!(n >= 10, "found only {n} KIND_* constants in common.wgsl");
    n
}

/// Specialize `module`'s `override`s with `constants` for `entry_point` (as
/// wgpu does at pipeline creation) and compile the result to SPIR-V, so a
/// shader that only breaks once a particular override value folds a branch
/// in or out is still caught.
fn specialize(
    name: &str,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    stage: naga::ShaderStage,
    entry_point: &str,
    constants: &[(&str, f64)],
) {
    let mut pc = naga::back::PipelineConstants::default();
    for (k, v) in constants {
        pc.insert((*k).to_string(), *v);
    }
    let (module, info) = naga::back::pipeline_constants::process_overrides(
        module,
        info,
        Some((stage, entry_point)),
        &pc,
    )
    .unwrap_or_else(|e| panic!("{name} {entry_point} {constants:?}: override error: {e:?}"));
    let pipeline = naga::back::spv::PipelineOptions {
        shader_stage: stage,
        entry_point: entry_point.to_string(),
    };
    naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        Some(&pipeline),
    )
    .unwrap_or_else(|e| panic!("{name} {entry_point} {constants:?}: SPIR-V error: {e:?}"));
}

/// `mandelbrot.wgsl` as `GpuPath::Storage` builds it (renderer.rs's
/// `GpuPath::mandelbrot_source`).
const MANDELBROT_SRC: &str = concat!(
    include_str!("../src/shaders/common.wgsl"),
    include_str!("../src/shaders/iterate_uniforms.wgsl"),
    include_str!("../src/shaders/mandelbrot.wgsl"),
    include_str!("../src/shaders/data_storage.wgsl"),
);

/// `mandelbrot.wgsl` as `GpuPath::Texture` (WebGL2) builds it.
const MANDELBROT_TEXTURE_SRC: &str = concat!(
    include_str!("../src/shaders/common.wgsl"),
    include_str!("../src/shaders/iterate_uniforms.wgsl"),
    include_str!("../src/shaders/mandelbrot.wgsl"),
    include_str!("../src/shaders/data_texture.wgsl"),
);

#[test]
fn mandelbrot_shader_is_valid() {
    validate("mandelbrot.wgsl", MANDELBROT_SRC);
    validate("mandelbrot.wgsl (texture)", MANDELBROT_TEXTURE_SRC);
}

/// Every `PipelineKey` renderer.rs can build: kind × Julia × DE × morph ×
/// deep × BLA, as override constants.
fn pipeline_keys() -> Vec<[(&'static str, f64); 6]> {
    let mut keys = Vec::new();
    for kind in 0..kind_count() {
        for julia in [0.0, 1.0] {
            for de in [0.0, 1.0] {
                for morph in [0.0, 1.0] {
                    for deep in [0.0, 1.0] {
                        // BLA: every kind but Phoenix (7), no morph (`bla::applies`).
                        let blas: &[f64] = if kind != 7 && morph == 0.0 {
                            &[0.0, 1.0]
                        } else {
                            &[0.0]
                        };
                        for &bla in blas {
                            keys.push([
                                ("KIND", kind as f64),
                                ("IS_JULIA", julia),
                                ("DE", de),
                                ("MORPH", morph),
                                ("DEEP", deep),
                                ("BLA", bla),
                            ]);
                        }
                    }
                }
            }
        }
    }
    keys
}

/// Every specialization renderer.rs can build, for every fragment entry
/// point, on both data paths.
#[test]
fn mandelbrot_shader_specializations_compile() {
    for (name, src) in [
        ("mandelbrot.wgsl", MANDELBROT_SRC),
        ("mandelbrot.wgsl (texture)", MANDELBROT_TEXTURE_SRC),
    ] {
        let (module, info) = validate(name, src);
        for constants in pipeline_keys() {
            for entry in ["fs_data", "fs_refine", "fs_color"] {
                specialize(
                    name,
                    &module,
                    &info,
                    naga::ShaderStage::Fragment,
                    entry,
                    &constants,
                );
            }
        }
    }
}

/// Translate `entry_point` to GLSL ES 3.00 as wgpu's GL backend does on
/// WebGL2 (`wgpu-hal`'s gles `create_pipeline`), and, when
/// `glslangValidator` is on the PATH, compile the result. naga's ES 3.00
/// output can call builtins ES 3.00 lacks (`ldexp`, `findLSB`), which only
/// a real GLSL compiler (or the browser) rejects.
fn to_gles300(
    name: &str,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    stage: naga::ShaderStage,
    entry_point: &str,
    constants: &[(&str, f64)],
) -> String {
    let mut pc = naga::back::PipelineConstants::default();
    for (k, v) in constants {
        pc.insert((*k).to_string(), *v);
    }
    let what = format!("{name} {entry_point} {constants:?}");
    let (module, info) = naga::back::pipeline_constants::process_overrides(
        module,
        info,
        Some((stage, entry_point)),
        &pc,
    )
    .unwrap_or_else(|e| panic!("{what}: override error: {e:?}"));
    let options = naga::back::glsl::Options {
        version: naga::back::glsl::Version::Embedded {
            version: 300,
            is_webgl: true,
        },
        ..Default::default()
    };
    let pipeline = naga::back::glsl::PipelineOptions {
        shader_stage: stage,
        entry_point: entry_point.to_string(),
        multiview: None,
    };
    let mut out = String::new();
    naga::back::glsl::Writer::new(
        &mut out,
        &module,
        &info,
        &options,
        &pipeline,
        naga::proc::BoundsCheckPolicies::default(),
    )
    .and_then(|mut w| w.write())
    .unwrap_or_else(|e| panic!("{what}: GLSL error: {e:?}"));
    glslang_check(&what, stage, &out);
    out
}

/// Compile GLSL `src` with `glslangValidator`, if installed.
fn glslang_check(what: &str, stage: naga::ShaderStage, src: &str) {
    use std::io::Write as _;
    let stage = match stage {
        naga::ShaderStage::Vertex => "vert",
        naga::ShaderStage::Fragment => "frag",
        _ => unreachable!("WebGL2 has no other stages"),
    };
    let child = std::process::Command::new("glslangValidator")
        .args(["--stdin", "-S", stage])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();
    let Ok(mut child) = child else {
        return; // not installed: naga's translation is all we can check
    };
    child
        .stdin
        .take()
        .unwrap()
        .write_all(src.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{what}: GLSL ES 3.00 doesn't compile:\n{}\n{src}",
        String::from_utf8_lossy(&out.stdout)
    );
}

/// The WebGL2 build's shaders (`GpuPath::Texture`, plus the passes shared
/// with WebGPU) translate to GLSL ES 3.00 that compiles, for every
/// specialization.
#[test]
fn webgl_shaders_compile_to_gles300() {
    let (module, info) = validate("mandelbrot.wgsl (texture)", MANDELBROT_TEXTURE_SRC);
    let keys = pipeline_keys();
    // One thread per chunk of keys: glslangValidator runs once per shader.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|scope| {
        for chunk in keys.chunks(keys.len().div_ceil(threads)) {
            let (module, info) = (&module, &info);
            scope.spawn(move || {
                for constants in chunk {
                    for entry in ["fs_data", "fs_refine", "fs_color"] {
                        to_gles300(
                            "mandelbrot.wgsl (texture)",
                            module,
                            info,
                            naga::ShaderStage::Fragment,
                            entry,
                            constants,
                        );
                    }
                }
            });
        }
    });
    to_gles300(
        "mandelbrot.wgsl (texture)",
        &module,
        &info,
        naga::ShaderStage::Vertex,
        "vs_main",
        &[],
    );

    let others: [(&str, &str, &[&str]); 4] = [
        (
            "colorize.wgsl",
            concat!(
                include_str!("../src/shaders/common.wgsl"),
                include_str!("../src/shaders/iterate_uniforms.wgsl"),
                include_str!("../src/shaders/colorize.wgsl"),
            ),
            &["fs_main"],
        ),
        (
            "lipschitz.wgsl",
            concat!(
                include_str!("../src/shaders/common.wgsl"),
                include_str!("../src/shaders/lipschitz.wgsl"),
            ),
            &["fs_seed", "fs_jump", "fs_compose"],
        ),
        (
            "blit.wgsl",
            concat!(
                include_str!("../src/shaders/common.wgsl"),
                include_str!("../src/shaders/blit.wgsl"),
            ),
            &["fs_main"],
        ),
        ("ci_sample.wgsl", CI_SAMPLE_SRC, &["fs_main"]),
    ];
    for (name, src, entries) in others {
        let (module, info) = validate(name, src);
        to_gles300(
            name,
            &module,
            &info,
            naga::ShaderStage::Vertex,
            "vs_main",
            &[],
        );
        for entry in entries {
            to_gles300(
                name,
                &module,
                &info,
                naga::ShaderStage::Fragment,
                entry,
                &[],
            );
        }
    }
}

/// The texture path's bit-level `frexp`/`ldexp`/`ctz` replacements must not
/// fall back on the builtins GLSL ES 3.00 lacks or polyfills badly (naga's
/// `frexp` polyfill goes through log2).
#[test]
fn texture_path_avoids_es31_builtins() {
    let (module, info) = validate("mandelbrot.wgsl (texture)", MANDELBROT_TEXTURE_SRC);
    let constants = [
        ("KIND", 0.0),
        ("IS_JULIA", 0.0),
        ("DE", 1.0),
        ("MORPH", 0.0),
        ("DEEP", 1.0),
        ("BLA", 1.0),
    ];
    let glsl = to_gles300(
        "mandelbrot.wgsl (texture)",
        &module,
        &info,
        naga::ShaderStage::Fragment,
        "fs_data",
        &constants,
    );
    for builtin in ["ldexp(", "frexp(", "findLSB("] {
        assert!(!glsl.contains(builtin), "texture path GLSL calls {builtin}");
    }
}

#[test]
fn colorize_shader_is_valid() {
    validate(
        "colorize.wgsl",
        concat!(
            include_str!("../src/shaders/common.wgsl"),
            include_str!("../src/shaders/iterate_uniforms.wgsl"),
            include_str!("../src/shaders/colorize.wgsl"),
        ),
    );
}

#[test]
fn lipschitz_shader_is_valid() {
    validate(
        "lipschitz.wgsl",
        concat!(
            include_str!("../src/shaders/common.wgsl"),
            include_str!("../src/shaders/lipschitz.wgsl"),
        ),
    );
}

#[test]
fn blit_shader_is_valid() {
    validate(
        "blit.wgsl",
        concat!(
            include_str!("../src/shaders/common.wgsl"),
            include_str!("../src/shaders/blit.wgsl"),
        ),
    );
}

const BUDDHABROT_SRC: &str = concat!(
    include_str!("../src/shaders/common.wgsl"),
    include_str!("../src/shaders/buddhabrot.wgsl"),
);

#[test]
fn buddhabrot_shader_is_valid() {
    validate("buddhabrot.wgsl", BUDDHABROT_SRC);
}

/// Every per-kind accumulation pipeline buddhabrot.rs can build.
#[test]
fn buddhabrot_shader_specializations_compile() {
    let (module, info) = validate("buddhabrot.wgsl", BUDDHABROT_SRC);
    for kind in 0..kind_count() {
        specialize(
            "buddhabrot.wgsl",
            &module,
            &info,
            naga::ShaderStage::Compute,
            "cs_main",
            &[("KIND", kind as f64)],
        );
    }
}

const CI_SAMPLE_SRC: &str = concat!(
    include_str!("../src/shaders/common.wgsl"),
    include_str!("../src/shaders/ci_sample.wgsl"),
);

#[test]
fn ci_sample_shader_is_valid() {
    validate("ci_sample.wgsl", CI_SAMPLE_SRC);
}

#[test]
fn ci_stats_shader_is_valid() {
    validate(
        "ci_stats.wgsl",
        include_str!("../src/shaders/ci_stats.wgsl"),
    );
}
