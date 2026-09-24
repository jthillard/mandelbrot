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

const MANDELBROT_SRC: &str = concat!(
    include_str!("../src/shaders/common.wgsl"),
    include_str!("../src/shaders/iterate_uniforms.wgsl"),
    include_str!("../src/shaders/mandelbrot.wgsl"),
);

#[test]
fn mandelbrot_shader_is_valid() {
    validate("mandelbrot.wgsl", MANDELBROT_SRC);
}

/// Every specialization renderer.rs can build (`PipelineKey`: kind × Julia ×
/// DE × morph), for every fragment entry point.
#[test]
fn mandelbrot_shader_specializations_compile() {
    let (module, info) = validate("mandelbrot.wgsl", MANDELBROT_SRC);
    for kind in 0..kind_count() {
        for julia in [0.0, 1.0] {
            for de in [0.0, 1.0] {
                for morph in [0.0, 1.0] {
                    let constants = [
                        ("KIND", kind as f64),
                        ("IS_JULIA", julia),
                        ("DE", de),
                        ("MORPH", morph),
                    ];
                    for entry in ["fs_data", "fs_refine", "fs_color"] {
                        specialize(
                            "mandelbrot.wgsl",
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
