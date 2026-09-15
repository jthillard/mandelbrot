//! Static validation of the WGSL shaders. `cargo build` does not type-check
//! WGSL (that happens at pipeline creation), so this parses and validates each
//! shader with the same `naga` version wgpu uses — catching shader errors
//! without needing a GPU or a display.

fn validate(name: &str, src: &str) {
    let module = match naga::front::wgsl::parse_str(src) {
        Ok(m) => m,
        Err(e) => panic!("{name}: WGSL parse error:\n{}", e.emit_to_string(src)),
    };
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    if let Err(e) = validator.validate(&module) {
        panic!("{name}: WGSL validation error:\n{}", e.emit_to_string(src));
    }
}

#[test]
fn mandelbrot_shader_is_valid() {
    validate(
        "mandelbrot.wgsl",
        include_str!("../src/shaders/mandelbrot.wgsl"),
    );
}

#[test]
fn blit_shader_is_valid() {
    validate("blit.wgsl", include_str!("../src/shaders/blit.wgsl"));
}
