//! Compiles every shader in the tree on the real device.
//!
//! The point is timing: `naga` validation at startup, not at the moment a pass is first recorded.
//! A shader error found here is a five-second fix; the same error found by launching the app is a
//! broken-looking window and a hunt through the include graph.
//!
//! This also catches the failure mode the preprocessor makes possible: an include that resolves to a
//! syntactically valid file which, once inlined, produces a duplicate declaration or a use-before-
//! declaration. Only compiling the composed unit finds that.

use std::path::PathBuf;

use boids_gpu::context::GpuContext;

use crate::common::Check;

/// Recursively collects every `.wgsl` file under the shader root.
fn shader_files(root: &std::path::Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) => panic!("reading {}: {e}", dir.display()),
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "wgsl") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// Compiles each shader as its own compilation unit.
///
/// Compiling each file standalone (rather than only the entry points) is deliberate: a header with a
/// syntax error would otherwise only be caught if some entry point happens to include it, and the
/// point of this check is to cover the whole tree.
pub fn compile_all(ctx: &GpuContext) -> Check {
    let root = ctx.shaders.root().to_path_buf();
    let files = shader_files(&root);
    if files.is_empty() {
        return Err(format!("no .wgsl files found under {}", root.display()));
    }

    let mut compiled = 0usize;
    let mut problems = Vec::new();
    for path in &files {
        let relative = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        // `shaders/tests/cycle_*.wgsl` are deliberate negative fixtures: they exist so the
        // preprocessor's cycle detection has something to detect. Compiling them here would be
        // asserting that broken input compiles, so they are skipped by name.
        if relative.contains("tests/cycle_") {
            continue;
        }
        // `ctx.shader_module` panics on failure, which for a test run is exactly the wrong behaviour:
        // it hides the other files. Compile through the loader so failures accumulate.
        match ctx.shaders.compile(&relative) {
            Err(e) => problems.push(format!("{relative}: {e}")),
            Ok(unit) => {
                if unit.source.contains("//#include") {
                    problems.push(format!("{relative}: unresolved include left in the output"));
                    continue;
                }
                // `create_shader_module` is where naga validates. wgpu reports validation failures
                // through the error scope, and our uncaptured-error handler turns those into a panic,
                // so a failure here aborts the suite with the WGSL error text.
                let _ = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&relative),
                    source: wgpu::ShaderSource::Wgsl(unit.source.into()),
                });
                compiled += 1;
            }
        }
    }

    if problems.is_empty() {
        println!("\n    compiled {compiled} shader files");
        Ok(())
    } else {
        Err(problems.join("\n    "))
    }
}
