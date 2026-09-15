//! Shared helpers for the off-screen render checks.

#![allow(dead_code)]

use boids_gpu::context::{GpuContext, GpuContextDescriptor};

/// Render target size. Small on purpose: the checks are about whether content exists, not about
/// fidelity, and a software adapter renders 400x300 far faster than 2560x1440.
pub const WIDTH: u32 = 400;
/// Render target height.
pub const HEIGHT: u32 = 300;

/// Off-screen format. sRGB so the bytes read back the way the window would show them.
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// A check that either passes or reports why it did not.
pub type Check = Result<(), String>;

/// Creates the device the whole suite shares.
///
/// Prefers a real GPU, falls back to a software adapter so the suite runs on a machine with no usable
/// GPU. `BOIDS_TEST_REQUIRE_GPU=1` refuses the fallback.
pub fn test_context() -> GpuContext {
    let require_gpu = std::env::var("BOIDS_TEST_REQUIRE_GPU").is_ok();
    match GpuContext::new(&GpuContextDescriptor {
        high_performance: true,
        want_timestamps: true,
        force_backends: None,
        force_fallback: false,
    }) {
        Ok(ctx) => {
            if require_gpu && is_software(&ctx) {
                panic!("BOIDS_TEST_REQUIRE_GPU is set but the adapter is {}", ctx.info.name);
            }
            ctx
        }
        Err(e) if require_gpu => panic!("BOIDS_TEST_REQUIRE_GPU is set but no device is available: {e}"),
        Err(e) => {
            eprintln!("note: no hardware adapter ({e}); falling back to a software adapter");
            GpuContext::new(&GpuContextDescriptor {
                high_performance: false,
                want_timestamps: true,
                force_backends: None,
                force_fallback: true,
            })
            .unwrap_or_else(|e| panic!("no usable GPU device, hardware or software: {e}"))
        }
    }
}

/// Whether the adapter is a software rasteriser.
pub fn is_software(ctx: &GpuContext) -> bool {
    let name = ctx.info.name.to_ascii_lowercase();
    ctx.info.device_type == wgpu::DeviceType::Cpu
        || name.contains("llvmpipe")
        || name.contains("lavapipe")
        || name.contains("swiftshader")
}
