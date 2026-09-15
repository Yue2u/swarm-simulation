//! Probe: does `ComputePass::set_immediates` actually reach the shader, per dispatch?
//!
//! The sort carries its stage parameters in `var<immediate>` and reuses one pipeline and one bind
//! group for 153 dispatches in a frame (see `shaders/sim/sort.wgsl`). That only works if the value
//! written by `set_immediates` is what the *next* dispatch reads, which is new behaviour in `wgpu` 30
//! and is emulated differently on every backend (push constants on Vulkan, patched uniforms on GL).
//!
//! This check exists because the failure mode is silent: if the immediates of the first pass stick
//! for the rest of the frame, the sort runs its first stage 153 times and produces an array that is
//! almost, but not quite, sorted. The sort's own check would eventually catch it, but it would report
//! a wrong key rather than a broken binding, which is a much longer walk back to the cause.

use boids_gpu::context::GpuContext;

use crate::common::Check;

/// A shader whose only input is the immediate block, so nothing else can explain the output.
const PROBE_SHADER: &str = r"
struct Probe { value: u32, pad0: u32, pad1: u32, pad2: u32 }
var<immediate> probe: Probe;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn probe_main() {
    out[0u] = probe.value;
}
";

pub fn immediates_reach_the_shader(ctx: &GpuContext) -> Check {
    if !ctx.timestamps_enabled {
        // Nothing to do with timestamps, but this check needs the same device features path, and a
        // device without IMMEDIATES never gets here: `GpuContext::new` refuses it.
        return Err("no device".to_string());
    }

    let device = &ctx.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("immediates probe"),
        source: wgpu::ShaderSource::Wgsl(PROBE_SHADER.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("immediates probe"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("immediates probe"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 16,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("immediates probe"),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("probe_main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("immediates probe"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("immediates probe"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    // Three passes, three different values, one dispatch each: exactly the shape the sort uses.
    let values = [11u32, 22, 33];
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("immediates probe"),
    });
    for value in values {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("immediates probe"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_pipeline(&pipeline);
        pass.set_immediates(0, &value.to_le_bytes());
        pass.dispatch_workgroups(1, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));

    let got: Vec<u32> = boids_gpu::transfer::read_buffer(ctx, &buffer);
    if got[0] != *values.last().unwrap() {
        return Err(format!(
            "the shader read {} after the last of {} passes with immediates {values:?}, expected \
             {}. `set_immediates` is not reaching the shader.",
            got[0],
            values.len(),
            values[values.len() - 1]
        ));
    }
    println!("\n    3 dispatches, one pipeline, immediates {values:?} -> {}", got[0]);
    Ok(())
}
