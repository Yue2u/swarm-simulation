//! Device-side validation of the CPU/WGSL struct layout contract.
//!
//! This is the most important check in the project. Every other check compares behaviour, which can
//! only fail in ways that are visible. A layout mismatch between `layout.rs` and `layout.wgsl` does
//! not fail: the simulation runs, the numbers are plausible, and the swarm is subtly wrong because
//! `SimParams::r_percept` on the host is `SimParams::w_coh` on the device.
//!
//! The mechanism is a byte ramp. The host fills each input buffer with `f32` values `1, 2, 3, ...` at
//! successive 4-byte offsets, then the probe shader copies every struct field into a flat output
//! array. If WGSL agrees with Rust about where each field lives, output slot `i` holds exactly the
//! ramp value at that field's offset. Any disagreement shows up as a specific wrong number at a
//! specific index, reported with both byte offsets so the fix is mechanical.

use boids_core::layout::{Boid, CameraUniform, InteractionUniforms, MeshParams, SceneUniform, SimParams};
use boids_gpu::context::GpuContext;
use boids_gpu::transfer::read_buffer;

use crate::common::Check;

/// Number of scalar slots the probe writes. Must match `PROBE_SLOTS` in the probe shader.
const PROBE_SLOTS: usize = 64;

/// Bytes of ramp written into each input buffer. Comfortably larger than every struct, and a multiple
/// of 16 so the storage binding alignment rules are trivially satisfied.
const RAMP_BYTES: u64 = 512;

/// How a probe slot is typed in WGSL, which decides how the ramp value must be interpreted.
#[derive(Clone, Copy)]
enum Kind {
    /// The field is `f32`: the device reads the ramp value and reports it unchanged.
    F32,
    /// The field is `u32`: the device reads the ramp's *bit pattern* as a `u32` and converts that
    /// integer to `f32`. Six fields in the contract are `u32`, and getting this wrong makes the check
    /// report a false mismatch, which is worse than useless: it trains you to ignore the check.
    U32,
}

/// The expected byte offset and WGSL type of every scalar the probe reports, in output order.
///
/// Written out by hand rather than derived from the Rust struct, because the probe shader is the
/// independent witness: deriving both sides from the same source would make the check tautological.
/// The field order here mirrors `shaders/tests/layout_probe.wgsl` exactly.
fn expected_offsets() -> Vec<(usize, u32, Kind)> {
    macro_rules! scalar {
        ($out:expr, $slot:expr, $ty:ty, $name:ident, $kind:expr) => {
            $out.push(($slot, core::mem::offset_of!($ty, $name) as u32, $kind));
        };
    }
    macro_rules! triple {
        ($out:expr, $slot:expr, $ty:ty, $name:ident, $kind:expr) => {{
            let base = core::mem::offset_of!($ty, $name) as u32;
            $out.push(($slot, base, $kind));
            $out.push(($slot + 1, base + 4, $kind));
            $out.push(($slot + 2, base + 8, $kind));
        }};
    }

    let mut out = Vec::new();

    // Boid, slots 0..12.
    triple!(out, 0, Boid, pos, Kind::F32);
    scalar!(out, 3, Boid, species, Kind::F32);
    triple!(out, 4, Boid, vel, Kind::F32);
    scalar!(out, 7, Boid, phase, Kind::F32);
    triple!(out, 8, Boid, prev_dir, Kind::F32);
    scalar!(out, 11, Boid, color_seed, Kind::F32);

    // SimParams, slots 12..45. Slots 45..48 are the explicit padding scalars, which the probe leaves
    // at -1 on purpose: padding has no defined value and asserting on it would be a lie.
    triple!(out, 12, SimParams, grid_min, Kind::F32);
    scalar!(out, 15, SimParams, cell_size, Kind::F32);
    triple!(out, 16, SimParams, grid_dim, Kind::U32);
    scalar!(out, 19, SimParams, num_boids, Kind::U32);
    scalar!(out, 20, SimParams, w_sep, Kind::F32);
    scalar!(out, 21, SimParams, w_ali, Kind::F32);
    scalar!(out, 22, SimParams, w_coh, Kind::F32);
    scalar!(out, 23, SimParams, r_percept, Kind::F32);
    scalar!(out, 24, SimParams, r_sep, Kind::F32);
    scalar!(out, 25, SimParams, max_speed, Kind::F32);
    scalar!(out, 26, SimParams, min_speed, Kind::F32);
    scalar!(out, 27, SimParams, max_force, Kind::F32);
    scalar!(out, 28, SimParams, dt, Kind::F32);
    scalar!(out, 29, SimParams, time, Kind::F32);
    scalar!(out, 30, SimParams, sdf_strength, Kind::F32);
    scalar!(out, 31, SimParams, sdf_probe, Kind::F32);
    triple!(out, 32, SimParams, bounds_half, Kind::F32);
    scalar!(out, 35, SimParams, mode, Kind::U32);
    scalar!(out, 36, SimParams, wander, Kind::F32);
    scalar!(out, 37, SimParams, sep_boost, Kind::F32);
    scalar!(out, 38, SimParams, coh_falloff, Kind::F32);
    scalar!(out, 39, SimParams, r_safe, Kind::F32);
    scalar!(out, 40, SimParams, buoyancy, Kind::F32);
    scalar!(out, 41, SimParams, drag, Kind::F32);
    scalar!(out, 42, SimParams, env_scale, Kind::F32);
    scalar!(out, 43, SimParams, env_floor_y, Kind::F32);
    scalar!(out, 44, SimParams, env_id, Kind::U32);

    // InteractionUniforms, slots 48..60.
    triple!(out, 48, InteractionUniforms, ray_origin, Kind::F32);
    scalar!(out, 51, InteractionUniforms, mode, Kind::U32);
    triple!(out, 52, InteractionUniforms, focus_point, Kind::F32);
    scalar!(out, 55, InteractionUniforms, radius, Kind::F32);
    scalar!(out, 56, InteractionUniforms, strength, Kind::F32);
    scalar!(out, 57, InteractionUniforms, falloff, Kind::F32);
    scalar!(out, 58, InteractionUniforms, tangent, Kind::F32);
    scalar!(out, 59, InteractionUniforms, _pad, Kind::F32);

    out
}

/// Compiles and runs `shaders/tests/layout_probe.wgsl`, returning what the device read.
fn run_probe(ctx: &GpuContext) -> Vec<f32> {
    let ramp: Vec<f32> = (1..=(RAMP_BYTES / 4) as usize).map(|k| k as f32).collect();
    let ramp_bytes = bytemuck::cast_slice(&ramp).to_vec();

    let make_input = |label: &str| {
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: RAMP_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&buffer, 0, &ramp_bytes);
        buffer
    };
    let boid_in = make_input("probe boid input");
    let params_in = make_input("probe params input");
    let interaction_in = make_input("probe interaction input");
    let values_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe output"),
        size: (PROBE_SLOTS * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..4)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: binding != 3,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe layout"),
            entries: &entries,
        });
    fn bind<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
        wgpu::BindGroupEntry {
            binding,
            resource: buffer.as_entire_binding(),
        }
    }
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe bind group"),
        layout: &layout,
        entries: &[
            bind(0, &boid_in),
            bind(1, &params_in),
            bind(2, &interaction_in),
            bind(3, &values_out),
        ],
    });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
    let module = ctx.shader_module("layout probe", "tests/layout_probe.wgsl");
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("probe"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("probe encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("probe"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_pipeline(&pipeline);
        pass.dispatch_workgroups(1, 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    read_buffer(ctx, &values_out)
}

/// Checks that every struct field sits at the same byte offset on both sides.
pub fn wgsl_offsets_match_rust(ctx: &GpuContext) -> Check {
    let got = run_probe(ctx);

    let mut expected = vec![f32::NAN; PROBE_SLOTS];
    for (index, offset, kind) in expected_offsets() {
        assert!(index < PROBE_SLOTS);
        // The ramp holds the f32 value `offset / 4 + 1` at byte offset `offset`.
        if offset % 4 != 0 {
            return Err(format!(
                "host offset for probe slot {index} is {offset}, which is not 4-byte aligned; \
                 the struct has a misaligned field"
            ));
        }
        let ramp_value = (offset / 4) as f32 + 1.0;
        expected[index] = match kind {
            Kind::F32 => ramp_value,
            // `as f32`, not `f32::from_bits`: from_bits would be the identity and this branch would
            // silently become a no-op, which is exactly the bug this comment exists to prevent.
            Kind::U32 => ramp_value.to_bits() as f32,
        };
    }

    let mut mismatches = Vec::new();
    for index in 0..PROBE_SLOTS {
        if expected[index].is_nan() {
            continue;
        }
        if (got[index] - expected[index]).abs() > 0.5 {
            mismatches.push(format!(
                "slot {index}: device read {}, host wrote at byte offset {}",
                describe(got[index]),
                (expected[index] as u32).saturating_sub(1) as u32 * 4
            ));
        }
    }
    if mismatches.is_empty() {
        return Ok(());
    }
    Err(format!(
        "WGSL and Rust disagree about {} struct field(s):\n      {}\n    \
         Fix shaders/common/layout.wgsl or crates/boids-core/src/layout.rs so they match.",
        mismatches.len(),
        mismatches.join("\n      ")
    ))
}

/// Describes a probe value in terms of the ramp entry it corresponds to.
fn describe(v: f32) -> String {
    let max_ramp = RAMP_BYTES as f32 / 4.0;
    if (1.0..=max_ramp).contains(&v) && v.fract() == 0.0 {
        format!("ramp value {} (byte offset {})", v, (v - 1.0) as u32 * 4)
    } else {
        format!("{v} (not a plain ramp value; a u32 bit pattern or unread memory)")
    }
}

/// Checks the host-side sizes and alignments that the WGSL mirrors must satisfy.
///
/// These are compile-time assertions in `boids-core` as well; repeating them here means a failure is
/// reported as a named check with the actual numbers, which is far more useful than a compile error
/// pointing at a `const _` block.
pub fn struct_sizes_are_exact(_ctx: &GpuContext) -> Check {
    let table: [(&str, usize, usize, usize); 7] = [
        ("Boid", core::mem::size_of::<Boid>(), core::mem::align_of::<Boid>(), 48),
        ("SimParams", core::mem::size_of::<SimParams>(), core::mem::align_of::<SimParams>(), 144),
        (
            "InteractionUniforms",
            core::mem::size_of::<InteractionUniforms>(),
            core::mem::align_of::<InteractionUniforms>(),
            48,
        ),
        (
            "CameraUniform",
            core::mem::size_of::<CameraUniform>(),
            core::mem::align_of::<CameraUniform>(),
            192,
        ),
        ("MeshParams", core::mem::size_of::<MeshParams>(), core::mem::align_of::<MeshParams>(), 80),
        (
            "SceneUniform",
            core::mem::size_of::<SceneUniform>(),
            core::mem::align_of::<SceneUniform>(),
            304,
        ),
        (
            "KeyVal",
            core::mem::size_of::<boids_core::layout::KeyVal>(),
            core::mem::align_of::<boids_core::layout::KeyVal>(),
            8,
        ),
    ];

    let mut problems = Vec::new();
    for (name, size, align, expected_size) in table {
        if size != expected_size {
            problems.push(format!("{name} is {size} bytes, WGSL mirror expects {expected_size}"));
        }
        if align != 16 && name != "KeyVal" {
            problems.push(format!("{name} alignment is {align}, expected 16 for std430"));
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n    "))
    }
}
