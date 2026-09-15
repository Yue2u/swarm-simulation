//! Device-side check that the SDF and terrain fields agree with their Rust twins.
//!
//! The fields are the contract between three places that cannot see each other: the compute shader
//! that makes agents avoid a surface, the ocean raymarch that draws the same surface, and the CPU
//! reference that the physics is validated against. If any two disagree, nothing crashes and nothing
//! looks obviously wrong: the fish simply steer around a rock that the camera does not draw, or the
//! camera draws a rock the agents pass through. A sign flip is even worse, because both signs "look
//! like avoidance" until an agent is inside the solid.
//!
//! The mechanism is a grid of sample points evaluated on both sides. The device runs
//! `shaders/tests/sdf_probe.wgsl`, which calls every shared field; the host recomputes the same fields
//! from `boids_core::sdf` and `boids_core::terrain` and compares element by element. The tolerance is
//! `1e-4` metres, which is loose enough for the `sin` in the radius profile to differ by a last-bit
//! rounding round trip and tight enough that a wrong hash, radius, blend radius or sign is meters off.

use boids_core::sdf;
use boids_core::terrain;
use boids_gpu::context::GpuContext;
use boids_gpu::transfer::read_buffer;
use glam::{Vec2, Vec3, Vec3Swizzles};

use crate::common::Check;

/// Number of fields evaluated per sample. Must match `FIELDS` in `tests/sdf_probe.wgsl`.
const FIELDS: usize = 11;

/// Samples per axis. The plan asks for a 32^3 grid.
const GRID: usize = 32;

/// Reef repetition period handed to the device, metres. The same value is used for the terrain
/// amplitude, which is what `eval_field` does for the sky world.
const PERIOD: f32 = 48.0;

/// Seafloor height handed to the device, metres.
const FLOOR_Y: f32 = -24.0;

/// Terrain frequency used by `eval_field` for the sky world.
const TERRAIN_FREQUENCY: f32 = 0.0025;

/// Biome mask frequency.
const BIOME_FREQUENCY: f32 = 0.004;

/// Names of the fields, in the shader's output order, for failure reporting.
const FIELD_NAMES: [&str; FIELDS] = [
    "sphere",
    "box",
    "plane_y",
    "smin",
    "column",
    "radius_profile",
    "reef_field",
    "terrain_height",
    "biome.x",
    "biome.y",
    "biome.z",
];

/// The sample grid: a box spanning several reef periods, above and below the seafloor.
///
/// The `xz` range is wider than a period so more than one hash cell is exercised, and the `y` range
/// crosses both the floor and the tops of the columns so the smooth unions are sampled on both sides
/// of their blend radius.
fn sample_points() -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(GRID * GRID * GRID);
    #[allow(clippy::cast_precision_loss)]
    let denom = (GRID - 1) as f32;
    for iy in 0..GRID {
        for iz in 0..GRID {
            for ix in 0..GRID {
                #[allow(clippy::cast_precision_loss)]
                let (fx, fy, fz) = (ix as f32 / denom, iy as f32 / denom, iz as f32 / denom);
                out.push([
                    -96.0 + 192.0 * fx,
                    -30.0 + 90.0 * fy,
                    -96.0 + 192.0 * fz,
                    0.0,
                ]);
            }
        }
    }
    out
}

/// The Rust value of `field` at `p`, in the same order the shader evaluates it.
fn expected(field: usize, p: Vec3) -> f32 {
    let sphere = sdf::sphere(p, Vec3::ZERO, 5.0);
    let boxed = sdf::box_sdf(p, Vec3::new(2.0, 1.0, -3.0), Vec3::new(4.0, 2.0, 1.0));
    let t = ((p.y + 5.0) / 10.0).clamp(0.0, 1.0);
    match field {
        0 => sphere,
        1 => boxed,
        2 => sdf::plane_y(p, 1.5),
        3 => sdf::smin(sphere, boxed, 0.75),
        4 => sdf::column(p, Vec2::new(3.0, -2.0), 2.5, -4.0, 6.0),
        5 => sdf::column_radius_profile(t, 3.0, 0.45),
        6 => sdf::reef_field(p, PERIOD, FLOOR_Y),
        7 => terrain::height_at(p.xz(), PERIOD, TERRAIN_FREQUENCY),
        8..=10 => terrain::biome_weights(p.xz(), BIOME_FREQUENCY)[field - 8],
        _ => unreachable!("field index {field} is out of range"),
    }
}

/// Runs `shaders/tests/sdf_probe.wgsl` over the sample grid and returns its flattened output.
fn run_probe(ctx: &GpuContext) -> Vec<f32> {
    let points = sample_points();
    #[allow(clippy::cast_possible_truncation)]
    let n = points.len() as u32;
    let points_bytes = bytemuck::cast_slice(&points);

    let points_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sdf probe points"),
        size: points_bytes.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue.write_buffer(&points_buf, 0, points_bytes);

    let params: [f32; 4] = [PERIOD, FLOOR_Y, 0.0, 0.0];
    let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sdf probe params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue
        .write_buffer(&params_buf, 0, bytemuck::cast_slice(&params));

    #[allow(clippy::cast_possible_truncation)]
    let values_bytes = n as u64 * FIELDS as u64 * 4;
    let values_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sdf probe values"),
        size: values_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sdf probe layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sdf probe bind group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: values_buf.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sdf probe pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
    let module = ctx.shader_module("sdf probe", "tests/sdf_probe.wgsl");
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sdf probe pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("probe_sdf"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sdf probe encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sdf probe"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_pipeline(&pipeline);
        pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
    }
    ctx.queue.submit(Some(encoder.finish()));
    read_buffer(ctx, &values_buf)
}

/// Compares every shared field on the host and the device over a 32^3 grid.
pub fn wgsl_matches_rust(ctx: &GpuContext) -> Check {
    let points = sample_points();
    let got = run_probe(ctx);
    let expected_len = points.len() * FIELDS;
    if got.len() != expected_len {
        return Err(format!(
            "the probe returned {} values, expected {expected_len} ({} samples x {FIELDS} fields)",
            got.len(),
            points.len()
        ));
    }

    const TOLERANCE: f32 = 1e-4;
    let mut first: Option<String> = None;
    let mut worst = 0.0f32;
    let mut worst_at = (0usize, 0usize);
    let mut mismatches = 0usize;

    for (i, sample) in points.iter().enumerate() {
        let p = Vec3::new(sample[0], sample[1], sample[2]);
        for field in 0..FIELDS {
            let want = expected(field, p);
            let have = got[i * FIELDS + field];
            let d = (have - want).abs();
            if d > worst {
                worst = d;
                worst_at = (i, field);
            }
            if d > TOLERANCE {
                mismatches += 1;
                if first.is_none() {
                    first = Some(format!(
                        "field '{}' at sample {i} {p:?}: device {have}, rust {want} (|d| {d:.3e})",
                        FIELD_NAMES[field]
                    ));
                }
            }
        }
    }

    if let Some(message) = first {
        return Err(format!(
            "{mismatches} of {expected_len} values disagree by more than {TOLERANCE:.0e}:\n      \
             {message}\n      \
             worst {worst:.3e} at sample {} field '{}'\n      \
             Fix shaders/common/sdf.wgsl or crates/boids-core/src/{{sdf,terrain}}.rs so the two \
             sides evaluate the same field.",
            worst_at.0, FIELD_NAMES[worst_at.1]
        ));
    }

    println!(
        "\n    {} samples x {FIELDS} fields agree within {TOLERANCE:.0e} (worst {worst:.2e})",
        points.len()
    );
    Ok(())
}
