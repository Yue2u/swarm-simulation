//! The sky world's ground: the heightfield map, the tree scatter and the numbers that describe both.
//!
//! # What lives here and why
//!
//! Three things, in dependency order:
//!
//! * [`terrain_params`] turns a [`SimConfig`] into the [`TerrainParams`] uniform. The height
//!   amplitude and the noise frequency in it are *the same values* the simulation's collision field
//!   uses, taken from the same config, because a mesh built from one field and a collision field
//!   built from another is the most expensive divergence this project can ship: birds would fly
//!   through the drawn ground and bounce off invisible ground somewhere else.
//! * [`TerrainGpu`] bakes the map with a compute pass and scatters the trees with another. Both run
//!   once, at startup; neither is re-run per frame, because neither input changes.
//! * The resolution and spacing constants below, which are the only sizing decisions in the module.
//!
//! # Why the map has a fixed texel size rather than a fixed resolution
//!
//! The app's world is 1.5 km across and the render suite's test worlds are 250 m; a fixed 1024^2 map
//! would spend 95% of its texels on the small world's out-of-bounds region and be visibly chunky in
//! the large one. Sizing from a target texel size instead makes the map *mean* the same thing in both
//! worlds, and it scales the startup cost with the world rather than with the constant.

use boids_core::config::SimConfig;
use boids_core::layout::{TerrainParams, TreeInstance};

/// How far the map extends beyond the simulation's steering box, per axis.
///
/// The birds are steered inside `±bounds_half`, so the ground has to reach at least that far. The
/// margin is what pushes the map's edge out of the shot: at 1.25 the boundary is 750 m from the
/// centre of the app's world, which the aerial perspective has turned into sky by the time the
/// camera at 0.8 world diagonals can see it.
const MAP_MARGIN: f32 = 1.25;

/// Target world size of one map texel, metres.
///
/// The terrain's finest octave has a wavelength of about 47 m at the app's world size, so 1.5 m
/// texels resolve it eleven times over. Smaller texels buy nothing visible; larger ones start to
/// round off the ridges.
const TEXEL_SIZE: f32 = 1.5;

/// Resolution bounds. Powers of two, so the mesh's grid lines can land exactly on texel centres.
const MIN_RESOLUTION: u32 = 64;
/// Upper bound on the map's resolution, the plan's 1024^2.
const MAX_RESOLUTION: u32 = 1024;

/// Texels per mesh cell.
///
/// The mesh samples every texel it draws, so this is also the ratio that decides how much of the
/// map's detail reaches the screen: at 1 the mesh is as dense as the map and costs 4x the vertices
/// for sub-pixel detail, at 8 the surface visibly loses the noise's smaller features.
const MESH_TEXELS_PER_CELL: u32 = 4;

/// Target spacing between scatter candidates, metres.
///
/// One candidate per cell of this size, jittered inside it (see `terrain/scatter.wgsl`), so this is
/// the minimum distance between two trees.
const TREE_SPACING: f32 = 12.0;

/// Tree height as a fraction of the terrain's amplitude.
///
/// Trees that are a tenth of the ridge height read as forest at this scale; at a quarter they become
/// the terrain's silhouette rather than its texture.
const TREE_HEIGHT_FRACTION: f32 = 0.13;

/// Instance slots per scatter candidate. Generous on purpose: the scatter pass clamps to the
/// capacity rather than growing, so too small a buffer silently thins the forest.
const SLOTS_PER_CANDIDATE: u32 = 4;

/// The terrain uniform for a world.
///
/// The two field-defining numbers (`amplitude`, `frequency`) are copied from the config rather than
/// recomputed, so there is exactly one place that decides what the terrain is: `SimConfig::for_mode`.
#[must_use]
pub fn terrain_params(config: &SimConfig) -> TerrainParams {
    let half = config.bounds_half;
    let size_xz = [2.0 * half.x * MAP_MARGIN, 2.0 * half.z * MAP_MARGIN];
    let resolution = map_resolution(size_xz[0]);
    let candidates = scatter_candidates(size_xz[0]);
    let trees = config.mode == boids_core::layout::SimMode::Birds;

    TerrainParams {
        min_xz: [-size_xz[0] * 0.5, -size_xz[1] * 0.5],
        size_xz,
        amplitude: config.env_scale,
        frequency: config.env_freq,
        // Biome cells are half the size of the terrain's own features: big enough to read as regions
        // rather than as patches, and correlated with the ridges without being the same field.
        biome_frequency: 2.0 * config.env_freq,
        segments: mesh_segments(resolution),
        tree_height: if trees {
            config.env_scale * TREE_HEIGHT_FRACTION
        } else {
            0.0
        },
        tree_capacity: if trees {
            (candidates * candidates / SLOTS_PER_CANDIDATE).max(256)
        } else {
            0
        },
        tree_candidates: if trees { candidates } else { 0 },
        resolution,
    }
}

/// Map resolution for a world span, in texels. Always a power of two.
#[must_use]
pub fn map_resolution(span: f32) -> u32 {
    let wanted = (span / TEXEL_SIZE).round().max(1.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let wanted = wanted as u32;
    wanted
        .next_power_of_two()
        .clamp(MIN_RESOLUTION, MAX_RESOLUTION)
}

/// Mesh segments per axis for a map resolution.
///
/// `resolution / MESH_TEXELS_PER_CELL`, floored to an integer so that a grid line lands exactly on a
/// texel centre: `heightfield.wgsl`'s vertex path relies on `resolution % segments == 0`, and an
/// off-by-one there would make every vertex interpolate between texels instead of reading one.
#[must_use]
pub fn mesh_segments(resolution: u32) -> u32 {
    (resolution / MESH_TEXELS_PER_CELL).max(8)
}

/// Scatter candidates per axis for a world span.
fn scatter_candidates(span: f32) -> u32 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n = (span / TREE_SPACING).round().max(8.0) as u32;
    n.max(8)
}

/// The baked heightfield and the scattered trees, ready to bind.
///
/// Created once per world, at startup: the compute passes it owns are recorded and submitted in
/// [`TerrainGpu::new`], including the one readback in the whole render path (the number of trees the
/// scatter accepted, which sizes the draw call).
#[derive(Debug)]
pub struct TerrainGpu {
    params: TerrainParams,
    params_buffer: wgpu::Buffer,
    heightfield: wgpu::Texture,
    heightfield_view: wgpu::TextureView,
    trees: wgpu::Buffer,
    counters: wgpu::Buffer,
    tree_count: u32,
}

impl TerrainGpu {
    /// Bakes the map and scatters the trees.
    ///
    /// # Panics
    /// Panics if a shader fails to compile or a pipeline fails validation. Both are programming
    /// errors, and failing at startup with the WGSL error text beats rendering a world with no
    /// ground.
    #[must_use]
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, config: &SimConfig) -> Self {
        let params = terrain_params(config);
        let resolution = map_resolution(params.size_xz[0]);
        let capacity = params.tree_capacity.max(1);

        let heightfield = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain heightfield"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // r is the height in metres, g the biome mask, b and a unused. `rgba32float` rather than
            // the two-channel `rg32float` because the latter has no storage support on the GL
            // adapter this is developed against; the two wasted channels cost 8 MiB at 1024^2 and
            // buy exact f32 heights. Not filterable, so every consumer uses `textureLoad` (see
            // `common/heightfield.wgsl`).
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                // For the test that compares the baked map against the analytic field.
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let heightfield_view = heightfield.create_view(&wgpu::TextureViewDescriptor::default());

        let make_buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let params_buffer = make_buffer(
            "terrain params",
            (core::mem::size_of::<TerrainParams>() as u64).next_multiple_of(16),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
        let trees = make_buffer(
            "tree instances",
            u64::from(capacity) * core::mem::size_of::<TreeInstance>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let counters = make_buffer(
            "tree counter",
            4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );

        let mut terrain = Self {
            params,
            params_buffer,
            heightfield,
            heightfield_view,
            trees,
            counters,
            tree_count: 0,
        };
        terrain.bake(device, queue, resolution, capacity);
        terrain
    }

    /// Records the two compute passes and reads the tree count back.
    fn bake(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, resolution: u32, capacity: u32) {
        let loader = boids_core::wgsl::ShaderLoader::workspace_default();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain bake"),
        });

        // The map first: the scatter reads it.
        let height_module = compile(device, &loader, "terrain/heightfield", "terrain/heightfield.wgsl");
        let height_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain heightfield layout"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let height_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain heightfield"),
            layout: &height_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.heightfield_view),
                },
            ],
        });
        let height_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain heightfield pipeline layout"),
            bind_group_layouts: &[Some(&height_layout)],
            immediate_size: 0,
        });
        let height_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("terrain_heightfield"),
                layout: Some(&height_pipeline_layout),
                module: &height_module,
                entry_point: Some("generate"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain heightfield"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&height_pipeline);
            pass.set_bind_group(0, &height_bind_group, &[]);
            let groups = resolution.div_ceil(16);
            pass.dispatch_workgroups(groups, groups, 1);
        }

        // Then the trees. The counter is zeroed on the queue before the dispatch; the pass clamps
        // against the array length, so a short buffer costs trees rather than validation errors.
        if self.params.tree_capacity > 0 {
            queue.write_buffer(&self.counters, 0, bytemuck::bytes_of(&0u32));
            let scatter_module = compile(device, &loader, "terrain/scatter", "terrain/scatter.wgsl");
            let scatter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("terrain scatter layout"),
                entries: &[
                    uniform_entry(0),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    storage_entry(2, false),
                    storage_entry(3, false),
                ],
            });
            let scatter_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("terrain scatter"),
                layout: &scatter_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.heightfield_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.trees.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.counters.as_entire_binding(),
                    },
                ],
            });
            let scatter_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("terrain scatter pipeline layout"),
                    bind_group_layouts: &[Some(&scatter_layout)],
                    immediate_size: 0,
                });
            let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("terrain_scatter"),
                layout: Some(&scatter_pipeline_layout),
                module: &scatter_module,
                entry_point: Some("scatter"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&scatter_pipeline);
            pass.set_bind_group(0, &scatter_bind_group, &[]);
            let candidates = self.params.tree_candidates.max(1);
            pass.dispatch_workgroups(candidates * candidates, 1, 1);
        }

        queue.submit(Some(encoder.finish()));

        // The one synchronisation in the render path, and it happens once, before the first frame.
        // The draw call needs the number of trees, and the number exists only on the device.
        self.tree_count = if self.params.tree_capacity > 0 {
            read_u32(device, queue, &self.counters).min(capacity)
        } else {
            0
        };
        log::info!(
            "terrain: {resolution}x{resolution} map over {:.0}x{:.0} m ({} segments), \
             amplitude {:.0} m, {} trees of {} slots",
            self.params.size_xz[0],
            self.params.size_xz[1],
            self.params.segments,
            self.params.amplitude,
            self.tree_count,
            self.params.tree_capacity,
        );
    }

    /// The uniform for this world's terrain.
    #[must_use]
    pub const fn params(&self) -> &TerrainParams {
        &self.params
    }

    /// The uniform buffer the render pass reads.
    #[must_use]
    pub const fn params_buffer(&self) -> &wgpu::Buffer {
        &self.params_buffer
    }

    /// The baked map, as a view the scene bind group can bind.
    #[must_use]
    pub const fn view(&self) -> &wgpu::TextureView {
        &self.heightfield_view
    }

    /// The map texture itself, for readback tests.
    #[must_use]
    pub const fn texture(&self) -> &wgpu::Texture {
        &self.heightfield
    }

    /// The scattered tree instances.
    #[must_use]
    pub const fn trees(&self) -> &wgpu::Buffer {
        &self.trees
    }

    /// Number of trees the scatter pass accepted.
    #[must_use]
    pub const fn tree_count(&self) -> u32 {
        self.tree_count
    }

    /// Map resolution in texels.
    #[must_use]
    pub fn resolution(&self) -> u32 {
        self.heightfield.width()
    }
}

fn compile(
    device: &wgpu::Device,
    loader: &boids_core::wgsl::ShaderLoader,
    label: &str,
    path: &str,
) -> wgpu::ShaderModule {
    let module = loader
        .compile(path)
        .unwrap_or_else(|e| panic!("compiling {path}: {e}"));
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(module.source.into()),
    })
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(core::mem::size_of::<TerrainParams>() as u64),
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Reads a single `u32` back, blocking until the GPU has finished.
///
/// Used exactly once, to size the tree draw call. Every other readback in the project is in a test
/// or in the deterministic screenshot mode.
fn read_u32(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> u32 {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tree counter readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tree counter copy"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, 4);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => panic!("mapping the tree counter failed: {e}"),
        Err(e) => panic!("tree counter readback channel closed: {e}"),
    }
    let value = {
        let view = slice.get_mapped_range().expect("mapped counter");
        u32::from_le_bytes([view[0], view[1], view[2], view[3]])
    };
    staging.unmap();
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use boids_core::layout::SimMode;

    #[test]
    fn the_mesh_grid_stays_aligned_to_the_map() {
        // The vertex path in `render/terrain.wgsl` divides the resolution by the segment count and
        // relies on the remainder being zero; anything else makes every vertex a blend of two texels
        // and softens the whole surface.
        for half in [40.0f32, 126.0, 300.0, 600.0] {
            let resolution = map_resolution(2.0 * half * MAP_MARGIN);
            let segments = mesh_segments(resolution);
            assert_eq!(
                resolution % segments,
                0,
                "{resolution} texels do not divide into {segments} segments"
            );
            assert!(resolution.is_power_of_two());
        }
    }

    #[test]
    fn the_map_covers_the_steering_box_with_margin() {
        for mode in [SimMode::Fish, SimMode::Birds] {
            let config = SimConfig::for_mode(mode, 1000);
            let p = terrain_params(&config);
            assert!(
                p.size_xz[0] >= 2.0 * config.bounds_half.x,
                "{mode:?}: the map is narrower than the world it must cover"
            );
            assert!(p.size_xz[1] >= 2.0 * config.bounds_half.z);
            assert!((p.min_xz[0] + 0.5 * p.size_xz[0]).abs() < 1e-3, "the map is not centred");
        }
    }

    /// The field the mesh is baked from must be the field the simulation avoids.
    #[test]
    fn terrain_params_carry_the_simulation_s_field() {
        for mode in [SimMode::Fish, SimMode::Birds] {
            let config = SimConfig::for_mode(mode, 1000);
            let p = terrain_params(&config);
            assert_eq!(p.amplitude, config.env_scale);
            assert_eq!(p.frequency, config.env_freq);
            // And the collision field's own height at the world's centre must be reachable by the
            // map: a bird that spawns above the terrain can sample that height.
            let height = boids_core::terrain::height_at(glam::Vec2::ZERO, p.amplitude, p.frequency);
            assert!(
                (0.0..=p.amplitude).contains(&height),
                "terrain height {height} escaped [0, {amplitude}]",
                amplitude = p.amplitude
            );
        }
    }

    #[test]
    fn trees_are_only_scattered_in_the_sky_world() {
        let birds = terrain_params(&SimConfig::for_mode(SimMode::Birds, 1000));
        let fish = terrain_params(&SimConfig::for_mode(SimMode::Fish, 1000));
        assert!(birds.tree_capacity > 0 && birds.tree_height > 0.0);
        assert_eq!(fish.tree_capacity, 0);
        assert_eq!(fish.tree_candidates, 0);
    }

    /// The capacity has to be comfortably above what the scatter can accept.
    ///
    /// If it is not, the forest is silently thinned: the pass clamps instead of growing, and the only
    /// symptom is a sparser wood.
    #[test]
    fn the_tree_capacity_covers_a_full_forest() {
        let config = SimConfig::for_mode(SimMode::Birds, 100_000);
        let p = terrain_params(&config);
        let candidates = u64::from(p.tree_candidates);
        let slots = u64::from(SLOTS_PER_CANDIDATE);
        // The scatter accepts at most every candidate.
        assert!(
            u64::from(p.tree_capacity) >= candidates * candidates / slots,
            "capacity {} cannot hold the candidates-squared worst case",
            p.tree_capacity
        );
        // And the spacing must be smaller than the perception radius is large, or a forest is a
        // handful of lonesome trees.
        assert!(TREE_SPACING < config.r_percept * 2.0);
    }
}
