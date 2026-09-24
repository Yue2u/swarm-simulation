//! The per-frame scene uniforms and the bind groups that expose simulation state to the render passes.
//!
//! # The ping-pong problem
//!
//! The simulation alternates between two agent buffers. A render pass binds whichever one the last
//! step wrote into, which changes every frame. Rebuilding the bind group every frame would work, but
//! it would allocate a driver object 60 times a second for no reason, so both bind groups are created
//! once and the renderer selects by parity. `SceneBinding` is the handle that makes that selection
//! typed rather than an index that can be passed wrong.
//!
//! # Why the uniforms share one group
//!
//! `SceneUniform`, `WaterParams`, `InteractionUniforms`, `PostParams` and `SkyParams` are five
//! buffers, uploaded once per frame and read by overlapping sets of passes. Keeping them in one bind
//! group means a pass sets one group instead of five, and means there is one place that knows which
//! passes can see what. The cost is that the bloom passes, which read none of them, still bind a group
//! that contains the agent array; they set their own group 0 instead, so nothing is wasted at draw
//! time.
//!
//! The terrain's three entries are static: the map is baked once, the scattered trees never move, and
//! `TerrainParams` describes a world rather than a frame. They are bound here anyway, so that a pass
//! that draws the ground needs exactly one bind group, the same as every other pass.

use boids_core::layout::{
    InteractionUniforms, PostParams, SceneUniform, SkyParams, TerrainParams, WaterParams,
};
use boids_gpu::context::GpuContext;
use boids_gpu::transfer::upload_uniform;

/// Which parity of the simulation's agent buffers a scene binding refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneBinding(usize);

impl SceneBinding {
    /// Binding for a given parity, 0 or 1.
    #[must_use]
    pub const fn new(parity: usize) -> Self {
        Self(parity & 1)
    }

    /// The parity as an index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// The `@group(0)` layout shared by every pass that draws scene geometry or shades the medium.
///
/// `SceneUniform` (304 bytes), the agent array, `WaterParams` (48), `InteractionUniforms` (48),
/// `PostParams` (48), `TerrainParams` (48), the baked heightfield, the tree instances, `SkyParams`
/// (48) and the landmark instances. The agents are a storage buffer rather than a vertex buffer, which
/// is what lets a single draw call read 100k agents where the simulation left them; the trees and the
/// landmarks are drawn the same way.
#[derive(Debug)]
pub struct SceneLayout {
    /// The bind group layout.
    pub layout: wgpu::BindGroupLayout,
    /// Uniform buffer holding the current `SceneUniform`.
    pub uniform: wgpu::Buffer,
    /// Uniform buffer holding the current `WaterParams`.
    pub water: wgpu::Buffer,
    /// Uniform buffer holding the current `InteractionUniforms`.
    pub interaction: wgpu::Buffer,
    /// Uniform buffer holding the current `PostParams`.
    ///
    /// Shared with the post chain, which builds its own bind groups over this same buffer: one upload
    /// per frame and one struct that can be stale, rather than two that can disagree.
    pub post: wgpu::Buffer,
    /// Uniform buffer holding the current `SkyParams`.
    pub sky: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
}

impl SceneLayout {
    /// Creates the layout, the uniform buffers and both parity bind groups.
    ///
    /// `terrain` supplies the three static entries: its parameter buffer, the baked heightfield and
    /// the scattered tree instances. They are borrowed rather than rebuilt because the map costs a
    /// compute dispatch and a readback to produce, and two copies of it would be two chances for the
    /// drawn ground to differ from the avoided ground. `landmarks` is the castle instance buffer,
    /// borrowed for the same reason: it is placed by a search and the pass that draws it must read
    /// exactly what was placed.
    #[must_use]
    pub fn new(
        ctx: &GpuContext,
        boids: [&wgpu::Buffer; 2],
        terrain: &boids_scene::TerrainGpu,
        landmarks: &wgpu::Buffer,
        sky: &SkyParams,
    ) -> Self {
        let uniform_entry =
            |binding: u32, size: u64, stages: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(size),
                },
                count: None,
            };
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene layout"),
                entries: &[
                    uniform_entry(
                        0,
                        core::mem::size_of::<SceneUniform>() as u64,
                        wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // The medium and the cursor state are fragment-only, and saying so is what lets a
                    // future backend place them in a smaller, cheaper binding space.
                    uniform_entry(
                        2,
                        core::mem::size_of::<WaterParams>() as u64,
                        wgpu::ShaderStages::FRAGMENT,
                    ),
                    uniform_entry(
                        3,
                        core::mem::size_of::<InteractionUniforms>() as u64,
                        wgpu::ShaderStages::FRAGMENT,
                    ),
                    uniform_entry(
                        4,
                        core::mem::size_of::<PostParams>() as u64,
                        wgpu::ShaderStages::FRAGMENT,
                    ),
                    uniform_entry(
                        5,
                        core::mem::size_of::<TerrainParams>() as u64,
                        wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ),
                    // Not filterable, and read with `textureLoad`: a 32-bit float texture needs
                    // `FLOAT32_FILTERABLE` to be sampled with a filtering sampler, and requiring a
                    // device feature for the ground would be a poor trade.
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Fragment only: the sky is evaluated while shading, never by a vertex shader.
                    uniform_entry(
                        8,
                        core::mem::size_of::<SkyParams>() as u64,
                        wgpu::ShaderStages::FRAGMENT,
                    ),
                    // Vertex only: the landmark instance is read once per instance to place the mesh,
                    // and the fragment shader is handed everything it needs through the varyings.
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Sizes come from the structs rather than literals, rounded up to the 16-byte multiple the
        // uniform address space requires. A hardcoded size silently goes stale the moment a field is
        // added, and the failure mode is a bind group validation error at startup.
        let make = |label: &str, size: u64| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.next_multiple_of(16),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let uniform = make("scene uniform", core::mem::size_of::<SceneUniform>() as u64);
        let water = make("water uniform", core::mem::size_of::<WaterParams>() as u64);
        let interaction = make(
            "render interaction uniform",
            core::mem::size_of::<InteractionUniforms>() as u64,
        );
        let post = make("post uniform", core::mem::size_of::<PostParams>() as u64);
        let sky_buffer = make("sky uniform", core::mem::size_of::<SkyParams>() as u64);
        upload_uniform(&ctx.queue, &sky_buffer, sky);

        fn make_bind_group(
            ctx: &GpuContext,
            layout: &wgpu::BindGroupLayout,
            buffers: [&wgpu::Buffer; 5],
            agents: &wgpu::Buffer,
            terrain: &boids_scene::TerrainGpu,
            landmarks: &wgpu::Buffer,
        ) -> wgpu::BindGroup {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: agents.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers[2].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffers[3].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: terrain.params_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(terrain.view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: terrain.trees().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: buffers[4].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: landmarks.as_entire_binding(),
                    },
                ],
            })
        }

        let scene_buffers = [&uniform, &water, &interaction, &post, &sky_buffer];
        let bind_groups = [
            make_bind_group(ctx, &layout, scene_buffers, boids[0], terrain, landmarks),
            make_bind_group(ctx, &layout, scene_buffers, boids[1], terrain, landmarks),
        ];

        Self {
            layout,
            uniform,
            water,
            interaction,
            post,
            sky: sky_buffer,
            bind_groups,
        }
    }

    /// Uploads the frame's uniforms.
    ///
    /// One call for all five: they are produced together by the app from one frame's state, and
    /// splitting it into five write sites is how a frame ends up with this frame's water and last
    /// frame's cursor.
    pub fn write(
        &self,
        queue: &wgpu::Queue,
        scene: &SceneUniform,
        water: &WaterParams,
        interaction: &InteractionUniforms,
        post: &PostParams,
        sky: &SkyParams,
    ) {
        upload_uniform(queue, &self.uniform, scene);
        upload_uniform(queue, &self.water, water);
        upload_uniform(queue, &self.interaction, interaction);
        upload_uniform(queue, &self.post, post);
        upload_uniform(queue, &self.sky, sky);
    }

    /// The bind group for a parity.
    #[must_use]
    pub fn bind_group(&self, binding: SceneBinding) -> &wgpu::BindGroup {
        &self.bind_groups[binding.index()]
    }
}
