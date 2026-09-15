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
//! `SceneUniform`, `WaterParams`, `InteractionUniforms` and `PostParams` are four buffers, uploaded
//! once per frame and read by overlapping sets of passes. Keeping them in one bind group means a pass
//! sets one group instead of four, and means there is one place that knows which passes can see what.
//! The cost is that the bloom passes, which read none of them, still bind a group that contains the
//! agent array; they set their own group 0 instead, so nothing is wasted at draw time.

use boids_core::layout::{InteractionUniforms, PostParams, SceneUniform, WaterParams};
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
/// `SceneUniform` (304 bytes), the agent array, `WaterParams` (48), `InteractionUniforms` (48) and
/// `PostParams` (48). The agents are a storage buffer rather than a vertex buffer, which is what lets
/// a single draw call read 100k agents where the simulation left them.
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
    bind_groups: [wgpu::BindGroup; 2],
}

impl SceneLayout {
    /// Creates the layout, the uniform buffers and both parity bind groups.
    #[must_use]
    pub fn new(ctx: &GpuContext, boids: [&wgpu::Buffer; 2]) -> Self {
        let uniform_entry = |binding: u32, size: u64, stages: wgpu::ShaderStages| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(size),
                },
                count: None,
            }
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

        fn make_bind_group(
            ctx: &GpuContext,
            layout: &wgpu::BindGroupLayout,
            buffers: [&wgpu::Buffer; 4],
            agents: &wgpu::Buffer,
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
                ],
            })
        }

        let bind_groups = [
            make_bind_group(ctx, &layout, [&uniform, &water, &interaction, &post], boids[0]),
            make_bind_group(ctx, &layout, [&uniform, &water, &interaction, &post], boids[1]),
        ];

        Self {
            layout,
            uniform,
            water,
            interaction,
            post,
            bind_groups,
        }
    }

    /// Uploads the frame's uniforms.
    ///
    /// One call for all four: they are produced together by the app from one frame's state, and
    /// splitting it into four write sites is how a frame ends up with this frame's water and last
    /// frame's cursor.
    pub fn write(
        &self,
        queue: &wgpu::Queue,
        scene: &SceneUniform,
        water: &WaterParams,
        interaction: &InteractionUniforms,
        post: &PostParams,
    ) {
        upload_uniform(queue, &self.uniform, scene);
        upload_uniform(queue, &self.water, water);
        upload_uniform(queue, &self.interaction, interaction);
        upload_uniform(queue, &self.post, post);
    }

    /// The bind group for a parity.
    #[must_use]
    pub fn bind_group(&self, binding: SceneBinding) -> &wgpu::BindGroup {
        &self.bind_groups[binding.index()]
    }
}
