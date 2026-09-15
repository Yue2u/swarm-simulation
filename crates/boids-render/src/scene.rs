//! The per-frame scene uniform and the bind groups that expose simulation state to the render passes.
//!
//! # The ping-pong problem
//!
//! The simulation alternates between two agent buffers. A render pass binds whichever one the last
//! step wrote into, which changes every frame. Rebuilding the bind group every frame would work, but
//! it would allocate a driver object 60 times a second for no reason, so both bind groups are created
//! once and the renderer selects by parity. `SceneBinding` is the handle that makes that selection
//! typed rather than an index that can be passed wrong.

use boids_core::layout::SceneUniform;
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

/// The `@group(0)` layout shared by every pass that draws agents or the background.
///
/// `SceneUniform` (144 bytes) plus the agent array. The agents are a storage buffer rather than a
/// vertex buffer, which is what lets a single draw call read 100k agents where the simulation left
/// them.
#[derive(Debug)]
pub struct SceneLayout {
    /// The bind group layout.
    pub layout: wgpu::BindGroupLayout,
    /// Uniform buffer holding the current `SceneUniform`.
    pub uniform: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
}

impl SceneLayout {
    /// Creates the layout and both parity bind groups.
    #[must_use]
    pub fn new(ctx: &GpuContext, boids: [&wgpu::Buffer; 2]) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                core::mem::size_of::<SceneUniform>() as u64,
                            ),
                        },
                        count: None,
                    },
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
                ],
            });

        // Sized from the struct rather than a literal, rounded up to the 16-byte multiple the uniform
        // address space requires. A hardcoded size here silently goes stale the moment a field is
        // added to `SceneUniform`, and the failure mode is a bind group validation error at startup.
        let uniform_size = (core::mem::size_of::<SceneUniform>() as u64).next_multiple_of(16);
        let uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scene uniform"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        fn make(
            ctx: &GpuContext,
            layout: &wgpu::BindGroupLayout,
            uniform: &wgpu::Buffer,
            agents: &wgpu::Buffer,
        ) -> wgpu::BindGroup {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: agents.as_entire_binding(),
                    },
                ],
            })
        }

        let bind_groups = [
            make(ctx, &layout, &uniform, boids[0]),
            make(ctx, &layout, &uniform, boids[1]),
        ];

        Self {
            layout,
            uniform,
            bind_groups,
        }
    }

    /// Uploads the frame's scene uniform.
    pub fn write(&self, queue: &wgpu::Queue, scene: &SceneUniform) {
        upload_uniform(queue, &self.uniform, scene);
    }

    /// The bind group for a parity.
    #[must_use]
    pub fn bind_group(&self, binding: SceneBinding) -> &wgpu::BindGroup {
        &self.bind_groups[binding.index()]
    }
}
