//! The underwater environment pass: a half-resolution raymarch of the reef distance field, then a
//! full-resolution resolve that reconstructs colour and depth.
//!
//! # Why a pass of its own
//!
//! The sky world's backdrop can be a full-screen gradient (`background.rs`), because nothing in it
//! needs to interact with the agents' depth. The reef cannot: the fish have to swim *behind* the
//! columns, which means the environment has to write real depth before the agents are drawn. So the
//! resolve writes `frag_depth` from the raymarched hit distance and runs first in the scene pass, and
//! the agent pipeline's `LessEqual` test then does the rest.
//!
//! # Why two passes
//!
//! Per pixel the raymarch costs up to 80 field evaluations, six more for the shading gradient, and
//! twelve shaft samples with four shadow taps each; it was the most expensive pass in the frame by a
//! wide margin. It is also a smooth integral: shading it at half resolution and letting a bilinear tap
//! put it back is not visible, and it is a quarter of the pixels. So the raymarch draws into a
//! half-size colour target and a separate, cheap full-size pass upsamples it into the scene, writing
//! the depth the agents test against.
//!
//! # Why the distance rides in the alpha channel
//!
//! The raymarch's real distance is what the agents need, but the half-size pass has no depth
//! attachment for the resolve to read: sampling a `texture_depth_2d` is not portable to the GL
//! backend this is developed against, where depth bindings lower to shadow samplers that `texelFetch`
//! cannot touch. The colour target is `Rgba16Float`, its alpha channel is otherwise unused, and the
//! resolve writes `frag_depth` from it. A linear distance in metres survives a half-float channel with
//! about 0.1 m of resolution at 200 m, which is all an occlusion test needs; the NDC depth of the
//! same distance would not, because it compresses the whole world into the top few thousandths of the
//! range.

use crate::scene::{SceneBinding, SceneLayout};
use crate::targets::{FrameTargets, DEPTH_FORMAT, HDR_FORMAT};

/// The underwater environment pipeline set.
#[derive(Debug)]
pub struct OceanPass {
    /// Half-resolution sphere trace into the half-size colour target.
    raymarch: wgpu::RenderPipeline,
    /// Full-resolution resolve: bilinear colour, nearest distance, hand-written depth.
    resolve: wgpu::RenderPipeline,
    /// Layout of the resolve's texture group (group 1).
    resolve_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// The resolve's textures. Rebuilt on resize, because the target it reads is.
    resolve_bind_group: wgpu::BindGroup,
}

impl OceanPass {
    /// Builds both pipelines and the resolve's bind group for a target set.
    ///
    /// # Panics
    /// Panics if a shader fails to compile. That is a programming error and belongs at startup, with
    /// the WGSL error text, rather than a black frame.
    #[must_use]
    pub fn new(
        ctx: &boids_gpu::context::GpuContext,
        scene: &SceneLayout,
        targets: &FrameTargets,
    ) -> Self {
        let raymarch_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ocean pipeline layout"),
                bind_group_layouts: &[Some(&scene.layout)],
                immediate_size: 0,
            });
        let module = ctx.shader_module("ocean", "render/ocean.wgsl");
        let raymarch = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ocean"),
                layout: Some(&raymarch_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    // No vertex buffer at all: the triangle is generated from `vertex_index`.
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        // HDR: the surface and the caustics are far above 1.0 and the bloom pass
                        // needs that range to have anything to spread.
                        format: HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                // No depth attachment: the raymarch is one triangle and a colour write, and its
                // distance reaches the agents through the alpha channel and the resolve instead.
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let resolve_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ocean resolve layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        // The resolve needs `scene` for the camera's near and far planes, so group 0 is the same
        // scene layout every other pass binds; group 1 is its own texture pair.
        let resolve_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("ocean resolve pipeline layout"),
                    bind_group_layouts: &[Some(&scene.layout), Some(&resolve_layout)],
                    immediate_size: 0,
                });
        let resolve_module = ctx.shader_module("ocean_resolve", "render/ocean_resolve.wgsl");
        let resolve = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ocean resolve"),
                layout: Some(&resolve_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &resolve_module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &resolve_module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    // The resolve is where the raymarch's distance enters the shared depth buffer, so
                    // the agents can be occluded by rock. It always writes and always passes, because
                    // its value comes from `@builtin(frag_depth)` and there is no earlier geometry.
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // Linear because the source is half-size: a nearest tap would show the raymarch's pixels as
        // blocks. Clamp so a tap at the frame edge cannot wrap the far side of the ocean in.
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ocean resolve sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let resolve_bind_group = make_resolve_bind_group(ctx, &resolve_layout, &sampler, targets);
        Self {
            raymarch,
            resolve,
            resolve_layout,
            sampler,
            resolve_bind_group,
        }
    }

    /// Rebuilds the resolve's texture bind group after the targets were reallocated.
    pub fn resize(
        &mut self,
        ctx: &boids_gpu::context::GpuContext,
        targets: &FrameTargets,
    ) {
        self.resolve_bind_group =
            make_resolve_bind_group(ctx, &self.resolve_layout, &self.sampler, targets);
    }

    /// Records the half-resolution raymarch into the ocean target.
    pub fn draw_raymarch<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a SceneLayout,
        binding: SceneBinding,
    ) {
        pass.set_pipeline(&self.raymarch);
        pass.set_bind_group(0, scene.bind_group(binding), &[]);
        // One triangle covering the whole target: three vertices, no index buffer.
        pass.draw(0..3, 0..1);
    }

    /// Records the full-resolution resolve into the scene pass.
    pub fn draw_resolve<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a SceneLayout,
        binding: SceneBinding,
    ) {
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(0, scene.bind_group(binding), &[]);
        pass.set_bind_group(1, &self.resolve_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn make_resolve_bind_group(
    ctx: &boids_gpu::context::GpuContext,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    targets: &FrameTargets,
) -> wgpu::BindGroup {
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ocean resolve bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(targets.ocean_view()),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}
