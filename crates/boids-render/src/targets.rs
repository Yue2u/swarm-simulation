//! Frame target management: the HDR colour intermediate and the depth attachment.
//!
//! Both are functions of the window size and both must be recreated together on resize, exactly once,
//! rather than by each pass that happens to notice the size changed. That is the whole reason they are
//! isolated here.
//!
//! # Why the scene is rendered into `Rgba16Float`
//!
//! The frame contains values far outside `[0, 1]`: the water surface seen from below, the sun disc
//! through Snell's window, and bioluminescent fish are all additive lights on top of a dim medium. An
//! 8-bit target clamps them at white, and the bloom pass then has nothing to spread: the halo would
//! come from the *clipped* values, which is the difference between "bright" and "glowing". Half-float
//! gives the range (and the exponent) to keep the ratio between a caustic and the water around it,
//! which the tone curve compresses at the end of the frame instead of the rasteriser truncating it at
//! the start.

/// Depth format. `Depth32Float` rather than `Depth24PlusStencil8`: nothing here uses stencil, and the
/// extra precision matters because the underwater raymarch writes real distances into this buffer and
/// the agents are depth-tested against terrain at a range of hundreds of metres.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Format of the scene intermediate every geometry pass draws into, before tonemapping.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The frame's colour and depth attachments.
#[derive(Debug)]
pub struct FrameTargets {
    /// The scene, in linear HDR.
    hdr: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// The underwater raymarch's own, half-resolution colour target. The raymarch is by far the most
    /// expensive pass in the frame and it is a full-screen integral; rendering it at half resolution
    /// and upsampling is a factor of four on its pixel cost for a soft image that a bilinear tap
    /// hides. The alpha channel carries the ray's hit distance in metres, because a depth texture
    /// cannot be sampled portably in the resolve shader on every backend.
    ocean: wgpu::Texture,
    ocean_view: wgpu::TextureView,
    ocean_size: (u32, u32),
    /// Current size in physical pixels.
    pub size: (u32, u32),
}

impl FrameTargets {
    /// Allocates both attachments at a given size.
    #[must_use]
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let hdr = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr scene"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            // `TEXTURE_BINDING` because the bloom bright pass and the composite sample it.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let hdr_view = hdr.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

        let ocean_size = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
        let ocean = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ocean half-res"),
            size: wgpu::Extent3d {
                width: ocean_size.0,
                height: ocean_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let ocean_view = ocean.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            hdr,
            hdr_view,
            depth,
            depth_view,
            ocean,
            ocean_view,
            ocean_size,
            size: (width, height),
        }
    }

    /// Recreates the targets if the size changed. Returns whether anything was reallocated, so the
    /// caller can log resize churn instead of guessing whether a resize happened.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        let (width, height) = (width.max(1), height.max(1));
        if self.size == (width, height) {
            return false;
        }
        *self = Self::new(device, width, height);
        true
    }

    /// View of the HDR intermediate, for the geometry passes to draw into and for the post chain to
    /// sample.
    #[must_use]
    pub fn hdr_view(&self) -> &wgpu::TextureView {
        &self.hdr_view
    }

    /// The HDR texture, exposed for the passes that need its size.
    #[must_use]
    pub fn hdr_texture(&self) -> &wgpu::Texture {
        &self.hdr
    }

    /// View to bind as the depth attachment.
    #[must_use]
    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// View of the half-resolution ocean target: the raymarch writes it, the resolve samples it.
    #[must_use]
    pub fn ocean_view(&self) -> &wgpu::TextureView {
        &self.ocean_view
    }

    /// Size of the ocean target, in texels.
    #[must_use]
    pub const fn ocean_size(&self) -> (u32, u32) {
        self.ocean_size
    }

    /// The ocean texture, exposed for tests that need its format or size.
    #[must_use]
    pub fn ocean_texture(&self) -> &wgpu::Texture {
        &self.ocean
    }

    /// The depth texture, exposed for passes that need its size.
    #[must_use]
    pub fn depth_texture(&self) -> &wgpu::Texture {
        &self.depth
    }
}
