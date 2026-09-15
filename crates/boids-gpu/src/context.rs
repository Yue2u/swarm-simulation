//! Device, adapter and surface acquisition.
//!
//! Kept deliberately small and explicit: one place decides which adapter to use, one place requests
//! limits and features, and one place configures a swapchain. Everything else receives a
//! `&GpuContext` and never touches the instance again.
//!
//! `boids-gpu` does not depend on `winit`. Surface creation takes any
//! `Into<wgpu::SurfaceTarget<'static>>`, which `Arc<winit::window::Window>` satisfies, so the window
//! type stays entirely inside `boids-app`.

use boids_core::wgsl::ShaderLoader;

/// Everything needed to create a [`GpuContext`].
#[derive(Debug, Clone)]
pub struct GpuContextDescriptor {
    /// Ask for the discrete adapter rather than the integrated one.
    pub high_performance: bool,
    /// Ask for `TIMESTAMP_QUERY` if the adapter has it. Enables the per-pass GPU profiler.
    pub want_timestamps: bool,
    /// Force a specific backend set, e.g. `Some(wgpu::Backends::GL)` for debugging.
    pub force_backends: Option<wgpu::Backends>,
    /// Use a software adapter. Only useful to make a test run on a machine with no usable GPU.
    pub force_fallback: bool,
}

impl Default for GpuContextDescriptor {
    fn default() -> Self {
        Self {
            high_performance: true,
            want_timestamps: true,
            force_backends: None,
            force_fallback: false,
        }
    }
}

/// Instance, adapter, device, queue and the shared shader loader.
pub struct GpuContext {
    instance: wgpu::Instance,
    /// The chosen adapter.
    pub adapter: wgpu::Adapter,
    /// The logical device.
    pub device: wgpu::Device,
    /// The submission queue.
    pub queue: wgpu::Queue,
    /// Adapter description, logged at startup and shown in the title bar.
    pub info: wgpu::AdapterInfo,
    /// Resolves `//#include` in WGSL sources.
    pub shaders: ShaderLoader,
    /// Whether `TIMESTAMP_QUERY` was actually granted.
    pub timestamps_enabled: bool,
}

impl core::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GpuContext")
            .field("info", &self.info)
            .field("timestamps_enabled", &self.timestamps_enabled)
            .finish_non_exhaustive()
    }
}

impl GpuContext {
    /// Requests an adapter and device without a surface.
    ///
    /// Used by the headless tests and by the app before the window exists.
    ///
    /// # Errors
    /// Returns a description when no adapter matches or the device request fails.
    pub fn new(desc: &GpuContextDescriptor) -> Result<Self, String> {
        // `new_without_display_handle` rather than a struct literal: `InstanceDescriptor` is
        // non-exhaustive and constructing it by hand would break on every wgpu minor bump. A
        // display handle is only needed to present through GLES on Wayland, which is not a target
        // here; the Vulkan path ignores it.
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = desc.force_backends.unwrap_or_else(wgpu::Backends::all);
        let instance = wgpu::Instance::new(instance_desc);

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: if desc.high_performance {
                wgpu::PowerPreference::HighPerformance
            } else {
                wgpu::PowerPreference::LowPower
            },
            force_fallback_adapter: desc.force_fallback,
            compatible_surface: None,
            // Disable limit buckets: the bucket rounding can silently give a device fewer resources
            // than requested, which would only show up as a failure at high agent counts.
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("no suitable adapter: {e}"))?;

        let info = adapter.get_info();
        log::info!(
            "adapter: {} type={:?} backend={:?} driver={} {}",
            info.name,
            info.device_type,
            info.backend,
            info.driver,
            info.driver_info
        );

        // Start from `downlevel_defaults` so the app also runs on a WebGL-class backend, then raise
        // the storage limits to whatever the adapter actually offers. The spatial grid needs a large
        // storage binding and a 100k-agent buffer is 4.8 MB, so keeping the downlevel defaults for
        // those two limits would fail only at high agent counts, which is the worst time to find out.
        let adapter_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffers_per_shader_stage: adapter_limits
                .max_storage_buffers_per_shader_stage
                .max(8),
            ..wgpu::Limits::downlevel_defaults()
        };

        let mut features = wgpu::Features::empty();
        let mut timestamps_enabled = false;
        if desc.want_timestamps && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
            timestamps_enabled = true;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("boids device"),
            required_features: features,
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("device request failed: {e}"))?;

        device.on_uncaptured_error(std::sync::Arc::new(|error: wgpu::Error| {
            // Panicking is intentional. An uncaptured validation error means the frame is already
            // wrong; continuing would produce a silently corrupt simulation, which is far more
            // expensive to debug than a crash with a message.
            log::error!("uncaptured wgpu error: {error}");
            panic!("uncaptured wgpu error: {error}");
        }));

        log::info!(
            "limits: max_storage_binding={} MiB max_buffer={} MiB max_workgroups={} timestamps={}",
            limits.max_storage_buffer_binding_size / (1 << 20),
            limits.max_buffer_size / (1 << 20),
            limits.max_compute_workgroups_per_dimension,
            timestamps_enabled
        );

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            info,
            shaders: ShaderLoader::workspace_default(),
            timestamps_enabled,
        })
    }

    /// Compiles a shader with `//#include` resolution and creates a module.
    ///
    /// # Panics
    /// Panics carrying the include-graph error, or the WGSL source plus the compiler error message,
    /// if the shader is broken. A broken shader is a programming error rather than a runtime
    /// condition, and failing loudly at startup with the actual error text is strictly better than
    /// rendering a black screen and guessing.
    #[must_use]
    pub fn shader_module(&self, label: &str, relative_path: &str) -> wgpu::ShaderModule {
        let compiled = self
            .shaders
            .compile(relative_path)
            .unwrap_or_else(|e| panic!("compiling {relative_path}: {e}"));
        if log::log_enabled!(log::Level::Debug) {
            log::debug!(
                "shader {label}: {} lines from {:?}",
                compiled.source.lines().count(),
                compiled
                    .sources
                    .iter()
                    .map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string())
                    .collect::<Vec<_>>()
            );
        }
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(compiled.source.into()),
            })
    }

    /// Creates and configures a surface for a window.
    ///
    /// `target` is normally an `Arc<winit::window::Window>`; taking it generically keeps `winit` out
    /// of this crate.
    ///
    /// # Errors
    /// Returns a description when the adapter cannot present to this surface, or when configuring
    /// the chosen format fails.
    pub fn create_surface(
        &self,
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<SurfaceState, String> {
        let surface = self
            .instance
            .create_surface(target)
            .map_err(|e| format!("surface creation failed: {e}"))?;
        SurfaceState::configure(self, surface, width, height)
    }

    /// Description of the adapter, for the window title or a HUD line.
    #[must_use]
    pub fn adapter_summary(&self) -> String {
        format!(
            "{} / {:?} / {:?}",
            self.info.name, self.info.backend, self.info.device_type
        )
    }

    /// Blocks until all submitted work has completed.
    ///
    /// Only used by tests and by the shutdown path. The frame loop never waits on the device.
    pub fn wait_idle(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }
}

/// A configured presentation surface.
#[derive(Debug)]
pub struct SurfaceState {
    /// The surface itself. Owns the window it was created from, hence `'static`.
    pub surface: wgpu::Surface<'static>,
    /// Current configuration.
    pub config: wgpu::SurfaceConfiguration,
    /// Whether the chosen format is sRGB.
    ///
    /// When it is, the shader's linear output is encoded by the hardware on write and the tonemapper
    /// must *not* apply its own sRGB transfer function. Getting this backwards is the classic
    /// "everything is washed out" bug.
    pub is_srgb: bool,
}

impl SurfaceState {
    fn configure(
        ctx: &GpuContext,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> Result<Self, String> {
        let caps = surface.get_capabilities(&ctx.adapter);
        if caps.formats.is_empty() {
            return Err("adapter offers no surface formats".to_string());
        }
        let format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);

        // Mailbox (uncapped, no tearing) where available, otherwise Fifo. On a 1440p display with a
        // 16 ms frame budget the difference is felt as input latency on the cursor interaction.
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            // Two frames in flight: enough to keep the GPU busy without adding a third frame of
            // input latency, which the cursor interaction would make very noticeable.
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: Vec::new(),
        };

        // `configure` returns unit and reports failure through the device error scope, which the
        // uncaptured-error handler turns into a panic with the driver's message.
        surface.configure(&ctx.device, &config);
        log::info!(
            "surface {}x{} format={:?} present={:?}",
            config.width,
            config.height,
            config.format,
            config.present_mode
        );

        Ok(Self {
            surface,
            is_srgb: config.format.is_srgb(),
            config,
        })
    }

    /// Reconfigures after a window resize. Zero-sized requests are ignored, since a surface cannot
    /// be configured with a zero extent and the window will send a real size shortly after.
    pub fn resize(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if self.config.width == width && self.config.height == height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&ctx.device, &self.config);
    }

    /// Viewport size in physical pixels.
    #[must_use]
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Acquires the next frame's texture.
    ///
    /// Returns `None` for every condition that means "skip this frame and try again": a timeout, an
    /// occluded window, an outdated or lost surface. Only the resize case is reported separately,
    /// because the caller can fix it immediately by reconfiguring.
    pub fn acquire(
        &mut self,
        ctx: &GpuContext,
        width: u32,
        height: u32,
    ) -> Option<wgpu::SurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                Some(t)
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.resize(ctx, width, height);
                None
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => None,
        }
    }
}
