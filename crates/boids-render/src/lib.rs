//! Frame graph and render passes for both worlds.
//!
//! `boids-render` owns the *presentation* half of the frame: it consumes the agent buffer the
//! simulation just produced and turns it into pixels. It never writes simulation state and it never
//! reads simulation state on the CPU.
//!
//! # Resolution strategy
//!
//! The target is whatever the window reports, from 1920x1080 to 3840x2160, and nothing here assumes a
//! size: the depth target is recreated on resize and every pass derives its work from the viewport. A
//! 1440p window is the reference case the constants are tuned for. Passes that are expensive per pixel
//! and forgiving of resolution (bloom, the underwater raymarch) will render at half resolution and
//! upscale, which is what keeps 4K within the frame budget; that lands with those passes.
//!
//! # Draw ordering
//!
//! 1. environment: the sky backdrop (a gradient, depth off) or the underwater raymarch (which writes
//!    the depth the agents are then tested against),
//! 2. agents, depth tested and depth writing, one instanced draw for the whole swarm,
//! 3. bloom and the tone mapping composite, into the swapchain.
//!
//! # HDR
//!
//! Everything before the composite renders into an `Rgba16Float` intermediate. The frame contains
//! lights far above 1.0 - the water surface, the sun through it, bioluminescent agents - and a bloom
//! pass fed by a clipped 8-bit target produces a halo with no relationship to the scene's brightness.
//! The tone curve belongs in exactly one place, the composite shader, and the transfer function is
//! chosen there from the target's own format (`is_srgb`).

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod background;
pub mod boid_pass;
pub mod ocean;
pub mod png;
pub mod post;
pub mod readback;
pub mod renderer;
pub mod scene;
pub mod targets;
pub mod terrain;
pub mod tree;

pub use renderer::{FrameInput, FrameStats, Renderer};
pub use scene::{SceneBinding, SceneLayout};
pub use terrain::TerrainPass;
pub use tree::TreePass;
pub use targets::{FrameTargets, DEPTH_FORMAT, HDR_FORMAT};
