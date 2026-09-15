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
//! 1. background, with depth writes off: fills every pixel so that distant geometry never has to be
//!    cleared twice,
//! 2. agents, depth tested and depth writing, one instanced draw for the whole swarm,
//! 3. post-processing, once it exists.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod background;
pub mod boid_pass;
pub mod png;
pub mod readback;
pub mod renderer;
pub mod scene;
pub mod targets;

pub use renderer::{FrameStats, Renderer};
pub use scene::SceneBinding;
pub use targets::FrameTargets;
