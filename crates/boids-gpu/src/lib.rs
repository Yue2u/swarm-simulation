//! GPU device setup, buffer management and compute pipelines for the simulation.
//!
//! This crate owns everything that talks to `wgpu` about *simulation state*: the ping-pong agent
//! buffers, the spatial grid, the sort, and the compute pipelines. It knows nothing about biomes,
//! rendering or input; those live in `boids-scene`, `boids-render` and `boids-app`.
//!
//! # The one rule
//!
//! Nothing here may touch the CPU per agent. Every buffer is filled once at startup and then only
//! ever written by compute shaders. The only per-frame CPU work is `write_buffer` on a handful of
//! uniform structs, which is why the simulation cost is independent of the agent count in the app's
//! frame budget.
//!
//! # Frame structure
//!
//! ```text
//! P0 clear_cells    cell_start[c] = EMPTY_CELL
//! P1 hash           keys[dst][i] = {cell_index(pos_i), i}, padding -> PAD_KEY
//! P2 sort           bitonic sort of keys by cell index, one pass per stage, parameters in immediates
//! P3 build_ranges   cell_start/cell_end from the sorted keys
//! P4 integrate      neighbour search over 27 cells -> forces -> integration
//! ```
//!
//! `SimPipelines::record_step` records P0..P4 for one step; `SimResources::swap` then flips the
//! ping-pong parity, so the renderer draws from the buffer the step just wrote while the next step
//! writes into the other one. A frame in flight never reads a buffer that is being written.
//!
//! The all-pairs strategy skips P0..P3: it needs no grid, and below a few thousand agents it is
//! faster than paying for the sort that feeds one.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod context;
pub mod mesh_profile;
pub mod profile;
pub mod sim;
pub mod transfer;

pub use context::{GpuContext, GpuContextDescriptor, SurfaceState};
pub use mesh_profile::{mesh_profile, scene_uniform};
pub use profile::{FrameTimings, GpuProfiler, PassTotal, MAX_TIMED_PASSES};
pub use sim::{
    dispatch_size, key_plan, sort_stages, KeyPlan, SimPipelines, SimResources, Strategy,
};
pub use transfer::{read_buffer, read_raw, upload_boids, upload_uniform};
