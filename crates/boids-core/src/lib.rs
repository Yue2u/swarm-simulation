//! Data contract and CPU-side reference math for the GPU boids simulation.
//!
//! This crate is the single source of truth for:
//!
//! * every `#[repr(C)]` struct that is also declared in WGSL (`layout`),
//! * the simulation parameters and grid geometry (`config`),
//! * the vector math shared by the CPU reference and the shaders (`math`, `sdf`),
//! * camera / ray math for cursor interaction (`camera`),
//! * the naive O(N^2) reference implementation used to validate the GPU version (`reference`),
//! * the WGSL source preprocessor (`wgsl`).
//!
//! It intentionally has **no dependency on `wgpu`** so that it can be unit tested without a GPU.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_debug_implementations)]

pub mod camera;
pub mod config;
pub mod layout;
pub mod math;
pub mod reference;
pub mod rng;
pub mod sdf;
pub mod spawn;
pub mod terrain;
pub mod wgsl;

pub use camera::{ray_from_ndc, ray_plane_intersect, OrbitCamera, Ray};
pub use config::{GridDims, SimConfig};
pub use layout::{
    Boid, CameraUniform, EnvironmentKind, InteractionMode, InteractionUniforms, KeyVal, MeshParams,
    SceneUniform, SimMode, SimParams, SortParams, EMPTY_CELL,
};
pub use math::{clamp_len, order_parameter, smoothstep};
pub use rng::Pcg32;
pub use spawn::{spawn_shell, spawn_swarm, validation_issues};
