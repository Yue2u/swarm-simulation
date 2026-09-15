//! Simulation configuration and the grid geometry derived from it.

use glam::Vec3;

use crate::layout::{EnvironmentKind, InteractionMode, InteractionUniforms, SimMode, SimParams};

/// Grid resolution and derived linearisation constants.
///
/// The domain is a fixed axis-aligned box (`grid_min`, `grid_dim * cell_size`) whose extents are
/// snapped to whole cells. Because the domain is bounded and known, cell lookup is a direct
/// `mul/add` followed by a `clamp`, with **no hash collisions** and no need for a hash table.
/// The tradeoff is preallocated `cell_start`/`cell_end` arrays of `grid_dim.x*y*z` entries; at
/// 128x64x128 cells that is 4.2 MB each, which is cheap on a desktop GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridDims {
    /// Cells per axis.
    pub dim: [u32; 3],
    /// World-space minimum corner.
    pub min: Vec3,
    /// Edge length of a cell.
    pub cell_size: f32,
}

impl GridDims {
    /// Total number of cells.
    #[must_use]
    pub const fn num_cells(&self) -> u32 {
        self.dim[0] * self.dim[1] * self.dim[2]
    }

    /// Linearises an integer cell coordinate. Caller must have clamped it into range.
    #[must_use]
    pub const fn flatten(&self, c: [u32; 3]) -> u32 {
        (c[2] * self.dim[1] + c[1]) * self.dim[0] + c[0]
    }

    /// Maps a world position to a linear cell index, clamping outside positions onto the border.
    #[must_use]
    pub fn cell_of(&self, p: Vec3) -> u32 {
        let local = (p - self.min) / self.cell_size;
        let c = [
            clamp_u32(local.x, self.dim[0]),
            clamp_u32(local.y, self.dim[1]),
            clamp_u32(local.z, self.dim[2]),
        ];
        self.flatten(c)
    }

    /// Builds grid geometry covering `domain_half` on each side, aiming for cells about
    /// `target_cell` meters across and never exceeding `max_cells_per_axis`.
    #[must_use]
    pub fn for_domain(domain_half: Vec3, target_cell: f32, max_cells_per_axis: u32) -> Self {
        let cell_size = target_cell.max(1e-3);
        let dim = [
            axis_cells(domain_half.x, cell_size, max_cells_per_axis),
            axis_cells(domain_half.y, cell_size, max_cells_per_axis),
            axis_cells(domain_half.z, cell_size, max_cells_per_axis),
        ];
        Self {
            dim,
            min: -domain_half,
            cell_size,
        }
    }
}

fn axis_cells(half_extent: f32, cell: f32, max_cells: u32) -> u32 {
    let n = (2.0 * half_extent / cell).round();
    (n.max(1.0) as u32).min(max_cells)
}

fn clamp_u32(v: f32, dim: u32) -> u32 {
    let i = v.floor();
    if i <= 0.0 {
        0
    } else {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let i = (i as u32).min(dim - 1);
        i
    }
}

/// Everything the simulation needs to run one step, in host form.
///
/// This is deliberately a plain data struct: the app toggles fields, and
/// [`SimConfig::to_params`] packs exactly what the shaders read.
#[derive(Debug, Clone, PartialEq)]
pub struct SimConfig {
    /// Number of live agents.
    pub num_boids: usize,
    /// Which world is active.
    pub mode: SimMode,
    /// Grid geometry.
    pub grid: GridDims,
    /// Soft bounds half-extent for steering.
    pub bounds_half: Vec3,

    /// Separation weight.
    pub w_sep: f32,
    /// Alignment weight.
    pub w_ali: f32,
    /// Cohesion weight.
    pub w_coh: f32,
    /// Perception radius, meters.
    pub r_percept: f32,
    /// Separation radius, meters.
    pub r_sep: f32,
    /// Density at which adaptive weights start to kick in.
    pub density_ref: f32,

    /// Speed clamp.
    pub min_speed: f32,
    /// Speed clamp.
    pub max_speed: f32,
    /// Steering force clamp.
    pub max_force: f32,
    /// Timestep, seconds.
    pub dt: f32,
    /// Wander force scale.
    pub wander: f32,

    /// SDF avoidance scale.
    pub sdf_strength: f32,
    /// Forward probe distance, meters.
    pub sdf_probe: f32,
    /// Margin kept from surfaces, meters.
    pub r_safe: f32,

    /// Constant vertical push.
    pub buoyancy: f32,
    /// Medium drag, 1/s.
    pub drag: f32,

    /// Which avoidance field the environment exposes.
    pub env: EnvironmentKind,
    /// Field scale: reef column repetition period, or terrain amplitude in metres.
    pub env_scale: f32,
    /// Seafloor height for the reef field. Ignored by the terrain field.
    pub env_floor_y: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self::for_mode(SimMode::Fish, 100_000)
    }
}

impl SimConfig {
    /// Preset for a given world and agent count.
    #[must_use]
    pub fn for_mode(mode: SimMode, num_boids: usize) -> Self {
        let (bounds_half, r_percept, max_speed, buoyancy, drag) = match mode {
            SimMode::Fish => (Vec3::new(160.0, 70.0, 160.0), 3.5, 14.0, 0.15, 0.35),
            SimMode::Birds => (Vec3::new(600.0, 220.0, 600.0), 14.0, 32.0, -0.05, 0.12),
        };
        // Cell size tracks the perception radius: near-neighbour search then only needs the 27
        // cells around a boid, which is what makes the grid worth its cost.
        let grid = GridDims::for_domain(bounds_half, 2.0 * r_percept / 3.0, 256);
        // The safety margin must be at least the distance needed to stop from cruise speed at the
        // avoidance force limit, otherwise agents tunnel through thin terrain: they enter the
        // smoothstep band already moving faster than it can arrest. This is the single most
        // important tuning relationship in the whole config.
        let sdf_strength = max_speed * 3.0;
        let stopping_distance = max_speed * max_speed / (2.0 * sdf_strength);
        let r_safe = (r_percept * 0.4).max(stopping_distance * 1.25);
        // The reef's repetition period is chosen so a column spans several perception radii: agents
        // then have to genuinely steer around an obstacle rather than drift past a bump. Terrain uses
        // the amplitude of the height field instead, and ignores the seafloor height.
        let (env, env_scale, env_floor_y) = match mode {
            SimMode::Fish => (EnvironmentKind::Reef, 42.0, -bounds_half.y),
            SimMode::Birds => (EnvironmentKind::Terrain, 120.0, 0.0),
        };
        Self {
            num_boids,
            mode,
            grid,
            bounds_half,
            w_sep: 1.6,
            w_ali: 1.0,
            w_coh: 0.9,
            r_percept,
            r_sep: r_percept * 0.35,
            density_ref: 12.0,
            min_speed: max_speed * 0.35,
            max_speed,
            max_force: max_speed * 3.0,
            dt: 1.0 / 60.0,
            wander: 0.35,
            sdf_strength,
            sdf_probe: (r_percept * 0.6).max(stopping_distance),
            r_safe,
            buoyancy,
            drag,
            env,
            env_scale,
            env_floor_y,
        }
    }

    /// Perception radius used by the shaders.
    #[must_use]
    pub fn radius(&self) -> f32 {
        self.r_percept
    }

    /// Packs the config into the GPU uniform. `density_ref` is converted into the shader-side
    /// reciprocal form the adaptive weights need.
    #[must_use]
    pub fn to_params(&self, time: f32, dt: f32) -> SimParams {
        SimParams {
            grid_min: self.grid.min.to_array(),
            cell_size: self.grid.cell_size,
            grid_dim: self.grid.dim,
            num_boids: u32::try_from(self.num_boids).unwrap_or(u32::MAX),
            w_sep: self.w_sep,
            w_ali: self.w_ali,
            w_coh: self.w_coh,
            r_percept: self.r_percept,
            r_sep: self.r_sep,
            max_speed: self.max_speed,
            min_speed: self.min_speed,
            max_force: self.max_force,
            dt,
            time,
            sdf_strength: self.sdf_strength,
            sdf_probe: self.sdf_probe,
            bounds_half: self.bounds_half.to_array(),
            mode: self.mode.as_u32(),
            wander: self.wander,
            sep_boost: 1.0 / self.density_ref.max(1e-3),
            coh_falloff: 1.0 / self.density_ref.max(1e-3),
            r_safe: self.r_safe,
            buoyancy: self.buoyancy,
            drag: self.drag,
            env_scale: self.env_scale,
            env_floor_y: self.env_floor_y,
            env_id: self.env.as_u32(),
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }

    /// Disables cursor influence. The app overwrites this every frame when the pointer moves.
    #[must_use]
    pub fn idle_interaction() -> InteractionUniforms {
        InteractionUniforms {
            ray_origin: [0.0; 3],
            mode: InteractionMode::Off.as_u32(),
            focus_point: [0.0; 3],
            radius: 0.0,
            strength: 0.0,
            falloff: 2.0,
            tangent: 0.0,
            _pad: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_lookup_is_clamped_and_injective_in_range() {
        let grid = GridDims::for_domain(Vec3::splat(100.0), 2.0, 256);
        let n = grid.num_cells();
        for p in [
            Vec3::splat(-1000.0),
            Vec3::splat(1000.0),
            Vec3::ZERO,
            Vec3::new(99.0, -99.0, 0.0),
        ] {
            assert!(grid.cell_of(p) < n, "cell index out of range for {p:?}");
        }
        // Distinct interior positions must not collide into the same cell index.
        let a = grid.cell_of(Vec3::new(0.0, 0.0, 0.0));
        let b = grid.cell_of(Vec3::new(40.0, 0.0, 0.0));
        assert_ne!(a, b);
    }

    #[test]
    fn flatten_matches_manual_indexing() {
        let grid = GridDims {
            dim: [4, 3, 2],
            min: Vec3::ZERO,
            cell_size: 1.0,
        };
        assert_eq!(grid.num_cells(), 24);
        assert_eq!(grid.flatten([1, 0, 0]), 1);
        assert_eq!(grid.flatten([0, 1, 0]), 4);
        assert_eq!(grid.flatten([0, 0, 1]), 12);
        assert_eq!(grid.flatten([3, 2, 1]), 23);
    }

    #[test]
    fn cell_size_follows_perception_radius() {
        let fish = SimConfig::for_mode(SimMode::Fish, 1000);
        assert!((fish.grid.cell_size - 2.0 * fish.r_percept / 3.0).abs() < 1e-6);
        let p = fish.to_params(0.0, fish.dt);
        assert_eq!(p.num_boids, 1000);
        assert_eq!(p.mode, SimMode::Fish.as_u32());
    }
}
