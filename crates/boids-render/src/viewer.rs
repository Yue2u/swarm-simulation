//! The model viewer: one mesh at a time, in a studio, to look at what the project actually draws.
//!
//! # Why it reuses the scene's pipelines
//!
//! The obvious implementation is a viewer shader with its own copy of every mesh. That copy is the
//! problem: the fish and the bird do not exist as data anywhere on the host - `render/boid.wgsl`
//! generates them from `vertex_index` and `MeshParams`, and the castle and the tree are host meshes
//! that the scene's shaders shade with the scene's lights. A viewer that re-drew any of them would be
//! a second source of truth for what a model looks like, and the failure mode is the quiet one: the
//! viewer keeps showing last month's fish.
//!
//! So the viewer draws through `BoidPass`, `TreePass` and `LandmarkPass` themselves, over a group 0
//! that is the scene's layout with three buffers swapped for the viewer's own:
//!
//! | binding | scene | viewer |
//! |---|---|---|
//! | 0 | `SceneUniform` | the studio's (unit scale, studio light) |
//! | 1 | the swarm | one synthetic agent |
//! | 7 | the scattered forest | one tree at the origin |
//! | 9 | the placed castle | one castle at the origin |
//!
//! Everything else - the water, the cursor, the terrain, the sky - is the scene's own buffer, because
//! the shaders read it for their lighting and the studio wants the real thing. The post chain is the
//! scene's too, so a model is shown through the same tone curve it is shown through in the world.
//!
//! # Why the meshes are drawn at unit scale
//!
//! In the scene every mesh is scaled by a framing decision: an agent's scale is a fraction of the
//! perception radius so a swarm reads at any world size, and a tree's is the scatter's. The *model*
//! is the unit mesh, so that is what the viewer draws, and each model brings its own camera framing.

use boids_core::layout::{Boid, SimMode, StaticInstance};
use boids_gpu::context::GpuContext;
use glam::Vec3;

use crate::scene::SceneLayout;

/// Background of the studio, linear HDR. Dark and slightly cool, so a pale castle and a glowing fish
/// both read against it.
pub(crate) const STUDIO_BACKGROUND: [f32; 3] = [0.020, 0.023, 0.028];

/// Speed the synthetic agent is given, in metres per second.
///
/// Small on purpose: `render/boid.wgsl` stretches the body by up to 18% with speed, which is right in
/// the scene and wrong in a viewer whose whole point is the shape. Small rather than zero because the
/// vertex shader builds its orientation basis from the velocity, and a zero vector has no direction.
const CREATURE_SPEED: f32 = 0.01;

/// Which mesh the viewer shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Model {
    /// The fish the underwater world draws.
    #[default]
    Fish,
    /// The bird the sky world draws.
    Bird,
    /// The tree the scatter pass plants.
    Tree,
    /// The castle the landmark pass draws.
    Castle,
}

impl Model {
    /// Every model, in the order the viewer cycles them.
    pub const ALL: [Model; 4] = [Model::Fish, Model::Bird, Model::Tree, Model::Castle];

    /// Name, for the window title, the log line and `--model`.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Model::Fish => "fish",
            Model::Bird => "bird",
            Model::Tree => "tree",
            Model::Castle => "castle",
        }
    }

    /// Looks a model up by name, for the command line.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.label() == name)
    }

    /// The next model, wrapping.
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Model::Fish => Model::Bird,
            Model::Bird => Model::Tree,
            Model::Tree => Model::Castle,
            Model::Castle => Model::Fish,
        }
    }

    /// The previous model, wrapping.
    #[must_use]
    pub const fn previous(self) -> Self {
        match self {
            Model::Fish => Model::Castle,
            Model::Bird => Model::Fish,
            Model::Tree => Model::Bird,
            Model::Castle => Model::Tree,
        }
    }

    /// The world whose creature this is, or whose grade and medium show a static mesh best.
    ///
    /// The fish is shown in water and the other three in air, because that is where each of them is
    /// drawn: the medium is part of what a fish looks like, and a castle shown through a metre of
    /// water would be a different picture from the one the scene produces.
    #[must_use]
    pub const fn world(self) -> SimMode {
        match self {
            Model::Fish => SimMode::Fish,
            Model::Bird | Model::Tree | Model::Castle => SimMode::Birds,
        }
    }

    /// Whether the mesh is the vertex-pulled creature rather than a host-built static mesh.
    #[must_use]
    pub const fn is_creature(self) -> bool {
        matches!(self, Model::Fish | Model::Bird)
    }

    /// Point the viewer's camera orbits, in model space. The static meshes stand on the origin and
    /// are framed about their middle; the creatures are centred on their bodies.
    #[must_use]
    pub fn target(self) -> Vec3 {
        match self {
            Model::Fish | Model::Bird => Vec3::new(0.0, 0.0, -0.1),
            Model::Tree | Model::Castle => Vec3::new(0.0, 0.5, 0.0),
        }
    }

    /// How far the camera sits from the target, in model units.
    ///
    /// Sized so each mesh fills about two thirds of the frame at the default field of view, which is
    /// what makes the four of them comparable at a glance: a bird is three units across the wings and
    /// a castle is one unit tall, and a single distance would show one as a speck and the other as a
    /// wall.
    #[must_use]
    pub const fn distance(self) -> f32 {
        match self {
            Model::Fish => 2.6,
            Model::Bird => 3.6,
            Model::Tree => 1.7,
            Model::Castle => 1.6,
        }
    }

    /// Elevation the camera starts at, radians.
    #[must_use]
    pub const fn pitch(self) -> f32 {
        match self {
            Model::Fish | Model::Bird => 0.30,
            Model::Tree | Model::Castle => 0.24,
        }
    }

    /// The synthetic agent a creature model is instanced from.
    ///
    /// One agent, facing +Z, with a mid-palette species and no colour jitter, so the viewer shows a
    /// representative individual rather than the swarm's spread. `phase` is zero: the animation runs
    /// off `scene.camera.time`, so the tail and the wings move while the viewer is open.
    #[must_use]
    fn agent(self) -> Boid {
        Boid {
            pos: [0.0; 3],
            species: 0.4,
            vel: [0.0, 0.0, CREATURE_SPEED],
            phase: 0.0,
            prev_dir: [0.0, 0.0, 1.0],
            color_seed: 0.5,
        }
    }

    /// The single instance a static model is drawn from.
    ///
    /// A fixed yaw so the first frame shows the model at an angle rather than as a flat elevation:
    /// a castle seen from straight on is a rectangle, and a tree is a line.
    #[must_use]
    fn instance(self) -> StaticInstance {
        StaticInstance {
            pos: [0.0; 3],
            scale: 1.0,
            yaw: match self {
                // The gate is on +Z, so a small yaw turns it towards the default orbit.
                Model::Castle => 0.35,
                // Trees are grown with a golden-angle spiral, so any yaw shows a different silhouette.
                Model::Tree => 0.7,
                Model::Fish | Model::Bird => 0.0,
            },
            // Mid-range species and the forest end of the biome mask, so a tree is the green conifer
            // the scatter plants most of rather than one of the dry stragglers.
            kind: 0.3,
            mask: 0.0,
            pad1: 0.0,
        }
    }
}

/// The viewer's own group 0: the scene's layout with three buffers swapped.
#[derive(Debug)]
pub struct ModelViewer {
    /// One synthetic agent, so the creature pipeline has something to instance.
    agents: wgpu::Buffer,
    /// One tree at the origin, for the tree pass.
    trees: wgpu::Buffer,
    /// One castle at the origin, for the landmark pass.
    landmarks: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl ModelViewer {
    /// Allocates the single-instance buffers and builds the bind group.
    ///
    /// `terrain` supplies bindings 5 and 6, which no shader the viewer draws reads: the tree, the
    /// landmark and the agent shaders all take the ground for granted and none of them samples it.
    /// They are bound because the group has to be complete, and borrowing the real ones is cheaper
    /// than three dummy buffers that exist only to satisfy a layout.
    #[must_use]
    pub fn new(ctx: &GpuContext, scene: &SceneLayout, terrain: &boids_scene::TerrainGpu) -> Self {
        let make = |label: &str, size: u64| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let agents = make("viewer agent", core::mem::size_of::<Boid>() as u64);
        let trees = make("viewer tree", core::mem::size_of::<StaticInstance>() as u64);
        let landmarks = make(
            "viewer landmark",
            core::mem::size_of::<StaticInstance>() as u64,
        );

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewer scene bind group"),
            layout: &scene.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agents.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene.water.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scene.interaction.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scene.post.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: terrain.params_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(terrain.view()),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: trees.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: scene.sky.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: landmarks.as_entire_binding(),
                },
            ],
        });

        Self {
            agents,
            trees,
            landmarks,
            bind_group,
        }
    }

    /// Uploads the instance data for a model.
    ///
    /// All three buffers are written every time the model changes rather than only the one being
    /// drawn: it is 80 bytes in total, and a stale instance in one of them is a model that shows the
    /// wrong thing the next time it is selected.
    pub fn set_model(&self, queue: &wgpu::Queue, model: Model) {
        boids_gpu::transfer::upload_uniform(queue, &self.agents, &model.agent());
        let instance = model.instance();
        boids_gpu::transfer::upload_uniform(queue, &self.trees, &instance);
        boids_gpu::transfer::upload_uniform(queue, &self.landmarks, &instance);
    }

    /// The group every viewer draw binds.
    #[must_use]
    pub const fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_model_is_named_cycled_and_looked_up() {
        // A model that is missing from `ALL` cannot be selected from the command line and is never
        // drawn; one that is missing from the cycle is unreachable from the keyboard.
        for (i, model) in Model::ALL.iter().enumerate() {
            assert_eq!(Model::from_name(model.label()), Some(*model));
            assert_eq!(model.next(), Model::ALL[(i + 1) % Model::ALL.len()]);
            assert_eq!(
                model.previous(),
                Model::ALL[(i + Model::ALL.len() - 1) % Model::ALL.len()]
            );
        }
        assert_eq!(Model::from_name("dragon"), None);
    }

    /// The camera has to be outside the model and its near plane inside it, or the viewer shows the
    /// inside of a wall.
    #[test]
    fn every_model_is_framed_from_outside_itself() {
        for model in Model::ALL {
            assert!(
                model.distance() > 0.5,
                "{:?} is framed from {} units away",
                model,
                model.distance()
            );
            // The static meshes stand on y = 0 and are one unit tall; the creatures are centred on
            // the origin and are about two units long.
            assert!((0.0..=1.0).contains(&model.target().y));
        }
    }

    /// The studio shows the fish in water and everything else in air.
    #[test]
    fn the_medium_follows_the_model() {
        assert_eq!(Model::Fish.world(), SimMode::Fish);
        assert!(Model::Fish.is_creature());
        for model in [Model::Bird, Model::Tree, Model::Castle] {
            assert_eq!(model.world(), SimMode::Birds);
        }
        assert!(Model::Bird.is_creature());
        assert!(!Model::Tree.is_creature() && !Model::Castle.is_creature());
    }
}
