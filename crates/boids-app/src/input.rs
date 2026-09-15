//! Input state accumulated from window events.
//!
//! # Two layers, on purpose
//!
//! `handle` translates a `winit` event into one of four primitive state transitions
//! ([`InputState::on_cursor_move`], [`InputState::on_button`], [`InputState::on_scroll`],
//! [`InputState::on_key`]). Everything that can be wrong lives in the primitives, and the primitives
//! are directly unit testable without constructing a `WindowEvent`, which in `winit` 0.30 requires a
//! device id that cannot be made in a test without `unsafe`.
//!
//! Events accumulate and are consumed once per frame. That matters for two reasons: `winit` can
//! deliver many motion events between frames and handling each one would do redundant work, and a
//! frame that ran while events were still arriving would see a half-updated camera.

use boids_core::camera::OrbitCamera;
use boids_core::layout::InteractionMode;
use glam::Vec2;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{Key, NamedKey};

/// Which drag gesture is in progress. Exactly one at a time: a mouse has one cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Drag {
    /// No button held.
    #[default]
    None,
    /// Left button: orbit.
    Orbit,
    /// Right button: pan.
    Pan,
    /// Middle button: momentary cursor attractor.
    Attract,
}

/// Accumulated input for the current frame.
#[derive(Debug)]
pub struct InputState {
    /// Cursor position in physical pixels, top-left origin. `None` until the pointer has moved once.
    pub cursor: Option<Vec2>,
    /// Pointer motion since the previous frame, in pixels.
    pub motion: Vec2,
    /// Scroll accumulated since the previous frame, in lines.
    pub scroll: f32,
    /// Which drag gesture is active.
    pub drag: Drag,
    /// Cursor interaction mode, changed by the number keys.
    pub mode: InteractionMode,
    /// Whether the cursor is influencing the swarm right now. Kept separate from `mode` so that
    /// releasing the middle button turns the influence off without forgetting the selected mode.
    pub active: bool,
    /// Whether the simulation is paused.
    pub paused: bool,
    /// Set when the user asked to quit.
    pub quit: bool,
    /// Set when the user asked to switch worlds.
    pub toggle_world: bool,
    /// Set when the user asked to reset the swarm.
    pub reset: bool,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            cursor: None,
            motion: Vec2::ZERO,
            scroll: 0.0,
            drag: Drag::None,
            mode: InteractionMode::Attract,
            active: true,
            paused: false,
            quit: false,
            toggle_world: false,
            reset: false,
        }
    }
}

impl InputState {
    /// Records a new pointer position and accumulates the delta since the previous one.
    ///
    /// The first position only establishes a baseline: `winit` reports an absolute position, so
    /// treating the first sample as motion would rotate the camera by the distance from the origin.
    pub fn on_cursor_move(&mut self, position: Vec2) {
        if let Some(previous) = self.cursor {
            self.motion += position - previous;
        }
        self.cursor = Some(position);
    }

    /// Records a mouse button transition.
    pub fn on_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => self.drag = if pressed { Drag::Orbit } else { Drag::None },
            MouseButton::Right => self.drag = if pressed { Drag::Pan } else { Drag::None },
            MouseButton::Middle => {
                self.drag = if pressed { Drag::Attract } else { Drag::None };
                // Holding the middle button is a momentary override, so the selected mode survives it.
                self.active = pressed;
            }
            _ => {}
        }
    }

    /// Accumulates scroll, in lines. Trackpad pixel deltas are converted at ~50 pixels per notch.
    pub fn on_scroll(&mut self, delta: MouseScrollDelta) {
        self.scroll += match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(p) => {
                #[allow(clippy::cast_possible_truncation)]
                let lines = p.y as f32 / 50.0;
                lines
            }
        };
    }

    /// Records a key press. Repeats are ignored so that holding `tab` does not thrash the world.
    pub fn on_key(&mut self, key: &Key, repeat: bool) {
        if repeat {
            return;
        }
        match key {
            Key::Named(NamedKey::Tab) => self.toggle_world = true,
            Key::Named(NamedKey::Space) => self.paused = !self.paused,
            Key::Named(NamedKey::Escape) => self.quit = true,
            Key::Character(c) => match c.as_str() {
                "1" => {
                    self.mode = InteractionMode::Attract;
                    self.active = true;
                }
                "2" => {
                    self.mode = InteractionMode::Repel;
                    self.active = true;
                }
                "0" => {
                    self.mode = InteractionMode::Off;
                    self.active = false;
                }
                "r" | "R" => self.reset = true,
                _ => {}
            },
            _ => {}
        }
    }

    /// Translates a window event into a primitive transition.
    ///
    /// Returns whether the event was one this state machine handles, which the app uses to decide
    /// whether a redraw is warranted.
    pub fn handle(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                #[allow(clippy::cast_possible_truncation)]
                let pos = Vec2::new(position.x as f32, position.y as f32);
                self.on_cursor_move(pos);
                true
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.on_button(*button, *state == ElementState::Pressed);
                true
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.on_scroll(*delta);
                true
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed {
                    return false;
                }
                self.on_key(&event.logical_key, event.repeat);
                true
            }
            _ => false,
        }
    }

    /// Applies the accumulated motion and scroll to the camera and drains the one-shot flags.
    ///
    /// Clearing here, rather than in the caller, is what keeps the "accumulate in the event handler,
    /// consume exactly once per frame" contract in one place. A key press that was not drained this
    /// frame would otherwise be applied again on the next one.
    pub fn consume(&mut self, camera: &mut OrbitCamera) -> FrameActions {
        let actions = FrameActions {
            toggle_world: std::mem::take(&mut self.toggle_world),
            reset: std::mem::take(&mut self.reset),
            quit: std::mem::take(&mut self.quit),
        };

        match self.drag {
            Drag::Orbit => camera.orbit(self.motion),
            Drag::Pan => camera.pan(self.motion),
            Drag::Attract | Drag::None => {}
        }
        if self.scroll != 0.0 {
            camera.zoom(self.scroll);
        }

        self.motion = Vec2::ZERO;
        self.scroll = 0.0;
        actions
    }

    /// Whether the cursor should influence the swarm this frame.
    #[must_use]
    pub fn interaction_enabled(&self) -> bool {
        self.active && self.mode != InteractionMode::Off && self.cursor.is_some()
    }
}

/// One-shot actions for the frame, drained by the consumer.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameActions {
    /// Switch between the underwater and sky worlds.
    pub toggle_world: bool,
    /// Respawn the swarm.
    pub reset: bool,
    /// Exit.
    pub quit: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interaction_requires_a_cursor_a_mode_and_activation() {
        let mut input = InputState::default();
        // No cursor position yet, so nothing can influence the swarm.
        assert!(!input.interaction_enabled());

        input.on_cursor_move(Vec2::new(10.0, 10.0));
        assert!(input.interaction_enabled(), "default mode is attract");

        input.on_key(&Key::Character("0".into()), false);
        assert!(!input.interaction_enabled(), "mode 0 must disable influence");

        input.on_key(&Key::Character("2".into()), false);
        assert_eq!(input.mode, InteractionMode::Repel);
        assert!(input.interaction_enabled());
    }

    #[test]
    fn middle_button_overrides_activation_without_losing_the_mode() {
        let mut input = InputState::default();
        input.on_cursor_move(Vec2::ZERO);
        input.on_key(&Key::Character("2".into()), false);

        input.on_button(MouseButton::Middle, true);
        assert!(input.interaction_enabled());
        input.on_button(MouseButton::Middle, false);
        assert!(!input.interaction_enabled());
        assert_eq!(
            input.mode,
            InteractionMode::Repel,
            "the selected mode must survive the gesture"
        );
    }

    #[test]
    fn first_cursor_sample_produces_no_motion() {
        let mut input = InputState::default();
        let mut camera = OrbitCamera::default();
        input.drag = Drag::Orbit;

        input.on_cursor_move(Vec2::new(500.0, 400.0));
        assert_eq!(input.motion, Vec2::ZERO, "the first sample is a baseline, not motion");
        let yaw = camera.yaw;
        input.consume(&mut camera);
        assert_eq!(camera.yaw, yaw, "no motion means no rotation");

        input.on_cursor_move(Vec2::new(520.0, 400.0));
        assert_eq!(input.motion, Vec2::new(20.0, 0.0));
    }

    #[test]
    fn motion_accumulates_and_is_consumed_once() {
        let mut input = InputState::default();
        let mut camera = OrbitCamera::default();
        input.on_cursor_move(Vec2::new(100.0, 100.0));
        input.drag = Drag::Orbit;

        // Three successive 20-pixel steps: the deltas must sum, not overwrite. Reporting the same
        // position three times would produce one delta, which is what an accumulate-by-assignment bug
        // would also produce, so the positions have to advance.
        for step in 1..=3 {
            input.on_cursor_move(Vec2::new(100.0 + 20.0 * step as f32, 100.0));
        }
        assert_eq!(input.motion.x, 60.0, "three 20-pixel steps should accumulate");

        let yaw_before = camera.yaw;
        input.consume(&mut camera);
        assert_ne!(camera.yaw, yaw_before, "the orbit drag should have rotated the camera");
        assert_eq!(input.motion, Vec2::ZERO, "motion must be cleared after consumption");

        // A second consume with no new events must not move the camera again.
        let yaw_after = camera.yaw;
        input.consume(&mut camera);
        assert_eq!(camera.yaw, yaw_after, "a drained input must not be applied twice");
    }

    #[test]
    fn scroll_accumulates_and_zooms_in() {
        let mut input = InputState::default();
        let mut camera = OrbitCamera::default();
        let before = camera.distance;
        for _ in 0..4 {
            input.on_scroll(MouseScrollDelta::LineDelta(0.0, 1.0));
        }
        assert_eq!(input.scroll, 4.0);
        input.consume(&mut camera);
        assert!(camera.distance < before, "scrolling up should zoom in");
        assert_eq!(input.scroll, 0.0);
    }

    #[test]
    fn one_shot_flags_survive_until_consumed() {
        let mut input = InputState::default();
        let mut camera = OrbitCamera::default();
        input.on_key(&Key::Named(NamedKey::Tab), false);
        input.on_key(&Key::Character("r".into()), false);

        let actions = input.consume(&mut camera);
        assert!(actions.toggle_world && actions.reset);

        let actions = input.consume(&mut camera);
        assert!(
            !actions.toggle_world && !actions.reset,
            "one-shot flags must not repeat"
        );
    }

    #[test]
    fn key_repeats_are_ignored() {
        let mut input = InputState::default();
        let mut camera = OrbitCamera::default();
        input.on_key(&Key::Named(NamedKey::Tab), true);
        assert!(!input.consume(&mut camera).toggle_world);
    }

    #[test]
    fn space_toggles_pause_both_ways() {
        let mut input = InputState::default();
        assert!(!input.paused);
        input.on_key(&Key::Named(NamedKey::Space), false);
        assert!(input.paused);
        input.on_key(&Key::Named(NamedKey::Space), false);
        assert!(!input.paused);
    }
}
