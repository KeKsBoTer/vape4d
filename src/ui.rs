use std::{ops::RangeInclusive, time::Duration};

use crate::WindowContext;
use cgmath::{Euler, Matrix3, Quaternion};
#[cfg(not(target_arch = "wasm32"))]
use egui::Vec2b;
use egui::{emath::Numeric, epaint::Shadow, Align2, Color32, Pos2, RichText, Vec2, Visuals};

pub(crate) fn ui(state: &mut WindowContext) {
    let ctx = state.ui_renderer.winit.egui_ctx();
    egui::Window::new("test").show(ctx, |ui| {
        egui::Grid::new("render_settings")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("progress");
                ui.add(
                    egui::Slider::new(&mut state.render_settings.time, (0.)..=(1.))
                        .clamp_to_range(true),
                );
                ui.end_row();

                ui.label("Clipping Box Min");
                ui.horizontal(|ui| {
                    let aabb_min = &mut state.render_settings.clipping_aabb.min;
                    ui.add(
                        egui::DragValue::new(&mut aabb_min.x)
                            .speed(0.01)
                            .suffix("x")
                            .clamp_range((0.)..=(1.)),
                    );
                    ui.add(
                        egui::DragValue::new(&mut aabb_min.y)
                            .speed(0.01)
                            .suffix("y")
                            .clamp_range((0.)..=(1.)),
                    );
                    ui.add(
                        egui::DragValue::new(&mut aabb_min.z)
                            .speed(0.01)
                            .suffix("z")
                            .clamp_range((0.)..=(1.)),
                    );
                });
                ui.end_row();

                ui.label("Clipping Box Max");
                ui.horizontal(|ui| {
                    let aabb_max = &mut state.render_settings.clipping_aabb.max;
                    ui.add(
                        egui::DragValue::new(&mut aabb_max.x)
                            .speed(0.01)
                            .suffix("x")
                            .clamp_range((0.)..=(1.)),
                    );
                    ui.add(
                        egui::DragValue::new(&mut aabb_max.y)
                            .speed(0.01)
                            .suffix("y")
                            .clamp_range((0.)..=(1.)),
                    );
                    ui.add(
                        egui::DragValue::new(&mut aabb_max.z)
                            .speed(0.01)
                            .suffix("z")
                            .clamp_range((0.)..=(1.)),
                    );
                });
                ui.end_row();

                ui.label("DDA");
                ui.checkbox(&mut state.render_settings.use_dda, "");
                ui.end_row();

                ui.label("Step Size");
                ui.add(
                    egui::DragValue::new(&mut state.render_settings.step_size)
                        .speed(0.01)
                        .clamp_range((1e-4)..=(1.)),
                );
            });
    });

    egui::Window::new("⚙ Debug Visualization").show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
            for (name, group) in state.debug_lines.all_lines() {
                ui.checkbox(&mut group.visible, name.to_string());
            }
            if ui.button("Organize windows").clicked() {
                ui.ctx().memory_mut(|mem| mem.reset_areas());
            }
        });
    });
    egui::Window::new("Volume Info").show(ctx, |ui| {
        egui::Grid::new("volume_info")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                // ui.label("timesteps");
                // ui.label(state.volume.timesteps.to_string());
                // ui.end_row();
                // ui.label("resolution");
                // ui.label(state.volume.timesteps.to_string());
                // ui.end_row();
            });
    });
}
