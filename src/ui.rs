use std::time::Duration;

use egui::vec2;
use egui_plot::{AxisHints, Corner, Legend, Plot, PlotImage, PlotPoint};

use crate::WindowContext;

pub(crate) fn ui(state: &mut WindowContext) {
    let ctx = state.ui_renderer.winit.egui_ctx();
    egui::Window::new("Render Settings").show(ctx, |ui| {
        egui::Grid::new("render_settings")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Progress");
                ui.add(
                    egui::Slider::new(&mut state.render_settings.time, (0.)..=(1.))
                        .clamp_to_range(true),
                );
                if ui.button(if state.playing{"||"}else{"▶"}).clicked() {
                    state.playing = !state.playing;
                }
                ui.end_row();
                ui.label("Duration");
                ui.add(egui::DragValue::from_get_set(|v|{
                    if let Some(v) = v {
                        state.animation_duration = Duration::from_secs_f64(v);
                        return v;
                    }else{
                        return state.animation_duration.as_secs_f64();
                    }
                }).suffix("s").clamp_range((0.)..=1000.));
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
                ui.end_row();

                ui.label("Distance Scale");
                ui.add(
                    egui::DragValue::new(&mut state.render_settings.distance_scale)
                        .speed(0.01)
                        .clamp_range((1e-4)..=(1000.)),
                );
                ui.end_row();

                ui.label("Colormap");
                egui::ComboBox::new("cmap_select", "")
                    .selected_text(state.selected_cmap.as_str())
                    .show_ui(ui, |ui| {
                        let mut keys:Vec<_> = state.cmaps.iter().collect();
                        keys.sort_by_key(|e|e.0);
                        for (name,(_,texture)) in keys {
                            ui.horizontal(|ui| {
                                ui.image(egui::ImageSource::Texture(egui::load::SizedTexture{ id: *texture, size: vec2(50., 10.) }));
                                ui.selectable_value(&mut state.selected_cmap, name.clone(), name);
                            });
                        }
                    });
            });
    });

    egui::Window::new("Transfer Function").default_size(vec2(300., 50.)).show(ctx, |ui| {

        let min_value = state.volume.min_value; 
        let max_value = state.volume.max_value; 
        let width = max_value-  min_value;
        let egui_texture = state.cmaps.get(&state.selected_cmap).unwrap().1;
        let image = PlotImage::new(
            egui_texture, 
            PlotPoint::new( min_value+width*0.5, width/10.),
             vec2(width, width/5.));

        let plot = Plot::new("items_demo")
            .show_x(true)
            .show_y(false)
            .show_background(false).show_grid(false)
            .custom_y_axes(vec![]);
        plot.show(ui, |plot_ui| {
            plot_ui.image(image.name("Image"));
        })
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
                ui.label("timesteps");
                ui.label(state.volume.timesteps.to_string());
                ui.end_row();
                ui.label("resolution");
                let res = state.volume.resolution;
                ui.label(format!("{}x{}x{} (WxDxH)", res.x, res.y, res.z));
                ui.end_row();
            });
    });
    state
        .debug_lines
        .update_clipping_box(&state.render_settings.clipping_aabb);
}
