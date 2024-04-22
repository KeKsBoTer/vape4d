use std::time::Duration;

use egui::vec2;
use egui_plot::{Plot, PlotImage, PlotPoint};

use crate::{
    cmap::{rasterize_tf, ColorMap, COLORMAPS},
    WindowContext,
};

pub(crate) fn ui(state: &mut WindowContext) {
    let ctx = state.ui_renderer.winit.egui_ctx();
    let with_animation = state.volumes[0].volume.timesteps > 1;
    egui::Window::new("Render Settings").show(ctx, |ui| {
        egui::Grid::new("render_settings")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                if with_animation {
                    ui.label("Progress");
                    ui.add(
                        egui::Slider::new(&mut state.render_settings.time, (0.)..=(1.))
                            .clamp_to_range(true),
                    );
                    if ui.button(if state.playing { "||" } else { "▶" }).clicked() {
                        state.playing = !state.playing;
                    }
                    ui.end_row();
                    ui.label("Duration");
                    ui.add(
                        egui::DragValue::from_get_set(|v| {
                            if let Some(v) = v {
                                state.animation_duration = Duration::from_secs_f64(v);
                                return v;
                            } else {
                                return state.animation_duration.as_secs_f64();
                            }
                        })
                        .suffix("s")
                        .clamp_range((0.)..=1000.),
                    );
                    ui.end_row();
                }

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

                ui.label("Step Size");
                ui.add(
                    egui::DragValue::new(&mut state.render_settings.step_size)
                        .speed(0.01)
                        .clamp_range((1e-3)..=(0.1)),
                );
                ui.end_row();

                ui.label("Distance Scale");
                ui.add(
                    egui::DragValue::new(&mut state.render_settings.distance_scale)
                        .speed(0.01)
                        .clamp_range((1e-4)..=(100000.)),
                );
                ui.end_row();
                if state.volumes.len() > 1 {
                    ui.label("Channel");
                    egui::ComboBox::new("selected_channel", "")
                        .selected_text(
                            state
                                .selected_channel
                                .map_or("All".to_string(), |v| v.to_string()),
                        )
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut state.selected_channel, None, "All");
                            for i in 0..state.volumes.len() {
                                ui.selectable_value(
                                    &mut state.selected_channel,
                                    Some(i),
                                    format!("{}", i),
                                );
                            }
                        });
                    ui.end_row();
                    if state.selected_channel.is_none() {
                        ui.label("Number of Rows");
                        let max_rows = state.volumes.len();
                        ui.add(
                            egui::DragValue::new(&mut state.num_columns)
                                .speed(0.01)
                                .clamp_range(1..=max_rows),
                        );
                        ui.end_row();
                    }
                }
            });
    });

    let mut cmap = state.cmap.color_map().clone();

    egui::Window::new("Transfer Function")
        .default_size(vec2(300., 50.))
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("vmax");
                ui.add(
                    egui::DragValue::new(&mut state.render_settings.vmin)
                        .speed(0.01)
                        .clamp_range(
                            state.volumes[0].volume.min_value..=state.render_settings.vmax,
                        ),
                );
                ui.label("vmin");
                ui.add(
                    egui::DragValue::new(&mut state.render_settings.vmax)
                        .speed(0.01)
                        .clamp_range(
                            state.render_settings.vmin..=state.volumes[0].volume.max_value,
                        ),
                );
            });
            ui.horizontal(|ui| {
                let cmaps = &COLORMAPS;
                let mut selected_cmap: String = ui.ctx().data_mut(|d| {
                    d.get_persisted_mut_or("selected_cmap".into(), "viridis".to_string())
                        .clone()
                });
                ui.label("Colormap");
                egui::ComboBox::new("cmap_select", "")
                    .selected_text(selected_cmap.clone())
                    .show_ui(ui, |ui| {
                        let mut keys: Vec<_> = cmaps.iter().collect();
                        keys.sort_by_key(|e| e.0);
                        for (name, cmap) in keys {
                            let texture = load_or_create(ui, cmap);
                            ui.horizontal(|ui| {
                                ui.image(egui::ImageSource::Texture(egui::load::SizedTexture {
                                    id: texture,
                                    size: vec2(50., 10.),
                                }));
                                ui.selectable_value(&mut selected_cmap, name.clone(), name);
                            });
                        }
                    });
                cmap = cmaps[&selected_cmap].clone();
                ui.ctx()
                    .data_mut(|d| d.insert_persisted("selected_cmap".into(), selected_cmap));
            });
            ColorMapBuilder::new("cmap_builder").show(
                ui,
                &mut cmap,
                state.render_settings.vmin,
                state.render_settings.vmax,
            );
        });

    state
        .cmap
        .set_color_map(cmap, &state.wgpu_context.device, &state.wgpu_context.queue);

    egui::Window::new("Volume Info").show(ctx, |ui| {
        egui::Grid::new("volume_info")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("timesteps");
                ui.label(state.volumes[0].volume.timesteps.to_string());
                ui.end_row();
                ui.label("channels");
                ui.label(state.volumes.len().to_string());
                ui.end_row();
                ui.label("resolution");
                let res = state.volumes[0].volume.resolution;
                ui.label(format!("{}x{}x{} (WxDxH)", res.x, res.y, res.z));
                ui.end_row();
                ui.label("value range");
                ui.label(format!(
                    "[{} , {}]",
                    state.volumes[0].volume.min_value, state.volumes[0].volume.max_value
                ));
                ui.end_row();
            });
    });
}

use cgmath::Vector2;
use egui::{epaint::PathShape, *};

pub fn tf_ui(ui: &mut Ui, points: &mut Vec<Vector2<f32>>) -> egui::Response {
    let (response, painter) = ui.allocate_painter(
        vec2(ui.available_width(), 100.),
        Sense::hover().union(Sense::click()),
    );

    let to_screen = emath::RectTransform::from_to(
        Rect::from_two_pos(Pos2::ZERO, Pos2::new(1., 1.)),
        response.rect,
    );

    let stroke = Stroke::new(1.0, Color32::from_rgb(25, 200, 100));
    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let pp = to_screen.inverse().transform_pos(pos);
            let pp = Vector2::new(pp.x, 1. - pp.y);
            let idx = points
                .iter()
                .enumerate()
                .find_map(|p| if p.1.x > pp.x { Some(p.0) } else { None })
                .unwrap_or(points.len() - 1);
            points.insert(idx, pp);
        }
    }

    let control_point_radius = 8.0;

    let n = points.len();
    let mut new_points = Vec::with_capacity(n);
    let mut control_point_shapes = Vec::with_capacity(n);
    for (i, point) in points.iter().enumerate() {
        let size = Vec2::splat(2.0 * control_point_radius);
        let pos = pos2(point.x, 1. - point.y);
        let point_in_screen = to_screen.transform_pos(pos);
        let point_rect = Rect::from_center_size(point_in_screen, size);
        let point_id = response.id.with(i);
        let point_response = ui.interact(point_rect, point_id, Sense::drag().union(Sense::click()));

        let is_edge = i == 0 || i == n - 1;

        if !point_response.secondary_clicked() || is_edge {
            let e = 1e-3;
            let mut t = point_response.drag_delta();
            if is_edge {
                // cant move last and first point
                t.x = 0.;
            }
            // point cannot move past its neighbors
            let left = if i == 0 { 0. } else { points[i - 1].x + e };
            let right = if i == n - 1 { 1. } else { points[i + 1].x - e };
            let bbox = Rect::from_min_max(Pos2::new(left, 0.), Pos2::new(right, 1.));

            let mut new_point = pos2(point.x, 1. - point.y);
            new_point += to_screen.inverse().scale() * t;
            new_point = to_screen.from().intersect(bbox).clamp(new_point);
            new_points.push(Vector2::new(new_point.x, 1. - new_point.y));

            let point_in_screen = to_screen.transform_pos(new_point);
            let stroke = ui.style().interact(&point_response).fg_stroke;

            control_point_shapes.push(Shape::circle_stroke(
                point_in_screen,
                control_point_radius,
                stroke,
            ));
        }
    }
    points.drain(0..n);
    points.extend(new_points);

    let points_in_screen: Vec<Pos2> = points
        .iter()
        .map(|p| to_screen.transform_pos(pos2(p.x, 1. - p.y)))
        .collect();

    painter.add(PathShape::line(points_in_screen, stroke));
    painter.extend(control_point_shapes);
    response
}

pub struct ColorMapBuilder {
    id_source: Id,
}

impl ColorMapBuilder {
    pub fn new(id_source: impl std::hash::Hash) -> Self {
        Self {
            id_source: Id::new(id_source),
        }
    }

    fn show(self, ui: &mut Ui, color_map: &mut ColorMap, vmin: f32, vmax: f32) {
        let id = ui.make_persistent_id(self.id_source);
        let texture = load_or_create(ui, &color_map);
        let width = vmax - vmin;
        let height = width / 5.;
        let image = PlotImage::new(
            texture,
            PlotPoint::new(vmin + width * 0.5, height / 2.),
            vec2(width, height),
        );
        let plot = Plot::new(id)
            .show_x(true)
            .show_y(false)
            .height(100.)
            .show_background(false)
            .show_grid(false)
            .custom_y_axes(vec![])
            .allow_boxed_zoom(false)
            .allow_double_click_reset(false)
            .allow_drag(false)
            .allow_scroll(false)
            .allow_zoom(false);
        plot.show(ui, |plot_ui| {
            plot_ui.image(image);
        });
        ui.label("Alpha Channel");

        let mut alpha_tf: Vec<Vector2<f32>> = ui.data_mut(|d| {
            d.get_persisted_mut_or(
                id.with("alpha_tf"),
                vec![Vector2::new(0., 0.), Vector2::new(1., 1.)],
            )
            .clone()
        });

        tf_ui(ui, &mut alpha_tf);

        let alpha_values = rasterize_tf(&alpha_tf, color_map.values().len() as u32);
        color_map.update_alpha(&alpha_values);

        ui.data_mut(|d: &mut util::IdTypeMap| {
            d.insert_persisted(id.with("alpha_tf"), alpha_tf);
        });
    }
}

fn load_or_create(ui: &egui::Ui, cmap: &ColorMap) -> egui::TextureId {
    let id = Id::new(cmap);
    let tex: Option<egui::TextureHandle> = ui.ctx().data_mut(|d| d.get_temp(id));
    match tex {
        Some(tex) => tex.id(),
        None => {
            let width = cmap.values().len();
            let tex = ui.ctx().load_texture(
                id.value().to_string(),
                egui::ColorImage::from_rgba_premultiplied(
                    [width, 1],
                    bytemuck::cast_slice(cmap.values()),
                ),
                egui::TextureOptions::LINEAR,
            );
            let tex_id = tex.id();
            ui.ctx().data_mut(|d| d.insert_temp(id, tex));
            return tex_id;
        }
    }
}
