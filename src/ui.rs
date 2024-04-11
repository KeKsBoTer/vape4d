use std::time::Duration;

use egui::vec2;
use egui_plot::{Plot, PlotImage, PlotPoint, PlotPoints};

use crate::{
    cmap::{rasterize_tf, ColorMap},
    WindowContext,
};

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
                        .clamp_range((1e-4)..=(1.)),
                );
                ui.end_row();

                ui.label("Distance Scale");
                ui.add(
                    egui::DragValue::new(&mut state.render_settings.distance_scale)
                        .speed(0.01)
                        .clamp_range((1e-4)..=(100000.)),
                );
                ui.end_row();

                ui.label("Temporal Interpolation");
                egui::ComboBox::new("filter_temporal", "")
                    .selected_text(filter_mode_name(&state.render_settings.temporal_filter))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.render_settings.temporal_filter,
                            wgpu::FilterMode::Nearest,
                            filter_mode_name(&wgpu::FilterMode::Nearest),
                        );
                        ui.selectable_value(
                            &mut state.render_settings.temporal_filter,
                            wgpu::FilterMode::Linear,
                            filter_mode_name(&wgpu::FilterMode::Linear),
                        );
                    });
                ui.end_row();

                ui.label("Spatial Interpolation");
                egui::ComboBox::new("filter_spatial", "")
                    .selected_text(filter_mode_name(&state.render_settings.spatial_filter))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.render_settings.spatial_filter,
                            wgpu::FilterMode::Nearest,
                            filter_mode_name(&wgpu::FilterMode::Nearest),
                        );
                        ui.selectable_value(
                            &mut state.render_settings.spatial_filter,
                            wgpu::FilterMode::Linear,
                            filter_mode_name(&wgpu::FilterMode::Linear),
                        );
                    });
                ui.end_row();

                ui.label("Colormap");
                egui::ComboBox::new("cmap_select", "")
                    .selected_text(state.selected_cmap.as_str())
                    .show_ui(ui, |ui| {
                        let mut keys: Vec<_> = state.cmaps.iter().collect();
                        keys.sort_by_key(|e| e.0);
                        for (name, (_, texture)) in keys {
                            ui.horizontal(|ui| {
                                ui.image(egui::ImageSource::Texture(egui::load::SizedTexture {
                                    id: *texture,
                                    size: vec2(50., 10.),
                                }));
                                ui.selectable_value(&mut state.selected_cmap, name.clone(), name);
                            });
                        }
                    });
            });
    });

    egui::Window::new("Transfer Function")
        .default_size(vec2(300., 50.))
        .show(ctx, |ui| {
            let min_value = state.volume.min_value;
            let max_value = state.volume.max_value;
            let width = max_value - min_value;
            let height = width / 5.;
            let (cmap, texture) = state.cmaps.get(&state.selected_cmap).unwrap();
            let image = PlotImage::new(
                *texture,
                PlotPoint::new(min_value + width * 0.5, height / 2.),
                vec2(width, height),
            );

            let plot = Plot::new("items_demo")
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
                plot_ui.image(image.name("Image"));
            });
            tf_ui(ui, &mut state.alpha_tf);
        });

    let alpha_values = rasterize_tf(&state.alpha_tf, 256);
    let new_cmap = state.cmaps[&state.selected_cmap]
        .0
        .values
        .iter()
        .zip(alpha_values)
        .map(|(v, a)| Vector4::new(v[0], v[1], v[2], a))
        .collect();
    state.cmap = ColorMap::new(
        &state.wgpu_context.device,
        &state.wgpu_context.queue,
        new_cmap,
    );

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
}

fn filter_mode_name(mode: &wgpu::FilterMode) -> &'static str {
    match mode {
        wgpu::FilterMode::Nearest => "Nearest",
        wgpu::FilterMode::Linear => "Linear",
    }
}

use cgmath::{Vector2, Vector4};
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
