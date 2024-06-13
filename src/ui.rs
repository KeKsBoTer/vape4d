use std::{f32::consts::PI, ops::RangeInclusive, time::Duration};

use egui::{emath::Numeric, epaint::TextShape, vec2};
use egui_plot::{Plot, PlotImage, PlotPoint};

use crate::{
    cmap::{ColorMap, COLORMAP_RESOLUTION},
    WindowContext,
};

#[cfg(feature = "colormaps")]
use crate::cmap::COLORMAPS;

pub(crate) fn ui(state: &mut WindowContext) {
    let ctx = state.ui_renderer.winit.egui_ctx();
    let with_animation = state.volumes[0].volume.timesteps > 1;
    egui::Window::new("Render Settings").show(ctx, |ui| {
        egui::Grid::new("render_settings")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                if with_animation {
                    ui.label("Time");
                    ui.add(
                        egui::Slider::new(&mut state.render_settings.time, (0.)..=(1.))
                            .clamp_to_range(true),
                    );
                    if ui.button(if state.playing { "||" } else { "▶" }).clicked() {
                        state.playing = !state.playing;
                    }
                    ui.end_row();
                    ui.label("Animation Duration");
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
                ui.label("Background Color");
                let mut bg = [
                    state.background_color.r as f32,
                    state.background_color.g as f32,
                    state.background_color.b as f32,
                    state.background_color.a as f32,
                ];
                ui.color_edit_button_rgba_premultiplied(&mut bg);
                state.background_color = wgpu::Color {
                    r: bg[0] as f64,
                    g: bg[1] as f64,
                    b: bg[2] as f64,
                    a: bg[3] as f64,
                };
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

                ui.label("Spatial Interpolation");

                egui::ComboBox::new("spatial_interpolation", "")
                    .selected_text(match state.render_settings.temporal_filter {
                        wgpu::FilterMode::Nearest => "Nearest",
                        wgpu::FilterMode::Linear => "Linear",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.render_settings.spatial_filter,
                            wgpu::FilterMode::Nearest,
                            "Nearest",
                        );
                        ui.selectable_value(
                            &mut state.render_settings.spatial_filter,
                            wgpu::FilterMode::Linear,
                            "Linear",
                        )
                    });
                ui.end_row();
                ui.label("Temporal Interpolation");
                egui::ComboBox::new("temporal_interpolation", "")
                    .selected_text(match state.render_settings.temporal_filter {
                        wgpu::FilterMode::Nearest => "Nearest",
                        wgpu::FilterMode::Linear => "Linear",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.render_settings.temporal_filter,
                            wgpu::FilterMode::Nearest,
                            "Nearest",
                        );
                        ui.selectable_value(
                            &mut state.render_settings.temporal_filter,
                            wgpu::FilterMode::Linear,
                            "Linear",
                        )
                    });
                ui.end_row();
            });
    });

    // let mut cmap = state.cmap.clone();

    if state.colormap_editor_visible {
        egui::Window::new("Transfer Function")
            .default_size(vec2(300., 50.))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("vmin");
                    let min_b = state
                        .render_settings
                        .vmin
                        .unwrap_or(state.volumes[0].volume.min_value);
                    let max_b = state
                        .render_settings
                        .vmax
                        .unwrap_or(state.volumes[0].volume.max_value);

                    let vmin_min = state.volumes[0].volume.min_value.min(min_b);
                    let vmax_max = state.volumes[0].volume.max_value.max(max_b);
                    optional_drag(
                        ui,
                        &mut state.render_settings.vmin,
                        Some(vmin_min..=max_b),
                        Some(0.01),
                        Some(vmin_min),
                    );
                    ui.label("vmax");
                    optional_drag(
                        ui,
                        &mut state.render_settings.vmax,
                        Some(min_b..=vmax_max),
                        Some(0.01),
                        Some(vmax_max),
                    );
                });
                #[cfg(feature = "colormaps")]
                if state.cmap_select_visible {
                    ui.horizontal(|ui| {
                        let cmaps = &COLORMAPS;
                        let mut selected_cmap: (String, String) = ui.ctx().data_mut(|d| {
                            d.get_persisted_mut_or(
                                "selected_cmap".into(),
                                ("seaborn".to_string(), "icefire".to_string()),
                            )
                            .clone()
                        });
                        let mut search_term: String = ui.ctx().data_mut(|d| {
                            d.get_temp_mut_or("cmap_search".into(), "".to_string())
                                .clone()
                        });
                        ui.label("Colormap");
                        let old_selected_cmap = selected_cmap.clone();
                        egui::ComboBox::new("cmap_select", "")
                            .selected_text(selected_cmap.1.clone())
                            .show_ui(ui, |ui| {
                                ui.add(
                                    egui::text_edit::TextEdit::singleline(&mut search_term)
                                        .hint_text("Search..."),
                                );
                                for (group, cmaps) in cmaps.iter() {
                                    ui.label(group);
                                    let mut sorted_cmaps: Vec<_> = cmaps.iter().collect();
                                    sorted_cmaps.sort_by_key(|e| e.0);
                                    for (name, cmap) in sorted_cmaps {
                                        if name.contains(&search_term) {
                                            let texture =
                                                load_or_create(ui, cmap, COLORMAP_RESOLUTION);
                                            ui.horizontal(|ui| {
                                                ui.image(egui::ImageSource::Texture(
                                                    egui::load::SizedTexture {
                                                        id: texture,
                                                        size: vec2(50., 10.),
                                                    },
                                                ));
                                                ui.selectable_value(
                                                    &mut selected_cmap,
                                                    (group.clone(), name.clone()),
                                                    name,
                                                );
                                            });
                                        }
                                    }
                                    ui.separator();
                                }
                            });
                        if old_selected_cmap != selected_cmap {
                            let old_alpha = state.cmap.a.clone();
                            state.cmap = cmaps[&selected_cmap.0][&selected_cmap.1]
                                .into_linear_segmented(COLORMAP_RESOLUTION);
                            if state.cmap.a.is_none()
                                || cmaps[&selected_cmap.0][&selected_cmap.1]
                                    .has_boring_alpha_channel()
                            {
                                state.cmap.a = old_alpha;
                            }
                            ui.ctx().data_mut(|d| {
                                d.insert_persisted("selected_cmap".into(), selected_cmap);
                            });
                        }
                        ui.ctx()
                            .data_mut(|d| d.insert_temp("cmap_search".into(), search_term));
                        if state.cmap.a.is_none() {
                            state.cmap.a = Some(vec![(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)]);
                        }
                        if ui.button("↔").clicked() {
                            state.cmap = (&state.cmap).reverse();
                        }
                    });
                }
                let vmin = state
                    .render_settings
                    .vmin
                    .unwrap_or(state.volumes[0].volume.min_value);
                let vmax = state
                    .render_settings
                    .vmax
                    .unwrap_or(state.volumes[0].volume.max_value);
                show_cmap(ui, egui::Id::new("cmap preview"), &state.cmap, vmin, vmax);

                ui.heading("Alpha Channel");
                ui.end_row();
                ui.horizontal_wrapped(|ui| {
                    ui.label("Presets:");
                    let v_hack = ui
                        .button("\\/")
                        .on_hover_text("double click for smooth version");
                    if v_hack.clicked() {
                        state.cmap.a = Some(vec![(0.0, 1.0, 1.0), (0.5, 0., 0.), (1.0, 1.0, 1.0)]);
                    }
                    if v_hack.double_clicked() {
                        state.cmap.a =
                            Some(build_segments(25, |x| ((x * 2. * PI).cos() + 1.) / 2.));
                    }
                    let slope_hack = ui
                        .button("/")
                        .on_hover_text("double click for smooth version");
                    if slope_hack.clicked() {
                        state.cmap.a = Some(build_segments(2, |x| (-(x * PI).cos() + 1.) / 2.));
                    }
                    if slope_hack.double_clicked() {
                        state.cmap.a = Some(build_segments(25, |x| (-(x * PI).cos() + 1.) / 2.));
                    }
                    let double_v_hack = ui
                        .button("/\\/\\")
                        .on_hover_text("double click for smooth version");
                    if double_v_hack.clicked() {
                        state.cmap.a =
                            Some(build_segments(5, |x| (-(x * 4. * PI).cos() + 1.) / 2.));
                    }
                    if double_v_hack.double_clicked() {
                        state.cmap.a =
                            Some(build_segments(25, |x| (-(x * 4. * PI).cos() + 1.) / 2.));
                    }
                    if ui.button("-").clicked() {
                        state.cmap.a = Some(vec![(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)]);
                    }
                });

                ui.separator();

                if let Some(a) = &mut state.cmap.a {
                    tf_ui(ui, a)
                    .on_hover_text("Drag anchor points to change transfer function.\nLeft-Click for new anchor point.\nRight-Click to delete anchor point.");
                }
                ui.end_row();
                #[cfg(not(target_arch = "wasm32"))]
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut state.cmap_save_path);
                    if ui.button("Save").clicked() {
                        let file = std::fs::File::create(state.cmap_save_path.clone()).unwrap();
                        serde_json::to_writer(file, &state.cmap).unwrap();
                    }
                });
            });
    }
    state
        .cmap_gpu
        .update(&state.wgpu_context.queue, &state.cmap);

    if state.volume_info_visible {
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

    let frame_rect = ctx.available_rect();
    egui::Area::new(egui::Id::new("orientation"))
        .fixed_pos(Pos2::new(frame_rect.left(), frame_rect.bottom()))
        .anchor(Align2::LEFT_BOTTOM, Vec2::new(0., 0.))
        .interactable(false)
        .order(Order::Background)
        .show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                vec2(100., 100.),
                Sense {
                    click: false,
                    drag: false,
                    focusable: false,
                },
            );

            let to_screen = emath::RectTransform::from_to(
                Rect::from_two_pos(Pos2::new(-1.2, -1.2), Pos2::new(1.2, 1.2)),
                response.rect,
            );
            let x_color = Color32::RED;
            let y_color = Color32::GREEN;
            let z_color = Color32::BLUE;

            let view_matrix = state.camera.view_matrix();
            let x_axis = view_matrix.transform_vector(Vector3::unit_x());
            // multiply with -1 because in egui origin is top left
            let y_axis = -view_matrix.transform_vector(Vector3::unit_y());
            let z_axis = view_matrix.transform_vector(Vector3::unit_z());
            let origin = view_matrix.transform_vector(Vector3::zero());

            let axes = [
                (x_axis, x_color, "X"),
                (y_axis, y_color, "Y"),
                (z_axis, z_color, "Z"),
            ];
            let depth = vec![-x_axis.z, -y_axis.z, -z_axis.z];
            let draw_order = argsort(&depth);
            for i in draw_order.iter() {
                let (axis, color, label) = axes[*i];
                let pos = to_screen.transform_pos(pos2(axis.x, axis.y));
                painter.add(PathShape::line(
                    vec![to_screen.transform_pos(pos2(origin.x, origin.y)), pos],
                    Stroke::new(3., color),
                ));
                painter.add(TextShape::new(
                    pos,
                    painter.layout_no_wrap(label.to_string(), FontId::default(), color),
                    color,
                ));
            }
        });
}

pub fn argsort<T: PartialOrd>(data: &[T]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..data.len()).collect::<Vec<_>>();
    indices.sort_by(|&a, &b| data[a].partial_cmp(&data[b]).unwrap());
    indices
}

use cgmath::{Transform, Vector3, Zero};
use egui::{epaint::PathShape, *};

pub fn tf_ui(ui: &mut Ui, points: &mut Vec<(f32, f32, f32)>) -> egui::Response {
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
            let pp = (pp.x, 1. - pp.y, 1. - pp.y);
            let idx = points
                .iter()
                .enumerate()
                .find_map(|p| if p.1 .0 > pp.0 { Some(p.0) } else { None })
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
        let pos = pos2(point.0, 1. - point.1);
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
            let left = if i == 0 { 0. } else { points[i - 1].0 + e };
            let right = if i == n - 1 { 1. } else { points[i + 1].0 - e };
            let bbox = Rect::from_min_max(Pos2::new(left, 0.), Pos2::new(right, 1.));

            let mut new_point = pos2(point.0, 1. - point.1);
            new_point += to_screen.inverse().scale() * t;
            new_point = to_screen.from().intersect(bbox).clamp(new_point);
            new_points.push((new_point.x, 1. - new_point.y, 1. - new_point.y));

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
        .map(|p| to_screen.transform_pos(pos2(p.0, 1. - p.1)))
        .collect();

    painter.add(PathShape::line(points_in_screen, stroke));
    painter.extend(control_point_shapes);
    response
}

use std::hash::Hash;
fn load_or_create<C>(ui: &egui::Ui, cmap: C, n: u32) -> egui::TextureId
where
    C: ColorMap + Hash,
{
    let id = Id::new(&cmap);
    let tex: Option<egui::TextureHandle> = ui.ctx().data_mut(|d| d.get_temp(id));
    match tex {
        Some(tex) => tex.id(),
        None => {
            let tex = ui.ctx().load_texture(
                id.value().to_string(),
                egui::ColorImage::from_rgba_unmultiplied(
                    [n as usize, 1],
                    bytemuck::cast_slice(&cmap.rasterize(n as usize)),
                ),
                egui::TextureOptions::LINEAR,
            );
            let tex_id = tex.id();
            ui.ctx().data_mut(|d| d.insert_temp(id, tex));
            return tex_id;
        }
    }
}

fn optional_drag<T: Numeric>(
    ui: &mut egui::Ui,
    opt: &mut Option<T>,
    range: Option<RangeInclusive<T>>,
    speed: Option<impl Into<f64>>,
    default: Option<T>,
) {
    let mut placeholder = default.unwrap_or(T::from_f64(0.));
    let mut drag = if let Some(ref mut val) = opt {
        egui_winit::egui::DragValue::new(val)
    } else {
        egui_winit::egui::DragValue::new(&mut placeholder).custom_formatter(|_, _| {
            if let Some(v) = default {
                format!("{:.2}", v.to_f64())
            } else {
                "—".into()
            }
        })
    };
    if let Some(range) = range {
        drag = drag.clamp_range(range);
    }
    if let Some(speed) = speed {
        drag = drag.speed(speed);
    }
    let changed = ui.add(drag).changed();
    if ui
        .add_enabled(opt.is_some(), egui::Button::new("↺"))
        .on_hover_text("Reset to default")
        .clicked()
    {
        *opt = None;
    }
    if changed && opt.is_none() {
        *opt = Some(placeholder);
    }
}

fn show_cmap(ui: &mut egui::Ui, id: egui::Id, cmap: impl ColorMap + Hash, vmin: f32, vmax: f32) {
    let texture = load_or_create(ui, cmap, COLORMAP_RESOLUTION);
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
}

fn build_segments<F: Fn(f32) -> f32>(n: usize, f: F) -> Vec<(f32, f32, f32)> {
    (0..n)
        .map(|i| {
            let x = i as f32 / (n as f32 - 1.);
            let v = f(x);
            (x, v, v)
        })
        .collect()
}
