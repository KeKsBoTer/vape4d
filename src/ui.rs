use std::{f32::consts::PI, ops::RangeInclusive, time::Duration};

use egui::{emath::Numeric, epaint::TextShape, vec2};
use egui_plot::{Plot, PlotImage, PlotPoint};
use egui_probe::EguiProbe;

use crate::{
    cmap::{ColorMap, COLORMAP_RESOLUTION},
    WindowContext,
};

use crate::cmap::COLORMAPS;

pub(crate) fn ui(state: &mut WindowContext) -> bool {
    let ctx = state.ui_renderer.winit.egui_ctx();
    let with_animation = state.volume.volume.timesteps() > 1;
    let render_scale_before = state.render_settings.render_scale;
    egui::Window::new("Render Settings").show(ctx, |ui| {
        egui::Grid::new("render_settings")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                if with_animation {
                    ui.label("Time");
                    ui.add(
                        egui::Slider::new(&mut state.render_settings.time, (0.)..=(1.))
                            .clamping(SliderClamping::Always)
                            .custom_formatter(|v, _| ((v * 100.).round() / 100.).to_string()), // fixed_decimals(2) modifies the value so do not use it here
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
                        .range((0.1)..=1000.),
                    );
                    ui.end_row();
                }

                // spatial filter is nearest we use DDA and do not need step size
                let uses_step_size =
                    state.render_settings.spatial_filter != wgpu::FilterMode::Nearest;

                ui.add_enabled(uses_step_size, egui::Label::new("Step Size"))
                    .on_hover_text("Step size for the raymarching algorithm");
                ui.add_enabled(
                    uses_step_size,
                    egui::DragValue::new(&mut state.render_settings.step_size)
                        .speed(0.01)
                        .range((1e-3)..=(0.1)),
                );
                ui.end_row();

                ui.label("Axis Scale")
                    .on_hover_text("Scale the volume in x, y and z direction");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut state.render_settings.axis_scale[0])
                            .speed(0.01)
                            .range(RangeInclusive::new(1., 1e2))
                            .clamp_existing_to_range(true)
                            .suffix("x"),
                    );
                    ui.add(
                        egui::DragValue::new(&mut state.render_settings.axis_scale[1])
                            .speed(0.01)
                            .range(RangeInclusive::new(1., 1e2))
                            .clamp_existing_to_range(true)
                            .suffix("y"),
                    );
                    ui.add(
                        egui::DragValue::new(&mut state.render_settings.axis_scale[2])
                            .speed(0.01)
                            .range(RangeInclusive::new(1., 1e2))
                            .clamp_existing_to_range(true)
                            .suffix("z"),
                    );
                });
                ui.end_row();

                ui.label("Density Scale").on_hover_text(
                    "Scale the density of the volume. A higher value makes the volume more opaque.",
                );
                ui.add(
                    egui::DragValue::new(&mut state.render_settings.distance_scale)
                        .speed(0.01)
                        .range((1e-4)..=(100000.)),
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
                if state.volume.volume.channels() > 1 {
                    ui.label("Channel");
                    egui::ComboBox::new("selected_channel", "")
                        .selected_text(
                            state
                                .selected_channel
                                .map_or("All".to_string(), |v| v.to_string()),
                        )
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut state.selected_channel, None, "All");
                            for i in 0..state.volume.volume.channels() {
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
                        let max_rows = state.volume.volume.channels();
                        ui.add(
                            egui::DragValue::new(&mut state.num_columns)
                                .range(1u32..=max_rows as u32),
                        );
                        ui.end_row();
                    }
                }

                ui.label("Spatial Interpolation");

                egui::ComboBox::new("spatial_interpolation", "")
                    .selected_text(match state.render_settings.spatial_filter {
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

                ui.label("Upscaling Method");
                egui::ComboBox::new("upscaling_method", "")
                    .selected_text(state.render_settings.upscaling_method.to_string())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.render_settings.upscaling_method,
                            crate::renderer::UpscalingMethod::Nearest,
                            crate::renderer::UpscalingMethod::Nearest.to_string()
                        );
                        ui.selectable_value(
                            &mut state.render_settings.upscaling_method,
                            crate::renderer::UpscalingMethod::Bilinear,
                            crate::renderer::UpscalingMethod::Bilinear.to_string()
                        );
                        ui.selectable_value(
                            &mut state.render_settings.upscaling_method,
                            crate::renderer::UpscalingMethod::Bicubic,
                            crate::renderer::UpscalingMethod::Bicubic.to_string()
                        );
                        ui.selectable_value(
                            &mut state.render_settings.upscaling_method,
                            crate::renderer::UpscalingMethod::Spline,
                            crate::renderer::UpscalingMethod::Spline.to_string()
                        );
                        ui.selectable_value(
                            &mut state.render_settings.upscaling_method,
                            crate::renderer::UpscalingMethod::Lanczos,
                            crate::renderer::UpscalingMethod::Lanczos.to_string()
                        );
                    });
                ui.end_row();
                if state.render_settings.upscaling_method == crate::renderer::UpscalingMethod::Spline {
                    ui.label("Framebuffer Channel");
                    egui::ComboBox::new("selected_channel_fb", "")
                        .selected_text(
                            state
                                .render_settings.selected_channel.to_string()
                        )
                        .show_ui(ui, |ui| {
                            for i in 0..4 {
                                ui.selectable_value(
                                    &mut state.render_settings.selected_channel,
                                    i,
                                    format!("{}", i),
                                );
                            }
                        });
                    ui.end_row();  
                }else{
                    state.render_settings.selected_channel = 0;
                }
                ui.label("Render Scale").on_hover_text("Render at reduced resolution and upscale to screen resolution");
                ui.add(egui::DragValue::new(&mut state.render_settings.render_scale)
                    .speed(0.1)
                    .range((1.)..=(16.))
                    .clamp_existing_to_range(true));
            });
    });

    egui::containers::Area::new("info icon".into())
        .anchor(egui::Align2::RIGHT_TOP, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            ui.heading("ℹ")
                .on_hover_cursor(egui::CursorIcon::Help)
                .on_hover_ui(|ui| {
                    egui::Grid::new("volume_info")
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.heading("Volume Info");
                            ui.end_row();
                            ui.label("timesteps");
                            ui.label(state.volume.volume.timesteps().to_string());
                            ui.end_row();
                            ui.label("channels");
                            ui.label(state.volume.volume.channels().to_string());
                            ui.end_row();
                            ui.label("resolution");
                            let res = state.volume.volume.size();
                            ui.label(format!("{}x{}x{} (WxHxD)", res.x, res.y, res.z));
                            ui.end_row();
                            ui.label("value range");
                            ui.label(format!(
                                "[{} , {}]",
                                state.volume.volume.min_value, state.volume.volume.max_value
                            ));
                            ui.end_row();
                        });
                });
        });
    // let mut cmap = state.cmap.clone();

    if state.colormap_editor_visible {
        egui::Window::new("Transfer Function")
            .default_size(vec2(300., 50.))
            .show(ctx, |ui| {
                egui::Grid::new("render_settings").show(ui, |ui|{
                    ui.label("Value Range");
                    let min_b = state
                        .render_settings
                        .vmin
                        .unwrap_or(state.volume.volume.min_value);
                    let max_b = state
                        .render_settings
                        .vmax
                        .unwrap_or(state.volume.volume.max_value);

                    let vmin_min = state.volume.volume.min_value.min(min_b);
                    let vmax_max = state.volume.volume.max_value.max(max_b);
                    ui.horizontal(|ui| {
                        ui.label("Min").on_hover_text("Minimum value for the colormap");
                       
                        optional_drag(
                            ui,
                            &mut state.render_settings.vmin,
                            Some(vmin_min..=max_b),
                            Some(0.01),
                            Some(vmin_min),
                        );
                    });
                    ui.end_row();
                    ui.label("");

                    ui.horizontal(|ui| {

                        ui.label("Max").on_hover_text("Maximum value for the colormap");
                        optional_drag(
                            ui,
                            &mut state.render_settings.vmax,
                            Some(min_b..=vmax_max),
                            Some(0.01),
                            Some(vmax_max),
                        );
                    });
                
                ui.end_row();
                if state.cmap_select_visible {
                    ui.label("Colormap");
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
                            let old_alpha = state.render_settings.cmap.a.clone();
                            state.render_settings.cmap = cmaps[&selected_cmap.0][&selected_cmap.1].clone();
                            if state.render_settings.cmap.a.is_none()
                                || cmaps[&selected_cmap.0][&selected_cmap.1]
                                    .has_boring_alpha_channel()
                            {
                                state.render_settings.cmap.a = old_alpha;
                            }
                            ui.ctx().data_mut(|d| {
                                d.insert_persisted("selected_cmap".into(), selected_cmap);
                            });
                        }
                        ui.ctx()
                            .data_mut(|d| d.insert_temp("cmap_search".into(), search_term));
                        if state.render_settings.cmap.a.is_none() {
                            state.render_settings.cmap.a = Some(vec![(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)]);
                        }
                        if ui.button("↔").on_hover_text("Flip colormap").clicked() {
                            state.render_settings.cmap = (&state.render_settings.cmap).reverse();
                        }
                    });
                }

            });
                let vmin = state
                    .render_settings
                    .vmin
                    .unwrap_or(state.volume.volume.min_value);
                let vmax = state
                    .render_settings
                    .vmax
                    .unwrap_or(state.volume.volume.max_value);
                show_cmap(ui, egui::Id::new("cmap preview"), &state.render_settings.cmap, vmin, vmax);

                ui.end_row();
                egui::CollapsingHeader::new("Transfer Function")
                    .default_open(true)
                    .show_unindented(ui, |ui| {
                    
                ui.horizontal_wrapped(|ui| {
                    ui.label("Presets:");
                    let v_hack = ui
                        .button("\\/")
                        .on_hover_text("double click for smooth version");
                    if v_hack.clicked() {
                        state.render_settings.cmap.a = Some(vec![(0.0, 1.0, 1.0), (0.5, 0., 0.), (1.0, 1.0, 1.0)]);
                    }
                    if v_hack.double_clicked() {
                        state.render_settings.cmap.a =
                            Some(build_segments(25, |x| ((x * 2. * PI).cos() + 1.) / 2.));
                    }
                    let slope_hack = ui
                        .button("/")
                        .on_hover_text("double click for smooth version");
                    if slope_hack.clicked() {
                        state.render_settings.cmap.a = Some(build_segments(2, |x| (-(x * PI).cos() + 1.) / 2.));
                    }
                    if slope_hack.double_clicked() {
                        state.render_settings.cmap.a = Some(build_segments(25, |x| (-(x * PI).cos() + 1.) / 2.));
                    }
                    let double_v_hack = ui
                        .button("/\\/\\")
                        .on_hover_text("double click for smooth version");
                    if double_v_hack.clicked() {
                        state.render_settings.cmap.a =
                            Some(build_segments(5, |x| (-(x * 4. * PI).cos() + 1.) / 2.));
                    }
                    if double_v_hack.double_clicked() {
                        state.render_settings.cmap.a =
                            Some(build_segments(25, |x| (-(x * 4. * PI).cos() + 1.) / 2.));
                    }
                    if ui.button("-").clicked() {
                        state.render_settings.cmap.a = Some(vec![(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)]);
                    }
                });

                ui.separator();

                if let Some(a) = &mut state.render_settings.cmap.a {
                    tf_ui(ui, a)
                    .on_hover_text("Drag anchor points to change transfer function.\nLeft-Click for new anchor point.\nRight-Click to delete anchor point.");
                }
                ui.end_row();
                if ui.button("Save Colormap").clicked(){
                    let cmap_data = serde_json::to_vec(&state.render_settings.cmap).unwrap();
                    #[cfg(target_arch = "wasm32")]
                    wasm_bindgen_futures::spawn_local(async move{
                        let file = rfd::AsyncFileDialog::new().set_file_name("colormap.json").save_file().await;
                        if let Some(file) = file{
                            file.write(&cmap_data).await.unwrap();
                        }
                    });
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let file = rfd::FileDialog::new().set_file_name("colormap.json").save_file();
                        if let Some(file) = file {
                            std::fs::write(file, cmap_data).unwrap();
                        }
                    }
                }
            });
            });
    }

    egui::Window::new("Debug").default_open(false).scroll(true).show(ctx, |ui| {
        ui.collapsing("Render Settings",|ui|{
            egui_probe::Probe::new(&mut state.render_settings)
                .show(ui);  
        });
        ui.collapsing("Camera",|ui|{
            egui_probe::Probe::new(&mut state.camera)
            .show(ui);  
        });

    });

    let frame_rect = ctx.available_rect();
    egui::Area::new(egui::Id::new("orientation"))
        .fixed_pos(Pos2::new(frame_rect.left(), frame_rect.bottom()))
        .anchor(Align2::LEFT_BOTTOM, Vec2::new(0., 0.))
        .interactable(false)
        .order(Order::Background)
        .show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                vec2(100., 100.),
                Sense::empty()
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

    if render_scale_before != state.render_settings.render_scale {
        state.frame_buffer.resize(&state.wgpu_context.device, 
            (state.config.width as f32 / state.render_settings.render_scale) as u32,
            (state.config.height as f32 / state.render_settings.render_scale) as u32
        );
    }

    let repaint = ctx.has_requested_repaint();
    return repaint;
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

fn load_or_create(ui: &egui::Ui, cmap: &ColorMap, n: u32) -> egui::TextureId
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

// stores colormap texture in egui context
// only updates texture if it changed
fn cmap_preview(ui: &egui::Ui, id: Id, cmap: &ColorMap, n: u32) -> egui::TextureId {
    let tex: Option<(Id, egui::TextureHandle)> = ui.ctx().data_mut(|d| d.get_temp(id));
    match tex {
        Some((old_id, mut tex)) => {
            if old_id != id.with(&cmap) {
                tex.set(
                    egui::ColorImage::from_rgba_unmultiplied(
                        [n as usize, 1],
                        bytemuck::cast_slice(&cmap.rasterize(n as usize)),
                    ),
                    egui::TextureOptions::LINEAR,
                );
            }
            tex.id()
        }
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
            ui.ctx()
                .data_mut(|d| d.insert_temp(id, (id.with(cmap), tex)));
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
        drag = drag.range(range);
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

fn show_cmap(ui: &mut egui::Ui, id: egui::Id, cmap: &ColorMap, vmin: f32, vmax: f32) {
    let texture = cmap_preview(ui, id, cmap, COLORMAP_RESOLUTION);
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
        .height(60.)
        .show_background(false)
        .show_grid(false)
        .include_x(vmin)
        .include_x(vmax)
        .custom_y_axes(vec![])
        .allow_boxed_zoom(false)
        .allow_double_click_reset(false)
        .allow_drag(false)
        .allow_scroll(false)
        .allow_zoom(false).set_margin_fraction(egui::Vec2::ZERO);
    plot.show(ui, |plot_ui| {
        plot_ui.image(image);
    });
}

/// builds transfer function segments from a list of values
/// out values are (x,y0,y1) where y0 are the values before x and y1 are the values after x
fn build_segments<F: Fn(f32) -> f32>(n: usize, f: F) -> Vec<(f32, f32, f32)> {
    (0..n)
        .map(|i| {
            let x = i as f32 / (n as f32 - 1.);
            let v = f(x);
            (x, v, v)
        })
        .collect()
}
