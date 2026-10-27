use glam::DVec3;
use vx_core::BudgetUse;
use vx_render::{CullStats, PoolStats};

use crate::app::FrameSections;
use crate::world::WorldStats;

/// Values shown in the debug overlay, gathered once per frame.
pub struct OverlayData {
    pub frame_ms: f64,
    pub cpu_ms: f64,
    pub cpu_ms_max: f64,
    pub sections: FrameSections,
    pub camera_m: DVec3,
    pub speed_m_s: f64,
    pub voxel_size_m: Option<f64>,
    pub view_radius_chunks: Option<u32>,
    pub seed: Option<u64>,
    pub workers: usize,
    pub world: Option<WorldStats>,
    pub pool: Option<PoolStats>,
    pub cull: Option<CullStats>,
    pub budgets: Vec<BudgetUse>,
    pub brush: String,
    pub flight_progress: Option<f64>,
    pub adapter: String,
    pub present_mode: String,
    pub resolution: (u32, u32),
    pub shader_error: Option<String>,
    pub world_error: Option<String>,
    pub last_capture: Option<String>,
    pub profiler: Option<String>,
}

pub fn draw(ctx: &egui::Context, d: &OverlayData) {
    egui::Window::new("Voxel Engine M1")
        .default_pos([12.0, 12.0])
        .resizable(false)
        .show(ctx, |ui| {
            egui::Grid::new("frame").num_columns(2).show(ui, |ui| {
                let fps = if d.frame_ms > 0.0 {
                    1.0e3 / d.frame_ms
                } else {
                    0.0
                };
                row(ui, "frame", format!("{:.2} ms  ({fps:.0} fps)", d.frame_ms));
                row(
                    ui,
                    "cpu work",
                    format!("{:.2} ms mean, {:.2} ms max", d.cpu_ms, d.cpu_ms_max),
                );
                row(
                    ui,
                    "sections",
                    format!(
                        "polls {:.2}, update {:.2}, egui {:.2}, draw list {:.2}, render {:.2} ms",
                        d.sections.polls_ms,
                        d.sections.update_ms,
                        d.sections.egui_ms,
                        d.sections.draw_list_ms,
                        d.sections.render_ms
                    ),
                );
                row(
                    ui,
                    "camera",
                    format!(
                        "{:.1}, {:.1}, {:.1} m",
                        d.camera_m.x, d.camera_m.y, d.camera_m.z
                    ),
                );
                row(ui, "speed", format!("{:.1} m/s", d.speed_m_s));
                row(
                    ui,
                    "world",
                    match (d.voxel_size_m, d.view_radius_chunks, d.seed) {
                        (Some(v), Some(r), Some(s)) => {
                            format!("voxel {v} m, radius {r} chunks, seed {s}")
                        }
                        _ => "loading".into(),
                    },
                );
                row(ui, "brush", d.brush.clone());
                if let Some(p) = d.flight_progress {
                    row(ui, "flight", format!("{:.0} %", p * 100.0));
                }
                row(ui, "gpu", d.adapter.clone());
                row(
                    ui,
                    "surface",
                    format!("{}x{} {}", d.resolution.0, d.resolution.1, d.present_mode),
                );
                if let Some(p) = &d.profiler {
                    row(ui, "profiler", p.clone());
                }
                if let Some(c) = &d.last_capture {
                    row(ui, "screenshot", c.clone());
                }
            });

            if let Some(w) = &d.world {
                ui.separator();
                ui.strong(format!("pipeline, {} workers", d.workers));
                egui::Grid::new("pipeline")
                    .num_columns(6)
                    .striped(true)
                    .show(ui, |ui| {
                        for h in ["stage", "queued", "running", "done", "last", "mean / max"] {
                            ui.weak(h);
                        }
                        ui.end_row();
                        for s in &w.stages {
                            ui.label(s.name);
                            ui.monospace(s.queued.to_string());
                            ui.monospace(s.running.to_string());
                            ui.monospace(format!("{} ({} cancelled)", s.completed, s.cancelled));
                            ui.monospace(format!("{:.2} ms", s.last_ms));
                            ui.monospace(format!("{:.2} / {:.2} ms", s.mean_ms, s.max_ms));
                            ui.end_row();
                        }
                        ui.label("upload");
                        ui.monospace(w.upload_queued.to_string());
                        ui.monospace("main");
                        ui.monospace(format!(
                            "{} ({} failed)",
                            w.uploaded_total, w.upload_failures
                        ));
                        ui.monospace(format!("{:.2} ms", w.upload_ms_frame));
                        ui.monospace(format!(
                            "{:.2} MB/frame, peak {:.2}",
                            w.upload_bytes_frame as f64 / 1.0e6,
                            w.upload_bytes_peak as f64 / 1.0e6
                        ));
                        ui.end_row();
                    });
                egui::Grid::new("streaming").num_columns(2).show(ui, |ui| {
                    row(
                        ui,
                        "chunks",
                        format!(
                            "{} resident, {} meshed, {} dense, {} pending",
                            w.resident, w.meshed, w.dense, w.pending
                        ),
                    );
                    row(
                        ui,
                        "voxel memory",
                        format!(
                            "{:.1} MB, peak {:.1} MB",
                            w.voxel_bytes as f64 / 1.0e6,
                            w.voxel_bytes_peak as f64 / 1.0e6
                        ),
                    );
                    row(
                        ui,
                        "world update",
                        format!(
                            "{:.2} ms, max {:.2} ms; draw list {:.2} ms",
                            w.update_ms, w.update_ms_max, w.draw_list_ms
                        ),
                    );
                    row(
                        ui,
                        "recentre",
                        format!(
                            "{:.2} ms last, {:.2} ms max",
                            w.stream_ms_last, w.stream_ms_max
                        ),
                    );
                    row(ui, "edits", w.edits.to_string());
                });
            }

            if let Some(p) = &d.pool {
                ui.separator();
                egui::Grid::new("gpu").num_columns(2).show(ui, |ui| {
                    row(
                        ui,
                        "vertex pool",
                        format!(
                            "{:.1} / {:.0} MB, largest free {:.1} MB",
                            p.vertex_bytes_used as f64 / 1.0e6,
                            p.vertex_bytes_capacity as f64 / 1.0e6,
                            p.largest_free_vertex_bytes as f64 / 1.0e6
                        ),
                    );
                    row(
                        ui,
                        "index pool",
                        format!(
                            "{:.1} / {:.0} MB, largest free {:.1} MB",
                            p.index_bytes_used as f64 / 1.0e6,
                            p.index_bytes_capacity as f64 / 1.0e6,
                            p.largest_free_index_bytes as f64 / 1.0e6
                        ),
                    );
                    row(
                        ui,
                        "gpu meshes",
                        format!(
                            "{}, textures {:.1} MB",
                            p.meshes,
                            p.texture_bytes as f64 / 1.0e6
                        ),
                    );
                    match d.cull {
                        Some(c) => row(
                            ui,
                            "culling",
                            format!(
                                "{} chunks, {} triangles ({} frames old)",
                                c.visible_chunks, c.triangles, c.frames_old
                            ),
                        ),
                        None => row(ui, "culling", "pending".into()),
                    }
                });
            }

            if !d.budgets.is_empty() {
                ui.separator();
                ui.strong("budgets");
                egui::Grid::new("budgets").num_columns(3).show(ui, |ui| {
                    for b in &d.budgets {
                        ui.label(b.name);
                        let text = format!("{:.2} / {:.1} {}", b.used, b.budget, b.unit);
                        let colour = if b.exceeded() {
                            egui::Color32::LIGHT_RED
                        } else if b.fraction() > 0.8 {
                            egui::Color32::YELLOW
                        } else {
                            egui::Color32::LIGHT_GREEN
                        };
                        ui.colored_label(colour, text);
                        ui.monospace(format!("{:.0} %", b.fraction() * 100.0));
                        ui.end_row();
                    }
                });
            }

            for error in [&d.shader_error, &d.world_error].into_iter().flatten() {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            }
            ui.weak("RMB look, WASD move, LMB brush, B dig/place, [ ] radius, F12 screenshot");
        });
}

fn row(ui: &mut egui::Ui, label: &str, value: String) {
    ui.label(label);
    ui.monospace(value);
    ui.end_row();
}
