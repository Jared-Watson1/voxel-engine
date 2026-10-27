use glam::DVec3;

/// Values shown in the debug overlay, gathered once per frame.
pub struct OverlayData {
    pub frame_ms: f64,
    pub cpu_ms: f64,
    pub triangles: Option<usize>,
    pub vertices: Option<usize>,
    pub camera_m: DVec3,
    pub speed_m_s: f64,
    pub voxel_size_m: Option<f64>,
    pub chunk_extent_m: Option<f64>,
    pub generate_ms: Option<f64>,
    pub mesh_ms: Option<f64>,
    pub adapter: String,
    pub present_mode: String,
    pub resolution: (u32, u32),
    pub shader_error: Option<String>,
    pub chunk_error: Option<String>,
    pub last_capture: Option<String>,
    pub profiler: Option<String>,
}

pub fn draw(ctx: &egui::Context, d: &OverlayData) {
    egui::Window::new("Voxel Engine M0")
        .default_pos([12.0, 12.0])
        .resizable(false)
        .show(ctx, |ui| {
            egui::Grid::new("stats").num_columns(2).show(ui, |ui| {
                let fps = if d.frame_ms > 0.0 {
                    1.0e3 / d.frame_ms
                } else {
                    0.0
                };
                row(ui, "frame", format!("{:.2} ms  ({fps:.0} fps)", d.frame_ms));
                row(ui, "cpu work", format!("{:.2} ms", d.cpu_ms));
                row(ui, "triangles", opt(d.triangles, |t| t.to_string()));
                row(ui, "vertices", opt(d.vertices, |v| v.to_string()));
                row(
                    ui,
                    "camera",
                    format!(
                        "{:.2}, {:.2}, {:.2} m",
                        d.camera_m.x, d.camera_m.y, d.camera_m.z
                    ),
                );
                row(ui, "speed", format!("{:.1} m/s", d.speed_m_s));
                row(ui, "voxel size", opt(d.voxel_size_m, |v| format!("{v} m")));
                row(
                    ui,
                    "chunk",
                    opt(d.chunk_extent_m, |e| format!("32^3, {e} m")),
                );
                row(
                    ui,
                    "generate, mesh",
                    match (d.generate_ms, d.mesh_ms) {
                        (Some(g), Some(m)) => format!("{g:.2} ms, {m:.2} ms"),
                        _ => "pending".into(),
                    },
                );
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
            for error in [&d.shader_error, &d.chunk_error].into_iter().flatten() {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            }
            ui.weak("RMB look, WASD move, Space/E up, Shift/Q down, scroll speed, F12 screenshot");
        });
}

fn row(ui: &mut egui::Ui, label: &str, value: String) {
    ui.label(label);
    ui.monospace(value);
    ui.end_row();
}

fn opt<T: Copy>(v: Option<T>, f: impl Fn(T) -> String) -> String {
    v.map_or_else(|| "pending".into(), f)
}
