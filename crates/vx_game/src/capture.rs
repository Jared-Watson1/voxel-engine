use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::time::{SystemTime, UNIX_EPOCH};

use vx_render::CapturedImage;

pub type CaptureOutcome = Result<PathBuf, String>;

/// Where F12 screenshots go when no path was given.
pub fn default_path() -> PathBuf {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    PathBuf::from("target/screenshots").join(format!("vx-{secs}.png"))
}

/// Encodes and writes the PNG on its own thread (D9) and reports the outcome on `tx`.
pub fn spawn_write(image: CapturedImage, path: PathBuf, tx: Sender<CaptureOutcome>) {
    let spawned = std::thread::Builder::new()
        .name("vx-capture".into())
        .spawn(move || {
            let outcome = write_png(&image, &path).map(|()| path);
            // A closed receiver means the app is shutting down.
            let _ = tx.send(outcome);
        });
    if let Err(e) = spawned {
        tracing::error!("could not spawn the capture writer: {e}");
    }
}

fn write_png(image: &CapturedImage, path: &Path) -> Result<(), String> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    }
    let file = File::create(path).map_err(|e| format!("creating {}: {e}", path.display()))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), image.width, image.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
    writer
        .write_image_data(&image.rgba)
        .map_err(|e| e.to_string())?;
    writer.finish().map_err(|e| e.to_string())
}
