use std::path::PathBuf;

use anyhow::{Context, bail};

/// Command line options. Every scale value still comes from the config files; `--voxel-size`
/// overrides `voxel_size_m` for one run and is validated against the same range.
#[derive(Debug, Clone)]
pub struct Args {
    pub assets_dir: PathBuf,
    pub voxel_size_m: Option<f64>,
    pub vsync: bool,
    /// Window size in physical pixels.
    pub width: u32,
    pub height: u32,
    /// Measure frame times for this many seconds after warm up, print a summary and exit.
    pub bench_seconds: Option<f64>,
    /// Save a screenshot here after warm up (after the benchmark when both are given) and exit.
    pub screenshot: Option<PathBuf>,
}

pub const USAGE: &str = "\
vx_game: Voxel Engine M0

USAGE: vx_game [OPTIONS]

OPTIONS:
    --assets <dir>          asset directory (default: $VX_ASSETS or the repo's assets/)
    --voxel-size <m>        override voxel_size_m from assets/config/world.ron for this run
    --no-vsync              present without waiting for vertical blank (uncapped frame rate)
    --size <WxH>            window size in physical pixels (default 1920x1080)
    --bench-seconds <s>     measure frame times after warm up, print a BENCH line, exit
    --screenshot <path>     save a PNG after warm up (and after any benchmark), exit
    -h, --help              print this help

CONTROLS: hold the right mouse button to look, WASD to move, Space or E up, Shift or Q down,
scroll to change speed, F12 screenshot, Esc quit.";

impl Args {
    /// A benchmark or capture run, which exits by itself and ignores input.
    pub fn automated(&self) -> bool {
        self.bench_seconds.is_some() || self.screenshot.is_some()
    }

    pub fn parse(mut it: impl Iterator<Item = String>) -> anyhow::Result<Option<Self>> {
        let mut args = Self {
            assets_dir: std::env::var_os("VX_ASSETS").map_or_else(
                || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"),
                PathBuf::from,
            ),
            voxel_size_m: None,
            vsync: true,
            width: 1920,
            height: 1080,
            bench_seconds: None,
            screenshot: None,
        };
        while let Some(flag) = it.next() {
            let mut value = || it.next().with_context(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "-h" | "--help" => return Ok(None),
                "--assets" => args.assets_dir = PathBuf::from(value()?),
                "--voxel-size" => args.voxel_size_m = Some(value()?.parse()?),
                "--no-vsync" => args.vsync = false,
                "--size" => {
                    let v = value()?;
                    let (w, h) = v.split_once('x').context("--size expects WxH")?;
                    args.width = w.parse()?;
                    args.height = h.parse()?;
                }
                "--bench-seconds" => args.bench_seconds = Some(value()?.parse()?),
                "--screenshot" => args.screenshot = Some(PathBuf::from(value()?)),
                other => bail!("unknown option {other}\n\n{USAGE}"),
            }
        }
        Ok(Some(args))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> anyhow::Result<Option<Args>> {
        Args::parse(s.split_whitespace().map(String::from))
    }

    #[test]
    fn parses_measurement_flags() {
        let a = parse(
            "--voxel-size 0.5 --no-vsync --size 1280x720 --bench-seconds 10 --screenshot a.png",
        )
        .unwrap()
        .unwrap();
        assert_eq!(a.voxel_size_m, Some(0.5));
        assert!(!a.vsync);
        assert_eq!((a.width, a.height), (1280, 720));
        assert_eq!(a.bench_seconds, Some(10.0));
        assert_eq!(a.screenshot, Some(PathBuf::from("a.png")));
    }

    #[test]
    fn rejects_unknown_and_incomplete_flags() {
        assert!(parse("--fast").is_err());
        assert!(parse("--voxel-size").is_err());
        assert!(parse("--size 100").is_err());
        assert!(parse("--help").unwrap().is_none());
    }
}
