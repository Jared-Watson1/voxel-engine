use std::path::PathBuf;

use anyhow::{Context, bail};

/// Command line options. Every scale value still comes from the config files; the overrides
/// below replace single fields for one run and are validated against the same ranges.
#[derive(Debug, Clone)]
pub struct Args {
    pub assets_dir: PathBuf,
    pub voxel_size_m: Option<f64>,
    pub lod_radius_chunks: Option<f64>,
    pub lod_levels: Option<u32>,
    /// Planet radius override for this run (the 50 km parameter run of D11).
    pub radius_km: Option<f64>,
    pub seed: Option<u64>,
    pub workers: Option<usize>,
    pub vsync: bool,
    /// Window size in physical pixels.
    pub width: u32,
    pub height: u32,
    /// Measure frame times for this many seconds after warm up, print a summary and exit. With a
    /// flight the measurement lasts the whole flight unless this is shorter.
    pub bench_seconds: Option<f64>,
    /// Fly a square loop of this side length during the measurement.
    pub flight_km: Option<f64>,
    pub flight_speed_m_s: f64,
    /// Fly the M2 descent: stand on the surface, climb to `descent_top_km`, come back down.
    pub descent: bool,
    pub descent_top_km: f64,
    /// Seconds for each of the climb and the way down.
    pub descent_climb_s: f64,
    /// Local solar time at the start position as a fraction of a day (0.25 sunrise, 0.5 noon).
    pub time_of_day: Option<f64>,
    /// Simulation seconds per real second at startup.
    pub warp: Option<f64>,
    /// Exposure in stops.
    pub exposure_ev: f32,
    /// Start this high above the terrain, looking down at the planet, instead of standing.
    pub altitude_km: Option<f64>,
    /// Start position in degrees (longitude east positive from the prime meridian).
    pub latitude_deg: Option<f64>,
    pub longitude_deg: Option<f64>,
    /// Apply a scripted dig or place every few frames during the measurement.
    pub edit_test: bool,
    /// Brush radius in voxels at startup.
    pub brush_radius_voxels: Option<f64>,
    /// Hide the debug overlay (for screenshots of the terrain alone).
    pub no_overlay: bool,
    /// Save a screenshot here after warm up (after the benchmark when both are given) and exit.
    pub screenshot: Option<PathBuf>,
}

/// Exposure at startup in stops: half a stop over the physical sun keeps the bright skies and
/// greens of the art direction.
pub const DEFAULT_EXPOSURE_EV: f32 = 0.5;

pub const USAGE: &str = "\
vx_game: Voxel Engine M2

USAGE: vx_game [OPTIONS]

OPTIONS:
    --assets <dir>          asset directory (default: $VX_ASSETS or the repo's assets/)
    --voxel-size <m>        override voxel_size_m from assets/config/world.ron for this run
    --lod-radius <chunks>   override lod_radius_chunks for this run
    --lod-levels <n>        override lod_levels for this run
    --radius-km <km>        override the planet radius from assets/config/system.ron
    --seed <n>              override the world seed for this run
    --workers <n>           worker threads for the job system (default: cores minus one)
    --no-vsync              present without waiting for vertical blank (uncapped frame rate)
    --size <WxH>            window size in physical pixels (default 1920x1080)
    --bench-seconds <s>     measure frame times after warm up, print a BENCH line, exit
    --flight-km <km>        fly a square loop of this side during the measurement
    --flight-speed <m/s>    flight speed (default 100)
    --descent               stand on the surface, climb to orbit and come back (the M2 exit run)
    --descent-top <km>      altitude of the descent's top (default 100)
    --descent-climb <s>     seconds for the climb and for the way down (default 30)
    --time-of-day <f>       local solar time at the start, fraction of a day (0.25 sunrise)
    --warp <x>              simulation seconds per real second at startup (default 1)
    --exposure <ev>         exposure in stops (default 0.5)
    --altitude <km>         start this high above the terrain looking down at the planet
    --latitude <deg>        start latitude (default 25)
    --longitude <deg>       start longitude, east positive (default 0)
    --edit-test             apply scripted brushes during the measurement
    --brush-radius <voxels> brush radius at startup (default 3)
    --no-overlay            hide the debug overlay
    --screenshot <path>     save a PNG after warm up (and after any benchmark), exit
    -h, --help              print this help

CONTROLS: hold the right mouse button to look, WASD to move, Space or E up, Shift or Q down
(up is away from the planet, speed follows altitude), scroll to scale the speed, left mouse
button to apply the brush, B to switch dig and place, [ and ] to change the brush radius,
, and . to slow or speed up time (the slowest step pauses), - and = to change the exposure,
F12 screenshot, Esc quit.";

impl Args {
    /// A benchmark, flight or capture run, which exits by itself and ignores input.
    pub fn automated(&self) -> bool {
        self.bench_seconds.is_some()
            || self.screenshot.is_some()
            || self.flight_km.is_some()
            || self.descent
    }

    pub fn parse(mut it: impl Iterator<Item = String>) -> anyhow::Result<Option<Self>> {
        let mut args = Self {
            assets_dir: std::env::var_os("VX_ASSETS").map_or_else(
                || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets"),
                PathBuf::from,
            ),
            voxel_size_m: None,
            lod_radius_chunks: None,
            lod_levels: None,
            radius_km: None,
            seed: None,
            workers: None,
            vsync: true,
            width: 1920,
            height: 1080,
            bench_seconds: None,
            flight_km: None,
            flight_speed_m_s: 100.0,
            descent: false,
            descent_top_km: 100.0,
            descent_climb_s: 30.0,
            time_of_day: None,
            warp: None,
            exposure_ev: DEFAULT_EXPOSURE_EV,
            altitude_km: None,
            latitude_deg: None,
            longitude_deg: None,
            edit_test: false,
            brush_radius_voxels: None,
            no_overlay: false,
            screenshot: None,
        };
        while let Some(flag) = it.next() {
            let mut value = || it.next().with_context(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "-h" | "--help" => return Ok(None),
                "--assets" => args.assets_dir = PathBuf::from(value()?),
                "--voxel-size" => args.voxel_size_m = Some(value()?.parse()?),
                "--lod-radius" => args.lod_radius_chunks = Some(value()?.parse()?),
                "--lod-levels" => args.lod_levels = Some(value()?.parse()?),
                "--radius-km" => args.radius_km = Some(value()?.parse()?),
                "--descent" => args.descent = true,
                "--descent-top" => args.descent_top_km = value()?.parse()?,
                "--descent-climb" => args.descent_climb_s = value()?.parse()?,
                "--time-of-day" => args.time_of_day = Some(value()?.parse()?),
                "--warp" => args.warp = Some(value()?.parse()?),
                "--exposure" => args.exposure_ev = value()?.parse()?,
                "--altitude" => args.altitude_km = Some(value()?.parse()?),
                "--latitude" => args.latitude_deg = Some(value()?.parse()?),
                "--longitude" => args.longitude_deg = Some(value()?.parse()?),
                "--seed" => args.seed = Some(value()?.parse()?),
                "--workers" => args.workers = Some(value()?.parse()?),
                "--no-vsync" => args.vsync = false,
                "--size" => {
                    let v = value()?;
                    let (w, h) = v.split_once('x').context("--size expects WxH")?;
                    args.width = w.parse()?;
                    args.height = h.parse()?;
                }
                "--bench-seconds" => args.bench_seconds = Some(value()?.parse()?),
                "--flight-km" => args.flight_km = Some(value()?.parse()?),
                "--flight-speed" => args.flight_speed_m_s = value()?.parse()?,
                "--edit-test" => args.edit_test = true,
                "--brush-radius" => args.brush_radius_voxels = Some(value()?.parse()?),
                "--no-overlay" => args.no_overlay = true,
                "--screenshot" => args.screenshot = Some(PathBuf::from(value()?)),
                other => bail!("unknown option {other}\n\n{USAGE}"),
            }
        }
        if args.flight_speed_m_s <= 0.0 || args.flight_km.is_some_and(|k| k <= 0.0) {
            bail!("flight distance and speed must be positive");
        }
        if args.descent_top_km <= 0.0 || args.descent_climb_s <= 0.0 {
            bail!("descent top and climb time must be positive");
        }
        if args.warp.is_some_and(|w| !(w >= 0.0 && w.is_finite())) {
            bail!("warp must be zero or positive");
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
            "--voxel-size 0.5 --no-vsync --size 1280x720 --bench-seconds 10 --screenshot a.png \
             --flight-km 2 --flight-speed 50 --edit-test --lod-radius 6 --seed 7 --workers 3 \
             --brush-radius 6 --no-overlay --radius-km 50 --lod-levels 10 --descent \
             --descent-top 80 --descent-climb 20 --time-of-day 0.25 --warp 60 --exposure -1",
        )
        .unwrap()
        .unwrap();
        assert_eq!(a.voxel_size_m, Some(0.5));
        assert!(!a.vsync);
        assert_eq!((a.width, a.height), (1280, 720));
        assert_eq!(a.bench_seconds, Some(10.0));
        assert_eq!(a.screenshot, Some(PathBuf::from("a.png")));
        assert_eq!(a.flight_km, Some(2.0));
        assert_eq!(a.flight_speed_m_s, 50.0);
        assert!(a.edit_test);
        assert_eq!(a.lod_radius_chunks, Some(6.0));
        assert_eq!(a.lod_levels, Some(10));
        assert_eq!(a.radius_km, Some(50.0));
        assert!(a.descent);
        assert_eq!((a.descent_top_km, a.descent_climb_s), (80.0, 20.0));
        assert_eq!(a.time_of_day, Some(0.25));
        assert_eq!(a.warp, Some(60.0));
        assert_eq!(a.exposure_ev, -1.0);
        assert_eq!(a.seed, Some(7));
        assert_eq!(a.workers, Some(3));
        assert_eq!(a.brush_radius_voxels, Some(6.0));
        assert!(a.no_overlay);
        assert!(a.automated());
    }

    #[test]
    fn rejects_unknown_and_incomplete_flags() {
        assert!(parse("--fast").is_err());
        assert!(parse("--voxel-size").is_err());
        assert!(parse("--size 100").is_err());
        assert!(parse("--flight-km 0").is_err());
        assert!(parse("--help").unwrap().is_none());
        assert!(!parse("").unwrap().unwrap().automated());
    }
}
