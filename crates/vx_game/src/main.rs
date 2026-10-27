//! Voxel Engine: the binary. Window, input, the app loop, camera and debug overlay.

mod app;
mod args;
mod camera;
mod capture;
mod overlay;
mod profiler;
mod scene;
mod stats;

use tracing_subscriber::EnvFilter;
use winit::event_loop::{ControlFlow, EventLoop};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,wgpu_core=warn,wgpu_hal=warn,naga=warn")),
        )
        .init();

    let Some(args) = args::Args::parse(std::env::args().skip(1))? else {
        println!("{}", args::USAGE);
        return Ok(());
    };

    #[cfg(feature = "profile")]
    let _profiler = profiler::start()?;

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = app::App::new(args);
    event_loop.run_app(&mut app)?;
    match app.exit_code() {
        0 => Ok(()),
        code => std::process::exit(code),
    }
}
