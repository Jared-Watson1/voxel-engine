//! Frame profiler (puffin through the `profiling` crate), enabled by the `profile` feature.
//!
//! puffin is used rather than Tracy because tracy-client compiles the Tracy C++ client, and the
//! repo asks before adding any C or C++ build. View captures with `puffin_viewer` connected to
//! `127.0.0.1:8585`.

#[cfg(feature = "profile")]
mod enabled {
    use std::collections::BTreeSet;
    use std::sync::{Mutex, OnceLock};

    /// Names of every scope puffin has recorded, gathered by a frame sink so the overlay and the
    /// log can show that the spans exist without an external viewer.
    static SCOPES: OnceLock<Mutex<BTreeSet<String>>> = OnceLock::new();

    /// Spans the engine itself opens; wgpu and egui add their own through the same backend.
    const ENGINE_SCOPES: [&str; 10] = [
        "frame",
        "update",
        "render",
        "World::update",
        "World::recentre",
        "World::poll_results",
        "World::upload",
        "World::apply_brush",
        "TerrainSampler::fill_chunk",
        "mesh_chunk",
    ];

    pub struct Profiler {
        _server: puffin_http::Server,
    }

    pub fn start() -> anyhow::Result<Profiler> {
        puffin::set_scopes_on(true);
        let address = format!("127.0.0.1:{}", puffin_http::DEFAULT_PORT);
        let server = puffin_http::Server::new(&address)?;
        let scopes = SCOPES.get_or_init(Default::default);
        puffin::GlobalProfiler::lock().add_sink(Box::new(move |frame| {
            if frame.scope_delta.is_empty() {
                return;
            }
            if let Ok(mut names) = scopes.lock() {
                let new: Vec<String> = frame
                    .scope_delta
                    .iter()
                    .map(|d| d.name().to_string())
                    .filter(|n| ENGINE_SCOPES.contains(&n.as_str()) && !names.contains(n))
                    .collect();
                if !new.is_empty() {
                    names.extend(new);
                    tracing::info!(
                        "puffin engine spans recorded: {}",
                        names.iter().cloned().collect::<Vec<_>>().join(", ")
                    );
                }
            }
        }));
        tracing::info!("puffin server on {address}; connect with puffin_viewer");
        Ok(Profiler { _server: server })
    }

    pub fn status() -> Option<String> {
        let names = SCOPES.get()?.lock().ok()?;
        Some(format!(
            "puffin :{}, {}/{} engine spans",
            puffin_http::DEFAULT_PORT,
            names.len(),
            ENGINE_SCOPES.len()
        ))
    }
}

#[cfg(feature = "profile")]
pub use enabled::{start, status};

#[cfg(not(feature = "profile"))]
pub fn status() -> Option<String> {
    None
}
