//! Headless tools: `bench-chunk` measures generate and mesh of single planet chunks on one thread
//! (engineering rule 5), so regressions are numbers.

use std::time::Instant;

use glam::DVec3;
use vx_core::{Budgets, CHUNK_SIZE, PlanetParams, WorldConfig, WorldPos};
use vx_voxel::{Chunk, ChunkKey, ChunkStore, Face, mesh_chunk, transition_slab};
use vx_worldgen::PlanetSampler;

const USAGE: &str = "\
vx_tools: Voxel Engine headless tools

USAGE: vx_tools bench-chunk [--radius-km <km>] [--voxel-size <m>] [--seed <n>] [--level <n>]
                            [--runs <n>] [--chunks <n>]

Generates and meshes surface chunks of the default planet on one thread and prints, per level
(0, 3 and 6 unless --level picks one) and per height cache state (cold: the cache is cleared before
each chunk; warm: the same chunk again), the median and p95 of generate (fill_chunk), the six
transition slabs (transition_slab and fill_block, none at level 0), regular meshing (mesh_chunk)
and their total, against the 4 ms budget. Also times the classification of all air and all rock
chunks and the compression of a surface chunk.";

struct Options {
    radius_km: Option<f64>,
    voxel_size_m: Option<f64>,
    seed: Option<u64>,
    level: Option<u8>,
    runs: usize,
    chunks: usize,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        println!("{USAGE}");
        return;
    };
    if command != "bench-chunk" {
        eprintln!("unknown command {command}\n\n{USAGE}");
        std::process::exit(2);
    }
    let mut options = Options {
        radius_km: None,
        voxel_size_m: None,
        seed: None,
        level: None,
        runs: 20,
        chunks: 8,
    };
    while let Some(flag) = args.next() {
        let value = args.next();
        let parsed: Result<(), String> = match (flag.as_str(), value) {
            ("--radius-km", Some(v)) => v
                .parse()
                .map(|v| options.radius_km = Some(v))
                .map_err(|e| e.to_string()),
            ("--voxel-size", Some(v)) => v
                .parse()
                .map(|v| options.voxel_size_m = Some(v))
                .map_err(|e| e.to_string()),
            ("--seed", Some(v)) => v
                .parse()
                .map(|v| options.seed = Some(v))
                .map_err(|e| e.to_string()),
            ("--level", Some(v)) => v
                .parse()
                .map(|v| options.level = Some(v))
                .map_err(|e| e.to_string()),
            ("--runs", Some(v)) => v
                .parse()
                .map(|v| options.runs = v)
                .map_err(|e| e.to_string()),
            ("--chunks", Some(v)) => v
                .parse()
                .map(|v| options.chunks = v)
                .map_err(|e| e.to_string()),
            _ => {
                eprintln!("bad option {flag}\n\n{USAGE}");
                std::process::exit(2);
            }
        };
        if let Err(e) = parsed {
            eprintln!("bad value for {flag}: {e}");
            std::process::exit(2);
        }
    }
    if let Err(e) = bench_chunk(&options) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

/// Timings of one chunk in milliseconds.
#[derive(Default)]
struct Samples {
    generate: Vec<f64>,
    slabs: Vec<f64>,
    mesh: Vec<f64>,
    total: Vec<f64>,
    triangles: Vec<usize>,
    /// Height grid nodes computed for the chunk and its slabs (cache misses times tile nodes): a
    /// load independent measure of the cold cost.
    nodes: Vec<u64>,
}

fn bench_chunk(options: &Options) -> Result<(), String> {
    let mut config = WorldConfig::default();
    if let Some(v) = options.voxel_size_m {
        config.voxel_size_m = v;
    }
    if let Some(s) = options.seed {
        config.seed = s;
    }
    config.validate().map_err(|e| e.to_string())?;
    let mut planet = PlanetParams::default();
    if let Some(r) = options.radius_km {
        planet.radius_km = r;
    }
    planet.validate().map_err(|e| e.to_string())?;
    let sampler = PlanetSampler::new(config.seed, &planet, config.voxel_size_m);
    let load = std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    println!("machine: {load}");
    let levels: Vec<u8> = options.level.map_or(vec![0, 3, 6], |l| vec![l]);
    let runs = options.runs.max(1);

    println!(
        "bench-chunk: radius_km={} voxel_size_m={} seed={} runs_each={} one thread; shell {:.1} to {:.1} m",
        planet.radius_km,
        config.voxel_size_m,
        config.seed,
        runs,
        sampler.shell_m().0,
        sampler.shell_m().1
    );
    for &level in &levels {
        let keys = surface_keys(&sampler, &config, level, options.chunks.max(1));
        let extent = ChunkKey::at_level(level, 0, 0, 0).extent_m(config.voxel_size_m);
        println!(
            "level {level}: voxel {} m, chunk extent {extent} m, {} surface chunks: {keys:?}",
            ChunkKey::at_level(level, 0, 0, 0).voxel_size_m(config.voxel_size_m),
            keys.len()
        );
        let mut cold = Samples::default();
        let mut warm = Samples::default();
        for key in &keys {
            for _ in 0..runs {
                sampler.clear_height_cache();
                measure(&sampler, &config, *key, &mut cold);
                measure(&sampler, &config, *key, &mut warm);
            }
        }
        for (label, samples) in [("cold", &mut cold), ("warm", &mut warm)] {
            report(&planet, &config, level, label, samples);
        }
    }

    let stats = sampler.height_cache_stats();
    println!(
        "height cache: {} of {} tiles resident ({} KiB), {} hits, {} misses since the last clear",
        stats.resident_tiles,
        stats.capacity_tiles,
        stats.resident_bytes / 1024,
        stats.hits,
        stats.misses
    );

    // Uniform chunks: classification decides them without noise.
    let (inner, outer) = sampler.shell_m();
    let air_key = ChunkKey::containing(WorldPos::body_fixed(DVec3::Y * (outer + 1000.0)), &config);
    let rock_key = ChunkKey::containing(
        WorldPos::body_fixed(DVec3::Y * (inner - 1000.0).max(0.0)),
        &config,
    );
    let air = time_fill(&sampler, &config, air_key, runs);
    let rock = time_fill(&sampler, &config, rock_key, runs);
    let key = surface_keys(&sampler, &config, 0, 1)[0];
    let mut chunk = Chunk::for_key(key, &config);
    sampler.fill_chunk(&mut chunk);
    let mut compress = Vec::new();
    let mut bytes = 0;
    for _ in 0..runs {
        let t0 = Instant::now();
        let store = ChunkStore::compress(&chunk);
        compress.push(ms(t0.elapsed()));
        bytes = store.resident_bytes();
    }
    println!(
        "  air chunk generate median {air:.4} ms; rock chunk generate median {rock:.4} ms; \
         compress median {:.3} ms ({bytes} bytes of {} raw)",
        percentile(&mut compress, 0.5),
        (CHUNK_SIZE + 3).pow(3) * 2
    );
    Ok(())
}

fn measure(sampler: &PlanetSampler, config: &WorldConfig, key: ChunkKey, out: &mut Samples) {
    let mut chunk = Chunk::for_key(key, config);
    let misses = sampler.height_cache_stats().misses;
    let t0 = Instant::now();
    sampler.fill_chunk(&mut chunk);
    let t1 = Instant::now();
    for face in Face::ALL {
        if let Some(mut slab) = transition_slab(key, face, config.voxel_size_m) {
            sampler.fill_block(&mut slab);
            std::hint::black_box(&slab);
        }
    }
    let t2 = Instant::now();
    let mesh = mesh_chunk(&chunk);
    let t3 = Instant::now();
    out.generate.push(ms(t1 - t0));
    out.slabs.push(ms(t2 - t1));
    out.mesh.push(ms(t3 - t2));
    out.total.push(ms(t3 - t0));
    out.triangles.push(mesh.triangle_count());
    let stats = sampler.height_cache_stats();
    out.nodes
        .push((stats.misses - misses) * stats.tile_nodes as u64);
}

fn report(planet: &PlanetParams, config: &WorldConfig, level: u8, cache: &str, s: &mut Samples) {
    let row = |name: &str, v: &mut Vec<f64>| {
        println!(
            "  {cache} {name:<9} min {:.3} ms  median {:.3} ms  p95 {:.3} ms  max {:.3} ms",
            percentile(v, 0.0),
            percentile(v, 0.5),
            percentile(v, 0.95),
            percentile(v, 1.0)
        );
    };
    row("generate", &mut s.generate);
    row("slabs", &mut s.slabs);
    row("mesh", &mut s.mesh);
    row("total", &mut s.total);
    let triangles = s.triangles.iter().sum::<usize>() as f64 / s.triangles.len() as f64;
    let nodes = s.nodes.iter().sum::<u64>() as f64 / s.nodes.len() as f64;
    println!("  {cache} nodes computed per chunk, slabs included: {nodes:.0} mean");
    println!(
        "BENCH-CHUNK radius_km={} voxel_size_m={} seed={} level={level} cache={cache} \
         generate_ms_p50={:.3} generate_ms_p95={:.3} slabs_ms_p50={:.3} slabs_ms_p95={:.3} \
         mesh_ms_p50={:.3} mesh_ms_p95={:.3} total_ms_p50={:.3} total_ms_p95={:.3} \
         triangles_mean={triangles:.0} nodes_mean={nodes:.0} budget_ms={}",
        planet.radius_km,
        config.voxel_size_m,
        config.seed,
        percentile(&mut s.generate, 0.5),
        percentile(&mut s.generate, 0.95),
        percentile(&mut s.slabs, 0.5),
        percentile(&mut s.slabs, 0.95),
        percentile(&mut s.mesh, 0.5),
        percentile(&mut s.mesh, 0.95),
        percentile(&mut s.total, 0.5),
        percentile(&mut s.total, 0.95),
        Budgets::CHUNK_GENERATE_MESH_MS
    );
}

/// Distinct chunks of `level` that hold the surface, found along directions of a Fibonacci
/// sphere from the full detail height.
fn surface_keys(
    sampler: &PlanetSampler,
    config: &WorldConfig,
    level: u8,
    count: usize,
) -> Vec<ChunkKey> {
    let mut keys = Vec::new();
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    let tries = 64 * count;
    for i in 0..tries {
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / tries as f64;
        let ring = (1.0 - y * y).sqrt();
        let angle = golden_angle * i as f64;
        let dir = DVec3::new(ring * angle.cos(), y, ring * angle.sin());
        let p = dir * (sampler.radius_m() + sampler.height_m(dir));
        let key = ChunkKey::containing_at(WorldPos::body_fixed(p), level, config);
        if keys.contains(&key) {
            continue;
        }
        let mut chunk = Chunk::for_key(key, config);
        sampler.fill_chunk(&mut chunk);
        if chunk.has_surface() {
            keys.push(key);
            if keys.len() == count {
                break;
            }
        }
    }
    keys
}

fn time_fill(sampler: &PlanetSampler, config: &WorldConfig, key: ChunkKey, runs: usize) -> f64 {
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let mut chunk = Chunk::for_key(key, config);
        let t0 = Instant::now();
        sampler.fill_chunk(&mut chunk);
        times.push(ms(t0.elapsed()));
    }
    percentile(&mut times, 0.5)
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1.0e3
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * p).round() as usize]
}
