//! Headless tools: `bench-chunk` measures generate and mesh of single chunks on one thread
//! (engineering rule 5), so regressions are numbers.

use std::time::Instant;

use glam::DVec3;
use vx_core::{Budgets, CHUNK_SIZE, WorldConfig, WorldPos};
use vx_voxel::{Chunk, ChunkKey, ChunkStore, mesh_chunk};
use vx_worldgen::TerrainSampler;

const USAGE: &str = "\
vx_tools: Voxel Engine headless tools

USAGE: vx_tools bench-chunk [--voxel-size <m>] [--seed <n>] [--runs <n>] [--chunks <n>]

Generates and meshes surface chunks of the default world on one thread and prints the median
and p95 of generate, mesh and generate plus mesh per chunk, against the 4 ms budget. Also times
an all air and an all rock chunk and the compression of a surface chunk.";

struct Options {
    voxel_size_m: Option<f64>,
    seed: Option<u64>,
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
        voxel_size_m: None,
        seed: None,
        runs: 100,
        chunks: 8,
    };
    while let Some(flag) = args.next() {
        let value = args.next();
        let parsed: Result<(), String> = match (flag.as_str(), value) {
            ("--voxel-size", Some(v)) => v
                .parse()
                .map(|v| options.voxel_size_m = Some(v))
                .map_err(|e| e.to_string()),
            ("--seed", Some(v)) => v
                .parse()
                .map(|v| options.seed = Some(v))
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

fn bench_chunk(options: &Options) -> Result<(), String> {
    let mut config = WorldConfig::default();
    if let Some(v) = options.voxel_size_m {
        config.voxel_size_m = v;
    }
    if let Some(s) = options.seed {
        config.seed = s;
    }
    config.validate().map_err(|e| e.to_string())?;
    let sampler = TerrainSampler::new(config.seed, config.terrain);
    let extent = config.chunk_extent_m();

    // Surface chunks: the chunk containing the terrain surface above a spread of columns.
    let mut keys = Vec::new();
    let mut i = 0u32;
    while keys.len() < options.chunks.max(1) {
        let angle = f64::from(i) * 2.399_963; // golden angle spiral, spread over a few km
        let r = 40.0 * f64::from(i).sqrt() * extent / 32.0;
        let (x, z) = (r * angle.cos(), r * angle.sin());
        let h = sampler.height_m(x, z);
        let key = ChunkKey::containing(WorldPos::body_fixed(DVec3::new(x, h, z)), &config);
        if !keys.contains(&key) {
            keys.push(key);
        }
        i += 1;
    }

    let runs = options.runs.max(1);
    let mut generate = Vec::new();
    let mut mesh = Vec::new();
    let mut both = Vec::new();
    let mut compress = Vec::new();
    let mut triangles = Vec::new();
    let mut bytes = Vec::new();
    for key in &keys {
        for _ in 0..runs {
            let mut chunk = Chunk::for_key(*key, &config);
            let t0 = Instant::now();
            sampler.fill_chunk(&mut chunk);
            let t1 = Instant::now();
            let m = mesh_chunk(&chunk);
            let t2 = Instant::now();
            let store = ChunkStore::compress(&chunk);
            let t3 = Instant::now();
            generate.push(ms(t1 - t0));
            mesh.push(ms(t2 - t1));
            both.push(ms(t2 - t0));
            compress.push(ms(t3 - t2));
            triangles.push(m.triangle_count());
            bytes.push(store.resident_bytes());
        }
    }

    let air_key = ChunkKey::new(0, 1_000, 0);
    let rock_key = ChunkKey::new(0, -1_000, 0);
    let air = time_fill(&sampler, &config, air_key, runs);
    let rock = time_fill(&sampler, &config, rock_key, runs);

    let tri_mean = triangles.iter().sum::<usize>() as f64 / triangles.len() as f64;
    let bytes_mean = bytes.iter().sum::<usize>() as f64 / bytes.len() as f64;
    println!(
        "bench-chunk: voxel_size_m={} seed={} chunk_extent_m={} surface_chunks={} runs_each={} one thread",
        config.voxel_size_m,
        config.seed,
        extent,
        keys.len(),
        runs
    );
    println!("  surface chunk keys: {keys:?}");
    println!(
        "  generate         median {:.3} ms  p95 {:.3} ms  max {:.3} ms",
        percentile(&mut generate, 0.5),
        percentile(&mut generate, 0.95),
        percentile(&mut generate, 1.0)
    );
    println!(
        "  mesh             median {:.3} ms  p95 {:.3} ms  max {:.3} ms  ({:.0} triangles mean)",
        percentile(&mut mesh, 0.5),
        percentile(&mut mesh, 0.95),
        percentile(&mut mesh, 1.0),
        tri_mean
    );
    println!(
        "  generate + mesh  median {:.3} ms  p95 {:.3} ms  max {:.3} ms  (budget {} ms)",
        percentile(&mut both, 0.5),
        percentile(&mut both, 0.95),
        percentile(&mut both, 1.0),
        Budgets::CHUNK_GENERATE_MESH_MS
    );
    println!(
        "  compress         median {:.3} ms  p95 {:.3} ms  ({:.0} bytes mean of {} raw)",
        percentile(&mut compress, 0.5),
        percentile(&mut compress, 0.95),
        bytes_mean,
        (CHUNK_SIZE + 3).pow(3) * 2
    );
    println!(
        "  air chunk generate   median {:.3} ms; rock chunk generate median {:.3} ms",
        air, rock
    );
    println!(
        "BENCH-CHUNK voxel_size_m={} seed={} generate_ms_p50={:.3} generate_ms_p95={:.3} mesh_ms_p50={:.3} \
         mesh_ms_p95={:.3} total_ms_p50={:.3} total_ms_p95={:.3} compress_ms_p50={:.3} triangles_mean={:.0} \
         store_bytes_mean={:.0} air_ms_p50={:.3} rock_ms_p50={:.3}",
        config.voxel_size_m,
        config.seed,
        percentile(&mut generate, 0.5),
        percentile(&mut generate, 0.95),
        percentile(&mut mesh, 0.5),
        percentile(&mut mesh, 0.95),
        percentile(&mut both, 0.5),
        percentile(&mut both, 0.95),
        percentile(&mut compress, 0.5),
        tri_mean,
        bytes_mean,
        air,
        rock
    );
    Ok(())
}

fn time_fill(sampler: &TerrainSampler, config: &WorldConfig, key: ChunkKey, runs: usize) -> f64 {
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
