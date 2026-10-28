# Voxel Engine

Guidance for Claude Code and any other agent working in this repository.

## What this is

Voxel Engine is a from scratch voxel game engine for procedurally generated planetary systems. Planets are made of editable smooth voxels (a signed distance field with materials) generated from plate tectonics, erosion, drainage and climate; the player walks on the surface, boards a spaceship, flies through the atmosphere into orbit and across a solar system of orbiting planets and moons with no loading screens; physics is believable; everything is reproducible from a seed. It is a solo passion project built piecewise in milestones, with performance on mid range hardware (Apple M1 as the baseline) treated as a feature.

## Where the thinking lives

Design, research, decisions and the roadmap live in the Obsidian vault at `~/Documents/projects/Voxel Engine/`, not in this repo. Before planning any non trivial work read, in order: `Voxel Engine/CLAUDE.md`, `Voxel Engine/Voxel Engine — Project Hub.md` (open threads), then the notes the task needs, usually the design document (`Product/Voxel Engine Design Document — Vision, Scope and Architecture.md`), the roadmap (`Product/Voxel Engine Roadmap — Milestones and Vertical Slices.md`) and the decision log (`Product/Voxel Engine Decision Log — Stack, Voxels, Frames and Generation.md`). Work orders for this repo are in `Voxel Engine/Plans/`. When a milestone step is done, tick it in the roadmap and update the relevant note's `status:` line the same day. Never open `Credentials.md` or any key file in the vault.

Decisions are numbered (D1, D2, ...). Code comments and commit messages that depend on one cite the number. A change that contradicts a decision does not happen until the decision log has a superseding entry.

## Art direction

The look is based on the indie voxel game Lay of the Land (Jared, 2026-10-27). The references show small cubic voxels for terrain, rocks, trees and buildings; lush saturated greens with blocky canopies and dense blocky grass; pale stone outcrops; warm soft sunlight with a strong blue sky ambient, soft shadows and ambient occlusion; strong aerial haze on distant ridges under cumulus clouds; crisp detailed textures on each block face. Reference captures and the full notes are in the vault: `Voxel Engine/Assets/art-reference-lay-of-the-land-vista.png`, `Voxel Engine/Assets/art-reference-lay-of-the-land-ground.png` and the Art direction section of the design document. The terrain stays smooth (D2, accepted by Jared 2026-10-27, because a cubic grid on a planet stair steps against the radial up); the look comes from lighting, sky ambient, haze, colours and block face materials, and a cubic layer for trees, grass, crops, rocks and structures on top of the smooth ground (the block building layer).

## Status

M2 (small planet, surface to orbit, under a real sky) built and demoed 2026-10-28 on branch `m2` (not merged into main yet), on top of M1 (editable terrain slab, done 2026-10-27) and M0. A 10 km planet in its body fixed frame is one voxel octree (D20) from 1 m voxels at the feet to the whole planet from 100 km: the `LodTree` requests keys, splits and merges commit only when the replacement is ready, every chunk is meshed with its six Transvoxel transition faces (vendored extraction with bridges, D22, D26) and the vertex shader moves vertices by the chunk's transition mask, so a neighbour's level change never remeshes. Heights come from fBm of the direction on a per level cube sphere node grid with a shared tile cache (D21, D27). Chunks draw from a persistent chunk table relative to a render origin on the voxel grid that rebases every 4 km (D23). Simulation time and Keplerian rails for a Sun like star, the planet (1 h day, 15 degree tilt) and a moon drive the sun and the moon (D24); the Hillaire atmosphere (four LUTs, deferred lighting, aerial perspective, HDR with the Khronos PBR Neutral tonemap) lights the terrain (D25). The fly camera keeps the horizon level against the radial up and scales speed with altitude. Provisional numbers on a loaded machine: the 70 s descent from the surface to 100 km and back runs at 58.8 fps with vsync (p99 21.5 ms) and 144 fps uncapped at 1920x1080; at 50 km and 0.5 m 59.3 fps. The performance gate is open: a quiet machine run, cold generate plus mesh over 4 ms at levels 3 and 6 (5.3 and 4.3 ms), the 50 km terrain and sky passes at their budgets, rare long frames. Details in the roadmap's M2 section and `Voxel Engine/Plans/2026-10-27-m2-small-planet.md`. D2 and D12 to D19 are accepted (2026-10-27); D20 to D27 are proposed. Update this paragraph when the state changes.

## Stack (D1, accepted 2026-10-26)

- Rust, stable toolchain (1.96 at the time of writing), edition 2024, cargo workspace. No nightly features.
- wgpu for graphics and compute (30.0.1 as of 2026-10-26; breaking majors ship about quarterly, upgrade only at milestone boundaries), WGSL shaders. Metal on macOS, Vulkan or DX12 on Windows, Vulkan on Linux. The renderer uses only features available on all three: compute, storage buffers, single indirect draws with vertex pulling, binding arrays for textures, subgroups, timestamp queries. No dependency on mesh shaders, ray tracing, multi draw indirect with count buffers (Metal emulates multi draw as a CPU loop) or f64 in shaders (Metal has none).
- winit 0.30.x for windowing and input (0.31 is beta; egui-wgpu 0.37 pins 0.30), egui 0.37 (egui-wgpu, egui-winit) for debug UI and tools, glam 0.34 for math (f64 types for world positions, f32 SIMD types for rendering), rayon or a custom work stealing pool for jobs, a frame profiler from the first commit (puffin through the `profiling` crate, chosen in M0 because tracy-client compiles C++; wgpu-profiler for GPU scopes later, Apple GPUs give no timestamps inside passes), tracing for logs, serde for data files.
- Physics: leaning `rapier3d` (pure Rust, has a voxel collider); `oxijolt` (Jolt bindings, needs CMake on macOS) is the fallback. Decided for good before M5 (D7). Orbital mechanics are hand written in `vx_universe`, never in the physics library (D6).
- Noise and generation are pure Rust and CPU authoritative (the world must be bit identical on every machine). The `fastnoise2` bindings are optional and need CMake or a prebuilt library. GPU noise in WGSL only for far field preview geometry outside the blend band.
- No game engine crates (Bevy, Fyrox) and no Godot or Unity. Those are the thing being built.
- Record every dependency's version and the reason it was added in the "Dependencies" section below when it lands.

## Dependencies

Versions are the latest stable on crates.io when added (2026-10-26). All pure Rust on macOS: `cargo tree -i cc` is empty.

| Crate | Version | License | Used by | Why | Added |
|---|---|---|---|---|---|
| wgpu | 30.0.1 | MIT OR Apache-2.0 | vx_render | graphics and compute on Metal, Vulkan, DX12 (D1) | M0 |
| winit | 0.30.13 | Apache-2.0 | vx_render, vx_game | window and input; 0.30 because egui-winit 0.37 pins it (D1) | M0 |
| egui, egui-wgpu, egui-winit | 0.37.0 | MIT OR Apache-2.0 | vx_render, vx_game | debug overlay; egui with `default_fonts` (D1) | M0 |
| glam | 0.34.1 | MIT OR Apache-2.0 | vx_core, vx_voxel, vx_render, vx_game | f64 world math and f32 render math (D1, D4) | M0 |
| bytemuck | 1.25.2 | Zlib OR Apache-2.0 OR MIT | vx_render | `Pod` uniform and vertex structs (engineering rule 9) | M0 |
| pollster | 1.0.1 | Apache-2.0 OR MIT | vx_render | resolves adapter, device and error scope futures on native | M0 |
| anyhow | 1.0.104 | MIT OR Apache-2.0 | vx_game | errors in the binary | M0 |
| thiserror | 2.0.21 | MIT OR Apache-2.0 | vx_core, vx_render | errors in library crates | M0 |
| tracing, tracing-subscriber | 0.1.44, 0.3.23 | MIT | vx_render, vx_game | logs | M0 |
| profiling | 1.0.18 | MIT OR Apache-2.0 | vx_voxel, vx_render, vx_game | profiler scopes, zero cost without the `profile` feature | M0 |
| puffin, puffin_http | 0.20.0, 0.17.0 | MIT OR Apache-2.0 | vx_game (`profile` feature) | profiler backend and server; tracy-client was not used because it compiles C++ | M0 |
| notify | 8.2.1 | CC0-1.0 | vx_core | file watching for shader and config hot reload (9.0 is a release candidate) | M0 |
| serde, ron | 1.0.229, 0.12.2 | MIT OR Apache-2.0 | vx_core, vx_game | `WorldConfig` and scene config files (D11) | M0 |
| transvoxel-data | 0.2.1 | MIT | vx_voxel | Lengyel's Transvoxel cell tables for the vendored extraction in `vx_voxel::transvoxel` (D22, D26); the `transvoxel` 2.0.0 crate itself was removed in M2 (its extraction is vendored with attribution, MIT OR Apache-2.0); the vault still lists the license of Lengyel's original tables as unverified | M2 |
| png | 0.18.1 | MIT OR Apache-2.0 | vx_game | screenshot encoding on a worker thread | M0 |
| fastnoise-lite | 1.1.1 | MIT | vx_worldgen | OpenSimplex2 and fBm noise, pure safe Rust, no transcendentals, `f64` input feature; deterministic across machines (D13); pulls only `num-traits` | M1 |

## Crate layout

```
Cargo.toml                workspace
crates/vx_core         f64/f32 math and frames, ids, time, job system and pipelines, allocators, config
crates/vx_voxel        chunk storage and compression, SDF ops and brushes, octree clipmap and LOD, Transvoxel meshing, edit deltas, serialization
crates/vx_worldgen     planet parameters, planet atlas (plates, uplift, erosion, drainage, climate, biomes), noise stack, SDF sampler, materials
crates/vx_universe     star system model, Keplerian propagation, spheres of influence, frames and rebasing, simulation time and warp
crates/vx_render       wgpu device, frame graph, GPU driven terrain pass, far surface patches, atmosphere, ocean, shadows, post, debug draw, egui
crates/vx_physics      rigid body integration, SDF terrain colliders, character controller, ship flight model, buoyancy
crates/vx_game         the binary: window, input, app loop, player and ship states, camera, HUD, settings
crates/vx_tools        planet atlas viewer, chunk inspector, headless benchmarks
assets/                   shaders (WGSL), config (RON), CC0 textures, data tables; no hand authored meshes
```

M2 state: `vx_core` (frames, `camera_relative`, reverse-Z `Projection`, `FixedStep`, `WorldConfig` with the LOD and render origin fields, `PlanetParams` with `TerrainParams` and `AtmosphereParams`, `FileWatcher`, the `JobSystem`, `Budgets` and `BudgetMonitor`, the 16 byte `PackedVertex` with the transition flag and `secondary_position_voxels`), `vx_voxel` (`ChunkKey` with levels, `Face`, the dense `Chunk`, `SampleBlock` and `transition_slab`, `ChunkStore`, `Brush`, the `LodTree`, `mesh_chunk_lod` through the vendored `transvoxel` module), `vx_worldgen` (`PlanetSampler` with its cube sphere height grid and tile cache, `material_textures`), `vx_universe` (`SimClock`, Kepler orbits, `RotationState`, `StarSystem` and `SystemConfig`), `vx_render` (device, persistent chunk table, compute culling with the horizon test and transition ranges, G-buffer terrain, Hillaire LUTs, deferred lighting, tonemap, GPU pass timings, egui, hot reload of every shader, frame capture), `vx_game` (the planet `World` driven by the octree, render origin, brushes and picking, the radial `FlyCamera`, the `Sky` clock, scripted surface loop and descent, overlay, automation flags) and `vx_tools` (`bench-chunk` on the planet) have code. `vx_physics` is an empty stub until M5. `vx_render` depends on `vx_core` only; `vx_worldgen` depends on `vx_voxel`; `vx_universe` on `vx_core`.

`vx` is short for Voxel Engine. Dependencies point downward only: `vx_game` depends on everything, `vx_core` on nothing in the workspace. `vx_render` never depends on `vx_worldgen` or `vx_physics`; it receives meshes and parameters.

## Engineering rules

These follow from the pillars in the design document. The pillar order (performance first, seamless scale, the world is data, believable not photoreal, piecewise and demoable) settles conflicts.

1. **Positions are f64 in an explicit frame; the GPU sees camera relative f32 (D4).** No raw world position ever reaches a shader. Every frame conversion goes through `vx_core` frames. Reverse-Z with a float depth buffer and an infinite far plane; depth clears to 0 and the compare is Greater.
2. **Generation is a pure function of (position, seed, planet atlas) (D5).** The same seed gives the same world on every machine and run. No randomness outside seeded generators; no reliance on iteration order of hash maps for anything that affects the world.
3. **The main thread never blocks (D9).** No file I/O, generation, meshing or compression on the main thread. Every heavy task is a job with a measured duration; every pipeline stage reports its queue length to the debug UI.
4. **Budgets are code.** Frame time, resident voxel memory, GPU pool size and per frame upload bytes are constants with names, checked at runtime in debug builds and shown in the egui overlay. The targets are in the design document and are revised only with measurements.
5. **Measure before optimising, and keep the measurement.** Profiler spans on every stage. A headless benchmark (`cargo bench` or a `vx_tools` subcommand) for chunk generate and mesh; regressions are numbers, not feelings.
6. **Seams are bugs.** Cracks between LOD levels, visible transitions between the voxel shell and the far surface, pops at frame switches and jitter at distance are fixed before new features land.
7. **Data oriented layout.** Chunks, meshes and entities live in flat arrays and pools indexed by handles; no `Rc<RefCell<>>` graphs for hot data. `unsafe` only in `vx_core` allocators and GPU upload paths, with a comment stating the invariant.
8. **Determinism in simulation.** Fixed time step for physics and orbital integration; rendering interpolates. Time warp multiplies the step count, never the step size, except for bodies on rails which are analytic.
9. **Shaders are code.** WGSL files live in `assets/shaders/`, hot reload in debug builds, compile errors log and keep the previous pipeline. Units in shaders are metres; every uniform struct has a matching `#[repr(C)]` Rust struct validated with `bytemuck`.
10. **No art pipeline.** Materials are procedural or CC0 textures (Poly Haven, ambientCG) with their license noted in `assets/LICENSES.md`. No hand authored meshes.
11. **Scale is a parameter with a design default (D11).** Voxel size, planet radius, LOD radii, atmosphere depth and every other world scale value is a named field in `WorldConfig` or `PlanetParams` with its default documented beside it (1 m voxels, a 10 km planet for M2, 600 km for generated worlds) and a supported range. `CHUNK_SIZE` (32) is the one compile time constant. Tests run at the defaults and at both ends of each range; performance numbers always state the parameter values they were measured at.

## Coding conventions

- `cargo fmt` and `cargo clippy --all-targets -- -D warnings` clean before every commit. `cargo test` green.
- Comments only where they add information the code cannot: an invariant, a non obvious reason, a reference (paper, decision number, URL). No comments that restate the code. No pronouns in comments or docs (no "we", "I", "you"); describe the thing.
- No emojis anywhere, including commit messages and UI strings. Avoid dashes inside sentences in docs and comments; use a period, comma, colon or parentheses.
- Names: `snake_case` for modules and functions, types in `CamelCase`, SI units in names where the unit is not obvious (`radius_m`, `dt_s`, `altitude_km`). Frames in type names where it matters (`BodyFixedPos`, `SystemPos`).
- Errors: `thiserror` in library crates, `anyhow` in the binary. No `unwrap()` outside tests and the first lines of `main` except on invariants that are documented as such.
- Public API of every crate is small and documented with `///`; internal modules are `pub(crate)`.
- Tests next to the code (`#[cfg(test)]`), integration tests in `tests/` for cross crate behaviour (frames, pipeline), property style tests for SDF and meshing invariants (closed meshes, outward normals, no NaN).
- Commit messages: a short imperative subject naming the milestone or system (`M0: reverse-Z projection and camera relative frame`), a body that says why, and the decision numbers touched.

## Commands

```
cargo run --release -p vx_game                                  run the engine (standing on the planet at a morning sun)
cargo run --release -p vx_game --features profile               with the puffin server on 127.0.0.1:8585 (view with puffin_viewer)
cargo run --release -p vx_game -- --descent                     stand, climb to 100 km, come back (the M2 exit run), print a BENCH line, exit
cargo run --release -p vx_game -- --descent --radius-km 50 --voxel-size 0.5 --lod-levels 10   the M2 parameter run
cargo run --release -p vx_game -- --flight-km 2                 fly a 2 km square loop over the surface, print a BENCH line, exit
cargo run --release -p vx_game -- --bench-seconds 30 --edit-test  measure standing still with scripted dig and place brushes
cargo run --release -p vx_game -- --time-of-day 0.27 --altitude 8   start at a local solar time, optionally high above the terrain
cargo run --release -p vx_game -- --no-vsync --warp 60 --exposure 1  uncapped, time warp, exposure (also --seed, --workers, --lod-radius)
cargo run --release -p vx_game -- --screenshot out.png          save a PNG after the world settles and exit (add --no-overlay)
cargo run --release -p vx_tools -- bench-chunk --level 3        generate, transition slabs and mesh per surface chunk, cold and warm cache
cargo test --workspace                                          tests
cargo clippy --all-targets -- -D warnings                       lints (also run with --features vx_game/profile)
```

Controls: hold the right mouse button to look, WASD to move, Space or E up and Shift or Q down (away from and towards the planet), scroll to scale the speed, left click to dig or place, B to switch, [ and ] for the brush radius, , and . to slow or speed up time (the slowest step pauses), - and = for exposure, F12 screenshot, Esc quit.

Automated runs (`--bench-seconds`, `--flight-km`, `--descent`, `--screenshot`) ignore keyboard and mouse input and hold the simulation clock unless `--warp` is given; they wait for the octree to settle, warm up for two seconds, then measure. Performance numbers state the planet radius, voxel size, LOD levels and radius, resolution and present mode they were measured at; the BENCH line includes all of them, plus GPU pass means where the adapter has timestamps. Measure on a quiet machine. The headless benchmark is the `vx_tools` subcommand above; it prints a BENCH-CHUNK line.

## Working here

- Work in milestones from the roadmap. Do not start a later milestone's feature to unblock a current one; write the thread into the hub's open threads instead.
- Each session that changes code ends with: tests and clippy green, the roadmap checkboxes updated, the vault note's `status:` updated, this file's Status section updated if the state changed.
- Prefer the vault for anything a future session needs to know and the code for anything the compiler can check. Do not duplicate the design document here; link to it.
- Ask before adding a dependency over roughly 50k lines or any dependency that pulls a C or C++ build (except the physics library chosen in D7); record it in the Dependencies section with its license.
