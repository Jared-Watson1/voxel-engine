# Voxel Engine

A from scratch voxel game engine in Rust and wgpu for procedurally generated planetary systems: editable smooth voxel planets (a signed distance field with materials) generated from plate tectonics, erosion, drainage and climate, a spaceship, and a seamless flight from the surface through the atmosphere into orbit and across a solar system of orbiting planets and moons with no loading screens. Performance on mid range hardware (Apple M1 as the baseline) is the first design pillar, and every world is reproducible from its seed.

Milestone M0 (scaffold) is in place: a cargo workspace, a winit window with a wgpu device, a free fly camera with f64 positions rendered camera relative with reverse-Z depth, one 32^3 chunk meshed with Transvoxel from an analytic signed distance sphere, an egui overlay, puffin profiler spans and WGSL hot reload.

Engineering rules, the stack, the crate layout and the conventions are in [CLAUDE.md](CLAUDE.md). Design, research, decisions and the roadmap live outside the repo in the project's notes vault.

## Run

```
cargo run --release -p vx_game
```

Hold the right mouse button to look, WASD to move, Space or E up, Shift or Q down, scroll to change speed, F12 for a screenshot, Esc to quit. `cargo run --release -p vx_game -- --help` lists the measurement flags (`--voxel-size`, `--no-vsync`, `--bench-seconds`, `--screenshot`).

World scale parameters are in `assets/config/world.ron` (voxel size, design default 1.0 m); the M0 test sphere is in `assets/config/m0_scene.ron`. Both files and `assets/shaders/terrain.wgsl` reload while the engine runs.

## Develop

```
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo run --release -p vx_game --features profile   # puffin server on 127.0.0.1:8585
```
