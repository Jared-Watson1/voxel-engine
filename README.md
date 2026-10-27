# Voxel Engine

A from scratch voxel game engine in Rust and wgpu for procedurally generated planetary systems: editable smooth voxel planets (a signed distance field with materials) generated from plate tectonics, erosion, drainage and climate, a spaceship, and a seamless flight from the surface through the atmosphere into orbit and across a solar system of orbiting planets and moons with no loading screens. Performance on mid range hardware (Apple M1 as the baseline) is the first design pillar, and every world is reproducible from its seed.

Milestone M1 (editable terrain slab) is in place on top of the M0 scaffold: a flat world of seeded noise terrain with four materials streamed around the camera through a job system (generate, mesh, upload), palette compressed chunks, pooled GPU buffers with compute culling and one indirect draw, triplanar materials, dig and place brushes that remesh across chunk borders, named performance budgets in the overlay, and a headless chunk benchmark.

Engineering rules, the stack, the crate layout and the conventions are in [CLAUDE.md](CLAUDE.md). Design, research, decisions and the roadmap live outside the repo in the project's notes vault.

## Run

```
cargo run --release -p vx_game
```

Hold the right mouse button to look, WASD to move, Space or E up, Shift or Q down, scroll to change speed, left mouse button to dig (B switches to place, `[` and `]` change the brush radius), F12 for a screenshot, Esc to quit. `cargo run --release -p vx_game -- --help` lists the measurement flags (`--flight-km`, `--edit-test`, `--bench-seconds`, `--screenshot`, `--no-vsync`, `--voxel-size`, `--view-radius`, `--seed`, `--workers`).

World scale parameters are in `assets/config/world.ron` (voxel size with a design default of 1.0 m, the seed, the view radius in chunks, the terrain shape and materials). The config and the shaders in `assets/shaders/` reload while the engine runs.

## Develop

```
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo run --release -p vx_game --features profile   # puffin server on 127.0.0.1:8585
cargo run --release -p vx_tools -- bench-chunk      # generate and mesh timings for one chunk, one thread
```
