use vx_voxel::Material;

use crate::seed::{STREAM_TEXTURES, derive_seed64, splitmix64};

/// Lattice cells across the image at the coarsest octave; each finer octave doubles it.
const BASE_PERIOD: u32 = 8;
/// Smallest lattice cell in texels. Finer octaves would put most of their variation between
/// adjacent texels, which reads as noise rather than surface detail.
const MIN_CELL_PX: u32 = 4;
/// Amplitude ratio between successive octaves.
const OCTAVE_GAIN: f64 = 0.5;
/// Octaves of the hue variation fractal: broad patches only.
const TINT_OCTAVES: usize = 3;
/// Ridge value (0 to 1) above which a rock vein darkens the surface.
const VEIN_THRESHOLD: f64 = 0.82;
/// Edge of a mosaic cell in texels.
const MOSAIC_CELL_PX: u32 = 4;
/// Debug colour of the air layer, never visible on a correct mesh.
const AIR_RGBA: [u8; 4] = [255, 0, 255, 255];

/// Procedural material albedo layers (D18): one RGBA8 sRGB image per material id, layer index
/// equal to the id (0 is air and gets a debug colour), all `size` by `size`, row major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialTextureSet {
    pub size: u32,
    pub layers: Vec<Vec<u8>>,
}

impl MaterialTextureSet {
    pub fn layer_count(&self) -> u32 {
        self.layers.len() as u32
    }

    pub fn bytes(&self) -> usize {
        self.layers.iter().map(Vec::len).sum()
    }
}

/// Generates [`vx_voxel::Material::COUNT`] tileable layers from seeded noise. Pure in
/// (seed, size); runs on a worker at startup.
///
/// Every layer is built from value noise on lattices that wrap at the image edge, so the result
/// tiles by construction. Only arithmetic, `floor`, `abs` and `clamp` are used (D5).
pub fn material_textures(seed: u64, size: u32) -> MaterialTextureSet {
    profiling::scope!("material_textures");
    let texture_seed = derive_seed64(seed, STREAM_TEXTURES);
    let layers = (0..Material::COUNT)
        .map(|id| layer(texture_seed, size, Material(id as u8)))
        .collect();
    MaterialTextureSet { size, layers }
}

/// Colour recipe of one material, sRGB 0 to 255 per channel.
#[derive(Clone, Copy)]
struct Recipe {
    /// Mean colour.
    base: [f64; 3],
    /// Per channel amplitude of the broad fractal variation (noise in -1 to 1).
    variation: [f64; 3],
    /// Colour added in proportion to a second, coarser fractal mapped to 0 to 1.
    tint: [f64; 3],
    /// Amplitude of the finest lattice, equal on all channels (grain).
    speckle: f64,
    /// Amplitude of the mosaic: one flat value per `MOSAIC_CELL_PX` square with hard edges, the
    /// crisp block face detail of the art direction.
    mosaic: f64,
    /// Darkening along the ridges of a third fractal (rock veins); zero disables it.
    veins: f64,
}

/// Albedo recipes after the Lay of the Land art direction (design document, Art direction): lush
/// saturated grass, pale cream grey limestone, warm brown soil, light sand and bright snow.
fn recipe(material: Material) -> Recipe {
    match material {
        Material::SOIL => Recipe {
            base: [128.0, 88.0, 56.0],
            variation: [16.0, 12.0, 8.0],
            tint: [16.0, 10.0, 4.0],
            speckle: 5.0,
            mosaic: 10.0,
            veins: 0.0,
        },
        Material::GRASS => Recipe {
            base: [80.0, 146.0, 44.0],
            variation: [10.0, 16.0, 6.0],
            tint: [34.0, 18.0, -8.0],
            speckle: 4.0,
            mosaic: 12.0,
            veins: 0.0,
        },
        Material::SAND => Recipe {
            base: [214.0, 198.0, 152.0],
            variation: [8.0, 8.0, 7.0],
            tint: [8.0, 5.0, -2.0],
            speckle: 4.0,
            mosaic: 6.0,
            veins: 0.0,
        },
        Material::SNOW => Recipe {
            base: [232.0, 237.0, 244.0],
            variation: [5.0, 5.0, 4.0],
            tint: [-6.0, -3.0, 2.0],
            speckle: 3.0,
            mosaic: 4.0,
            veins: 0.0,
        },
        _ => Recipe {
            base: [190.0, 184.0, 166.0],
            variation: [14.0, 14.0, 13.0],
            tint: [10.0, 7.0, 2.0],
            speckle: 5.0,
            mosaic: 12.0,
            veins: 34.0,
        },
    }
}

fn layer(texture_seed: u64, size: u32, material: Material) -> Vec<u8> {
    let texels = size as usize * size as usize;
    if material == Material::AIR {
        return AIR_RGBA.repeat(texels);
    }
    let recipe = recipe(material);
    let layer_seed = splitmix64(texture_seed ^ (u64::from(material.0) << 56));
    let variation = PeriodicFbm::new(splitmix64(layer_seed ^ 0x11), size, usize::MAX);
    let tint = PeriodicFbm::new(splitmix64(layer_seed ^ 0x22), size, TINT_OCTAVES);
    let veins = (recipe.veins > 0.0)
        .then(|| PeriodicFbm::new(splitmix64(layer_seed ^ 0x33), size, usize::MAX));
    let speckle = PeriodicLattice::new(splitmix64(layer_seed ^ 0x44), (size / 2).max(1));
    // Tiny layers (debug sizes) skip the mosaic: a few cells across would dominate the image.
    let mosaic_period = size / MOSAIC_CELL_PX;
    let mosaic_amplitude = if mosaic_period >= 8 {
        recipe.mosaic
    } else {
        0.0
    };
    let mosaic = PeriodicLattice::new(splitmix64(layer_seed ^ 0x55), mosaic_period.max(1));
    let size_f = f64::from(size);

    let mut out = Vec::with_capacity(texels * 4);
    for y in 0..size {
        for x in 0..size {
            let v = variation.at_texel(x, y);
            let t = tint.at_texel(x, y) * 0.5 + 0.5;
            let s = speckle.at_texel(x, y, size_f);
            let m = mosaic.at_cell(x, y, size_f);
            let dark = veins.as_ref().map_or(0.0, |veins| {
                let ridge = 1.0 - veins.at_texel(x, y).abs();
                let mask = ((ridge - VEIN_THRESHOLD) / (1.0 - VEIN_THRESHOLD)).clamp(0.0, 1.0);
                recipe.veins * fade(mask)
            });
            for ((base, variation), tint) in
                recipe.base.iter().zip(&recipe.variation).zip(&recipe.tint)
            {
                let value =
                    base + variation * v + tint * t + recipe.speckle * s + mosaic_amplitude * m
                        - dark;
                out.push(value.round().clamp(0.0, 255.0) as u8);
            }
            out.push(255);
        }
    }
    out
}

/// Value noise on a square lattice of `period` cells that wraps, so any image whose edge spans a
/// whole number of periods tiles seamlessly.
struct PeriodicLattice {
    period: u32,
    values: Vec<f64>,
}

impl PeriodicLattice {
    fn new(seed: u64, period: u32) -> Self {
        let values = (0..period)
            .flat_map(|j| {
                (0..period)
                    .map(move |i| unit(splitmix64(seed ^ (u64::from(j) << 32 | u64::from(i)))))
            })
            .collect();
        Self { period, values }
    }

    /// Value at texel `(x, y)` of an image `size` texels wide, in -1 to 1.
    fn at_texel(&self, x: u32, y: u32, size: f64) -> f64 {
        let period = f64::from(self.period);
        self.sample(f64::from(x) * period / size, f64::from(y) * period / size)
    }

    /// Value of the lattice cell holding texel `(x, y)`, without interpolation: hard edged
    /// squares.
    fn at_cell(&self, x: u32, y: u32, size: f64) -> f64 {
        let period = f64::from(self.period);
        let i = ((f64::from(x) * period / size).floor() as u32).min(self.period - 1);
        let j = ((f64::from(y) * period / size).floor() as u32).min(self.period - 1);
        self.values[(j * self.period + i) as usize]
    }

    /// Value at lattice coordinates `(u, v)` in cells, both at least zero.
    fn sample(&self, u: f64, v: f64) -> f64 {
        let cu = u.floor();
        let cv = v.floor();
        let fu = fade(u - cu);
        let fv = fade(v - cv);
        let p = self.period;
        let i0 = (cu as u32) % p;
        let i1 = (i0 + 1) % p;
        let j0 = (cv as u32) % p;
        let j1 = (j0 + 1) % p;
        let at = |i: u32, j: u32| self.values[(j * p + i) as usize];
        let top = at(i0, j0) + (at(i1, j0) - at(i0, j0)) * fu;
        let bottom = at(i0, j1) + (at(i1, j1) - at(i0, j1)) * fu;
        top + (bottom - top) * fv
    }
}

/// Octaves of [`PeriodicLattice`] from `BASE_PERIOD` cells up to cells of `MIN_CELL_PX` texels,
/// normalised to -1 to 1.
struct PeriodicFbm {
    size: f64,
    octaves: Vec<(PeriodicLattice, f64)>,
    norm: f64,
}

impl PeriodicFbm {
    fn new(seed: u64, size: u32, max_octaves: usize) -> Self {
        let mut octaves = Vec::new();
        let mut period = BASE_PERIOD.min((size / MIN_CELL_PX).max(1));
        let mut amplitude = 1.0;
        let mut norm = 0.0;
        loop {
            let octave = octaves.len() as u64;
            octaves.push((
                PeriodicLattice::new(splitmix64(seed ^ octave), period),
                amplitude,
            ));
            norm += amplitude;
            if octaves.len() >= max_octaves || period.saturating_mul(2 * MIN_CELL_PX) > size {
                break;
            }
            period *= 2;
            amplitude *= OCTAVE_GAIN;
        }
        Self {
            size: f64::from(size),
            octaves,
            norm,
        }
    }

    fn at_texel(&self, x: u32, y: u32) -> f64 {
        let sum: f64 = self
            .octaves
            .iter()
            .map(|(lattice, amplitude)| amplitude * lattice.at_texel(x, y, self.size))
            .sum();
        sum / self.norm
    }
}

/// Maps a hash to -1 to 1 from its top 53 bits.
fn unit(hash: u64) -> f64 {
    (hash >> 11) as f64 * (2.0 / (1u64 << 53) as f64) - 1.0
}

/// Quintic smoothstep (Perlin, "Improving noise", 2002): zero first and second derivatives at
/// both ends, so lattice cell borders stay invisible.
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// Upper bound on the mean per channel difference across a seam: a seam is gentle.
    const SEAM_MEAN_TOLERANCE: f64 = 12.0;

    /// Largest per channel difference between any two texels `step` apart along `axis` (0 for x,
    /// 1 for y), and the mean of the differences across the wrap when `step` is `size - 1`.
    fn adjacent_differences(layer: &[u8], size: u32, axis: usize, step: u32) -> (i32, f64) {
        let mut max = 0;
        let mut sum = 0.0;
        let mut count = 0.0;
        for j in 0..size {
            for i in 0..size - step {
                let (a, b) = if axis == 0 {
                    (texel(layer, size, i, j), texel(layer, size, i + step, j))
                } else {
                    (texel(layer, size, j, i), texel(layer, size, j, i + step))
                };
                for c in 0..3 {
                    let d = (a[c] - b[c]).abs();
                    max = max.max(d);
                    sum += f64::from(d);
                    count += 1.0;
                }
            }
        }
        (max, sum / count)
    }

    fn texel(layer: &[u8], size: u32, x: u32, y: u32) -> [i32; 4] {
        let i = ((y * size + x) * 4) as usize;
        [
            i32::from(layer[i]),
            i32::from(layer[i + 1]),
            i32::from(layer[i + 2]),
            i32::from(layer[i + 3]),
        ]
    }

    /// Mean luminance per material after the art direction: soil darkest, then grass, pale
    /// limestone, light sand and bright snow.
    fn expected_luminance(material: Material) -> std::ops::RangeInclusive<f64> {
        match material {
            Material::SOIL => 0.3..=0.45,
            Material::GRASS => 0.42..=0.6,
            Material::ROCK => 0.62..=0.8,
            Material::SAND => 0.7..=0.85,
            Material::SNOW => 0.85..=0.97,
            _ => 0.0..=1.0,
        }
    }

    #[test]
    fn grass_is_saturated_and_rock_is_pale() {
        let set = material_textures(1, 64);
        let mean = |id: Material| {
            let layer = &set.layers[usize::from(id.0)];
            let mut sum = [0.0f64; 3];
            for px in layer.chunks_exact(4) {
                for c in 0..3 {
                    sum[c] += f64::from(px[c]);
                }
            }
            sum.map(|v| v / (layer.len() / 4) as f64)
        };
        let grass = mean(Material::GRASS);
        assert!(
            grass[1] > 1.5 * grass[0] && grass[1] > 2.5 * grass[2],
            "{grass:?}"
        );
        let rock = mean(Material::ROCK);
        let spread = rock.iter().cloned().fold(f64::MIN, f64::max)
            - rock.iter().cloned().fold(f64::MAX, f64::min);
        assert!(spread < 30.0 && rock[0] > rock[2], "{rock:?}");
        assert_eq!(set.layer_count(), 6, "air, rock, soil, grass, sand, snow");
    }

    fn mean_luminance(layer: &[u8]) -> f64 {
        let texels = layer.len() / 4;
        let sum: f64 = layer
            .chunks_exact(4)
            .map(|px| {
                0.2126 * f64::from(px[0]) + 0.7152 * f64::from(px[1]) + 0.0722 * f64::from(px[2])
            })
            .sum();
        sum / (texels as f64 * 255.0)
    }

    #[test]
    fn textures_are_tileable_and_sized() {
        for size in [8u32, 64, 256] {
            let set = material_textures(1, size);
            assert_eq!(set.size, size);
            assert_eq!(set.layer_count(), Material::COUNT as u32);
            assert_eq!(set.bytes(), Material::COUNT * (size * size * 4) as usize);
            for (id, layer) in set.layers.iter().enumerate() {
                assert_eq!(layer.len(), (size * size * 4) as usize, "layer {id}");
                for px in layer.chunks_exact(4) {
                    assert_eq!(px[3], 255, "layer {id} alpha");
                    if id == 0 {
                        assert_eq!(px, AIR_RGBA, "layer 0 is magenta");
                    }
                }
                // The seam (left column against right column, top row against bottom row) is
                // no sharper than any interior pair of adjacent texels, and gentle on average.
                for axis in [0, 1] {
                    let (interior_max, _) = adjacent_differences(layer, size, axis, 1);
                    let (seam_max, seam_mean) = adjacent_differences(layer, size, axis, size - 1);
                    assert!(
                        seam_max <= interior_max,
                        "layer {id} size {size} axis {axis}: seam {seam_max} vs interior {interior_max}"
                    );
                    assert!(
                        seam_mean <= SEAM_MEAN_TOLERANCE,
                        "layer {id} size {size} axis {axis}: seam mean {seam_mean}"
                    );
                }
                if id > 0 {
                    let luminance = mean_luminance(layer);
                    let expected = expected_luminance(Material(id as u8));
                    assert!(
                        expected.contains(&luminance),
                        "layer {id} size {size} mean luminance {luminance}, expected {expected:?}"
                    );
                    if size >= 64 {
                        let (min, max) = layer
                            .chunks_exact(4)
                            .fold((255u8, 0u8), |(lo, hi), px| (lo.min(px[1]), hi.max(px[1])));
                        assert!(max - min >= 8, "layer {id} is flat: {min} to {max}");
                    }
                }
            }
        }
    }

    #[test]
    fn textures_are_pure_in_seed_and_size() {
        let a = material_textures(42, 64);
        let b = material_textures(42, 64);
        assert_eq!(a, b);
        let c = material_textures(43, 64);
        assert_ne!(a, c);
        assert_eq!(
            a.layers[0], c.layers[0],
            "the air layer is seed independent"
        );
        let d =
            std::thread::scope(|scope| scope.spawn(|| material_textures(42, 64)).join().unwrap());
        assert_eq!(a, d);
    }

    #[test]
    fn material_textures_timing() {
        let size = 512;
        let start = Instant::now();
        let set = material_textures(1, size);
        let elapsed = start.elapsed();
        println!(
            "material_textures size {size}: {:.1} ms for {} layers, {} bytes",
            elapsed.as_secs_f64() * 1e3,
            set.layer_count(),
            set.bytes()
        );
    }
}
