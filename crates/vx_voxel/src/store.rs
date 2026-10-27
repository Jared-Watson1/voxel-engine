use crate::chunk::{Chunk, SAMPLES_PER_AXIS, Voxel};

/// Words in the presence bitmap over the u16 palette key space.
const KEY_BITMAP_WORDS: usize = (u16::MAX as usize + 1) / u64::BITS as usize;

/// Compressed resident form of a chunk (D17).
///
/// `Uniform` holds one value for a block whose samples are all equal (sky, deep rock). `Palette`
/// holds the distinct `(distance, material)` pairs of the block and one bit packed index per
/// sample, using the smallest width that addresses the palette. Both are lossless.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkStore {
    Uniform(Voxel),
    Palette(PaletteBlock),
}

/// Palette compressed samples. Fields are private so the layout can change without touching
/// callers; [`ChunkStore::get`] and [`ChunkStore::decompress_into`] are the read paths.
///
/// Layout: `palette` is sorted ascending by [`Voxel::to_u16`], so two equal blocks pack to equal
/// stores. Sample `i` (in [`Chunk::samples`] order) is the `bits_per_sample` wide index starting
/// at bit `i * bits_per_sample` of the little endian bit stream formed by `words`, least
/// significant bit first; an index may straddle two words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaletteBlock {
    palette: Vec<Voxel>,
    bits_per_sample: u8,
    words: Box<[u64]>,
}

impl ChunkStore {
    /// Compresses a dense chunk. Never loses information: `decompress_into` restores every sample.
    pub fn compress(chunk: &Chunk) -> Self {
        profiling::scope!("ChunkStore::compress");
        let samples = chunk.samples();
        let present = KeyBitmap::of(samples);
        if present.distinct == 1 {
            Self::Uniform(samples[0])
        } else {
            Self::Palette(PaletteBlock::pack(samples, &present))
        }
    }

    /// Writes every sample into `chunk`, which keeps its key and voxel size.
    pub fn decompress_into(&self, chunk: &mut Chunk) {
        profiling::scope!("ChunkStore::decompress_into");
        match self {
            Self::Uniform(v) => chunk.fill_uniform(*v),
            Self::Palette(block) => block.unpack_into(chunk.samples_mut()),
        }
    }

    /// Sample at local coordinates -1 to `CHUNK_SIZE + 1`, decoded in constant time.
    pub fn get(&self, x: isize, y: isize, z: isize) -> Voxel {
        match self {
            Self::Uniform(v) => *v,
            Self::Palette(block) => {
                let i = Chunk::linear(stored(x), stored(y), stored(z));
                block.palette[block.index_at(i)]
            }
        }
    }

    pub fn uniform(&self) -> Option<Voxel> {
        match self {
            Self::Uniform(v) => Some(*v),
            Self::Palette(_) => None,
        }
    }

    /// True when the block holds both inside and outside samples, so meshing can produce
    /// triangles. A uniform block never has a surface.
    pub fn has_surface(&self) -> bool {
        match self {
            Self::Uniform(_) => false,
            Self::Palette(block) => {
                block.palette.iter().any(|v| v.is_inside())
                    && block.palette.iter().any(|v| !v.is_inside())
            }
        }
    }

    /// Bytes this store occupies in memory, inline and on the heap, for the resident voxel budget.
    pub fn resident_bytes(&self) -> usize {
        let heap = match self {
            Self::Uniform(_) => 0,
            Self::Palette(block) => {
                block.palette.capacity() * size_of::<Voxel>() + block.words.len() * size_of::<u64>()
            }
        };
        size_of::<Self>() + heap
    }
}

impl PaletteBlock {
    /// Builds the palette and packs one index per sample.
    ///
    /// The presence bitmap replaces sorting and a 64 Ki entry lookup table: scanning it in order
    /// yields the palette already sorted, and its rank query gives each sample's index in constant
    /// time.
    fn pack(samples: &[Voxel], present: &KeyBitmap) -> Self {
        let distinct = present.distinct;
        let mut palette = Vec::with_capacity(distinct);
        for (w, bits) in present.words.iter().enumerate() {
            let mut remaining = *bits;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                palette.push(Voxel::from_u16(((w << 6) | bit) as u16));
                remaining &= remaining - 1;
            }
        }

        let bits_per_sample = bits_for_palette(distinct);
        let bits = u32::from(bits_per_sample);
        let word_count = (samples.len() * bits as usize).div_ceil(u64::BITS as usize);
        let mut words = Vec::with_capacity(word_count);
        let mut acc = 0u64;
        let mut acc_bits = 0u32;
        for v in samples {
            let index = present.rank(v.to_u16()) as u64;
            // acc_bits < 64 here, so the shift is in range; bits above the word fall off and are
            // re-emitted into the next word below.
            acc |= index << acc_bits;
            acc_bits += bits;
            if acc_bits >= u64::BITS {
                words.push(acc);
                acc_bits -= u64::BITS;
                acc = if acc_bits > 0 {
                    index >> (bits - acc_bits)
                } else {
                    0
                };
            }
        }
        if acc_bits > 0 {
            words.push(acc);
        }
        debug_assert_eq!(words.len(), word_count);

        Self {
            palette,
            bits_per_sample,
            words: words.into_boxed_slice(),
        }
    }

    /// Palette index of sample `i` in [`Chunk::samples`] order.
    fn index_at(&self, i: usize) -> usize {
        let bits = u32::from(self.bits_per_sample);
        let offset = i * bits as usize;
        let w = offset / u64::BITS as usize;
        let shift = (offset % u64::BITS as usize) as u32;
        let mut v = self.words[w] >> shift;
        if shift + bits > u64::BITS {
            v |= self.words[w + 1] << (u64::BITS - shift);
        }
        (v & ((1u64 << bits) - 1)) as usize
    }

    /// Decodes every sample in order, consuming the bit stream word by word.
    fn unpack_into(&self, out: &mut [Voxel]) {
        let bits = u32::from(self.bits_per_sample);
        let mask = (1u64 << bits) - 1;
        let mut words = self.words.iter().copied();
        let mut acc = words.next().unwrap_or(0);
        let mut acc_bits = u64::BITS;
        for slot in out.iter_mut() {
            let mut index = acc & mask;
            if acc_bits >= bits {
                acc >>= bits;
                acc_bits -= bits;
            } else {
                let next = words.next().unwrap_or(0);
                index |= (next << acc_bits) & mask;
                let taken = bits - acc_bits;
                acc = next >> taken;
                acc_bits = u64::BITS - taken;
            }
            *slot = self.palette[index as usize];
        }
    }
}

/// One bit per possible [`Voxel::to_u16`] key, with a prefix popcount per word so the rank of any
/// present key (its index in the sorted palette) costs one popcount.
struct KeyBitmap {
    words: [u64; KEY_BITMAP_WORDS],
    /// Set bits below each word. Fits u16 because a block has fewer than u16::MAX samples.
    rank_base: [u16; KEY_BITMAP_WORDS],
    distinct: usize,
}

impl KeyBitmap {
    fn of(samples: &[Voxel]) -> Self {
        let mut words = [0u64; KEY_BITMAP_WORDS];
        for v in samples {
            let k = usize::from(v.to_u16());
            words[k >> 6] |= 1 << (k & 63);
        }
        let mut rank_base = [0u16; KEY_BITMAP_WORDS];
        let mut distinct = 0usize;
        for (w, bits) in words.iter().enumerate() {
            rank_base[w] = distinct as u16;
            distinct += bits.count_ones() as usize;
        }
        Self {
            words,
            rank_base,
            distinct,
        }
    }

    /// Index of a present key among the set keys in ascending order.
    fn rank(&self, key: u16) -> usize {
        let k = usize::from(key);
        let w = k >> 6;
        let below = self.words[w] & ((1u64 << (k & 63)) - 1);
        usize::from(self.rank_base[w]) + below.count_ones() as usize
    }
}

/// Smallest width that addresses `len` palette entries: `ceil(log2(len))`, at least 1 so a two
/// entry palette still tells its entries apart.
fn bits_for_palette(len: usize) -> u8 {
    let needed = usize::BITS - len.saturating_sub(1).leading_zeros();
    needed.max(1) as u8
}

/// Stored coordinate of a local coordinate in -1 to `CHUNK_SIZE + 1`.
fn stored(local: isize) -> usize {
    let i = local + 1;
    debug_assert!(
        (0..SAMPLES_PER_AXIS as isize).contains(&i),
        "sample {local} outside apron"
    );
    i as usize
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use glam::DVec3;
    use vx_core::WorldConfig;

    use super::*;
    use crate::brush::{Brush, BrushOp, BrushShape};
    use crate::chunk::Material;
    use crate::key::ChunkKey;
    use vx_core::WorldPos;

    const SAMPLE_COUNT: usize = SAMPLES_PER_AXIS * SAMPLES_PER_AXIS * SAMPLES_PER_AXIS;
    const RAW_BYTES: usize = SAMPLE_COUNT * size_of::<Voxel>();

    fn config() -> WorldConfig {
        WorldConfig::default()
    }

    fn empty_chunk() -> Chunk {
        Chunk::for_key(ChunkKey::new(0, 0, 0), &config())
    }

    fn uniform_air() -> Chunk {
        empty_chunk()
    }

    fn uniform_rock() -> Chunk {
        let mut chunk = empty_chunk();
        chunk.fill_uniform(Voxel::solid(Material::ROCK));
        chunk
    }

    /// A sphere well inside the chunk at the design default voxel size.
    fn sphere_chunk() -> Chunk {
        let mut chunk = empty_chunk();
        let extent = config().chunk_extent_m();
        let centre = DVec3::splat(0.5 * extent);
        let radius_m = 0.3 * extent;
        chunk.fill_from(|p| p.distance(centre) - radius_m);
        chunk
    }

    /// Ground at a third of the chunk height: rock in the lower half of the solid, soil above.
    fn two_material_chunk() -> Chunk {
        let mut chunk = empty_chunk();
        let extent = config().chunk_extent_m();
        let ground_m = extent / 3.0;
        chunk.fill_from_with_material(|p| {
            let material = if p.y < ground_m * 0.5 {
                Material::ROCK
            } else {
                Material::SOIL
            };
            (p.y - ground_m, material)
        });
        chunk
    }

    /// Every sample drawn from a seeded xorshift: any i8 distance (zero included) and any material
    /// id up to 15, so every value the packed vertex can carry appears.
    fn random_chunk() -> Chunk {
        let mut chunk = empty_chunk();
        let mut state = 0x9E37_79B9u32;
        for slot in chunk.samples_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *slot = Voxel::from_u16((state & 0x0FFF) as u16);
        }
        chunk.set(0, 0, 0, Voxel::from_u16(0x0F00));
        chunk.set(-1, -1, -1, Voxel::from_u16(0x0080));
        chunk.set(33, 33, 33, Voxel::from_u16(0x0000));
        chunk
    }

    /// Positive everywhere but not constant: ground just below the apron.
    fn all_air_varying() -> Chunk {
        let mut chunk = empty_chunk();
        let vs = chunk.voxel_size_m();
        chunk.fill_from(|p| p.y + 1.5 * vs);
        chunk
    }

    /// Exactly `n` distinct values: air plus `n - 1` rock samples at different distances.
    fn chunk_with_distinct_values(n: usize) -> Chunk {
        assert!(n >= 2 && n - 1 <= 256);
        let mut chunk = empty_chunk();
        for (i, slot) in chunk.samples_mut().iter_mut().take(n - 1).enumerate() {
            *slot = Voxel::from_u16(0x0100 | i as u16);
        }
        chunk
    }

    fn assert_round_trip(name: &str, chunk: &Chunk) -> ChunkStore {
        let store = ChunkStore::compress(chunk);
        let mut out = Chunk::new(chunk.key(), chunk.voxel_size_m());
        out.fill_uniform(Voxel::from_u16(0x0AAA));
        store.decompress_into(&mut out);
        assert_eq!(out.key(), chunk.key());
        assert_eq!(out.voxel_size_m(), chunk.voxel_size_m());
        assert!(out.samples() == chunk.samples(), "{name}: samples differ");
        let last = SAMPLES_PER_AXIS as isize - 2;
        for z in -1..=last {
            for y in -1..=last {
                for x in -1..=last {
                    assert_eq!(
                        store.get(x, y, z),
                        chunk.get(x, y, z),
                        "{name}: sample ({x}, {y}, {z})"
                    );
                }
            }
        }
        assert_eq!(
            store,
            ChunkStore::compress(&out),
            "{name}: recompress differs"
        );
        store
    }

    #[test]
    fn round_trip_is_lossless() {
        assert_round_trip("uniform air", &uniform_air());
        assert_round_trip("uniform rock", &uniform_rock());
        assert_round_trip("sphere", &sphere_chunk());
        assert_round_trip("two materials", &two_material_chunk());
        assert_round_trip("random", &random_chunk());
        assert_round_trip("all air varying", &all_air_varying());
    }

    #[test]
    fn uniform_chunks_collapse_to_one_value() {
        assert_eq!(
            ChunkStore::compress(&uniform_air()).uniform(),
            Some(Voxel::AIR)
        );
        assert_eq!(
            ChunkStore::compress(&uniform_rock()).uniform(),
            Some(Voxel::solid(Material::ROCK))
        );
        assert!(ChunkStore::compress(&uniform_air()).resident_bytes() < 64);
        assert!(ChunkStore::compress(&sphere_chunk()).uniform().is_none());
    }

    #[test]
    fn surface_chunk_is_smaller_than_raw() {
        let store = ChunkStore::compress(&sphere_chunk());
        let bytes = store.resident_bytes();
        println!("sphere chunk at 1.0 m: {bytes} resident bytes (raw {RAW_BYTES})");
        assert!(bytes < RAW_BYTES, "{bytes} >= {RAW_BYTES}");

        let random = ChunkStore::compress(&random_chunk());
        let ChunkStore::Palette(block) = &random else {
            panic!("random chunk compressed to uniform");
        };
        let palette_bytes = block.palette.len() * size_of::<Voxel>();
        let bytes = random.resident_bytes();
        println!(
            "random chunk: {bytes} resident bytes, {} palette entries, {} bits per sample",
            block.palette.len(),
            block.bits_per_sample
        );
        assert!(bytes <= RAW_BYTES + palette_bytes + size_of::<ChunkStore>());
    }

    #[test]
    fn palette_width_grows_with_distinct_values() {
        for (distinct, bits) in [(2, 1), (3, 2), (4, 2), (5, 3), (256, 8), (257, 9)] {
            let chunk = chunk_with_distinct_values(distinct);
            let ChunkStore::Palette(block) = ChunkStore::compress(&chunk) else {
                panic!("{distinct} distinct values compressed to uniform");
            };
            assert_eq!(block.palette.len(), distinct);
            assert_eq!(block.bits_per_sample, bits, "{distinct} distinct values");
            assert_eq!(
                block.words.len(),
                (SAMPLE_COUNT * bits as usize).div_ceil(64),
                "{distinct} distinct values"
            );
            assert_round_trip("distinct", &chunk);
        }
        assert_eq!(bits_for_palette(1), 1);
        assert_eq!(bits_for_palette(65_536), 16);
    }

    #[test]
    fn palette_is_sorted_and_deterministic() {
        let a = ChunkStore::compress(&random_chunk());
        let b = ChunkStore::compress(&random_chunk());
        assert_eq!(a, b);
        let ChunkStore::Palette(block) = a else {
            panic!("random chunk compressed to uniform");
        };
        assert!(
            block
                .palette
                .windows(2)
                .all(|w| w[0].to_u16() < w[1].to_u16())
        );
        assert_eq!(block.palette.capacity(), block.palette.len());
    }

    #[test]
    fn surface_detection() {
        assert!(!ChunkStore::compress(&uniform_air()).has_surface());
        assert!(!ChunkStore::compress(&uniform_rock()).has_surface());
        assert!(!ChunkStore::compress(&all_air_varying()).has_surface());
        assert!(ChunkStore::compress(&sphere_chunk()).has_surface());
        assert!(ChunkStore::compress(&two_material_chunk()).has_surface());
        for chunk in [
            uniform_air(),
            sphere_chunk(),
            all_air_varying(),
            random_chunk(),
        ] {
            assert_eq!(
                ChunkStore::compress(&chunk).has_surface(),
                chunk.has_surface()
            );
        }
    }

    #[test]
    fn resident_bytes_counts_inline_and_heap() {
        let store = ChunkStore::compress(&sphere_chunk());
        let ChunkStore::Palette(block) = &store else {
            panic!("sphere chunk compressed to uniform");
        };
        let expected = size_of::<ChunkStore>()
            + block.palette.len() * size_of::<Voxel>()
            + block.words.len() * size_of::<u64>();
        assert_eq!(store.resident_bytes(), expected);
        assert_eq!(
            ChunkStore::compress(&uniform_air()).resident_bytes(),
            size_of::<ChunkStore>()
        );
    }

    #[test]
    fn memory_report() {
        for (name, chunk) in [
            ("uniform air", uniform_air()),
            ("sphere", sphere_chunk()),
            ("two materials", two_material_chunk()),
            ("random", random_chunk()),
        ] {
            let store = ChunkStore::compress(&chunk);
            let detail = match &store {
                ChunkStore::Uniform(_) => "uniform".to_owned(),
                ChunkStore::Palette(block) => format!(
                    "{} palette entries, {} bits per sample",
                    block.palette.len(),
                    block.bits_per_sample
                ),
            };
            println!(
                "{name} chunk at {} m voxels: {} resident bytes of {RAW_BYTES} raw ({detail})",
                chunk.voxel_size_m(),
                store.resident_bytes()
            );
        }
    }

    fn median(mut samples: Vec<Duration>) -> Duration {
        samples.sort();
        samples[samples.len() / 2]
    }

    #[test]
    fn timing_report() {
        const RUNS: usize = 100;
        let chunk = sphere_chunk();
        let store = ChunkStore::compress(&chunk);

        let mut compress = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = Instant::now();
            std::hint::black_box(ChunkStore::compress(std::hint::black_box(&chunk)));
            compress.push(start.elapsed());
        }

        let mut out = Chunk::new(chunk.key(), chunk.voxel_size_m());
        let mut decompress = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let start = Instant::now();
            std::hint::black_box(&store).decompress_into(std::hint::black_box(&mut out));
            decompress.push(start.elapsed());
        }
        assert!(out.samples() == chunk.samples());

        let brush = Brush {
            shape: BrushShape::Sphere {
                centre: WorldPos::body_fixed(DVec3::splat(0.5 * config().chunk_extent_m())),
                radius_m: 4.0,
            },
            op: BrushOp::Dig,
        };
        let mut apply = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let mut target = chunk.clone();
            let start = Instant::now();
            std::hint::black_box(brush.apply(std::hint::black_box(&mut target)));
            apply.push(start.elapsed());
        }

        println!(
            "TIMING sphere chunk, {} m voxels, median of {RUNS} runs, one thread: compress {:?}, \
             decompress_into {:?}, Brush::apply sphere radius 4 m {:?}",
            chunk.voxel_size_m(),
            median(compress),
            median(decompress),
            median(apply)
        );
    }
}
