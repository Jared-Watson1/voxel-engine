//! Seed derivation (D5): distinct noise seeds from the u64 world seed through an integer hash,
//! never through floating point or anything platform dependent.

pub(crate) const STREAM_HEIGHT: u64 = 0x1;
pub(crate) const STREAM_DETAIL: u64 = 0x2;
pub(crate) const STREAM_TEXTURES: u64 = 0x3;

/// splitmix64 finaliser (Steele, Lea and Flood, "Fast splittable pseudorandom number generators",
/// 2014). Wrapping integer arithmetic only.
pub(crate) const fn splitmix64(x: u64) -> u64 {
    let z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A 64 bit seed for `stream` of the world seed.
pub(crate) const fn derive_seed64(world_seed: u64, stream: u64) -> u64 {
    splitmix64(world_seed ^ splitmix64(stream))
}

/// A noise crate seed for `stream` of the world seed, in 0 to 2^30 - 1. The range leaves room for
/// the per octave seed increments inside fastnoise-lite, which must not overflow an i32.
pub(crate) const fn derive_seed(world_seed: u64, stream: u64) -> i32 {
    (derive_seed64(world_seed, stream) >> 34) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_and_seeds_give_distinct_values_in_range() {
        for world_seed in [0, 1, 2, u64::MAX, 0xDEAD_BEEF] {
            let h = derive_seed(world_seed, STREAM_HEIGHT);
            let d = derive_seed(world_seed, STREAM_DETAIL);
            let t = derive_seed(world_seed, STREAM_TEXTURES);
            assert!(h != d && d != t && h != t, "seed {world_seed}");
            for s in [h, d, t] {
                assert!((0..1 << 30).contains(&s));
            }
        }
        assert_ne!(derive_seed(1, STREAM_HEIGHT), derive_seed(2, STREAM_HEIGHT));
        assert_eq!(
            derive_seed64(7, STREAM_TEXTURES),
            derive_seed64(7, STREAM_TEXTURES)
        );
    }
}
