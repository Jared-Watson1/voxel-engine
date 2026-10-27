//! First fit range allocator over byte offsets for the pooled terrain buffers (D10). The free
//! list is sorted by offset and neighbours coalesce on free, so fragmentation is visible as the
//! difference between the free total and the largest free range.

/// A byte range inside a pool. `size` is already rounded up to the pool alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Range {
    pub offset: usize,
    pub size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutOfSpace {
    pub needed: usize,
    pub largest_free: usize,
}

pub(crate) struct RangeAllocator {
    capacity: usize,
    alignment: usize,
    /// Sorted by offset; no two entries touch.
    free: Vec<Range>,
    used: usize,
}

impl RangeAllocator {
    /// `alignment` must be a power of two. Capacity is rounded down to it.
    pub(crate) fn new(capacity: usize, alignment: usize) -> Self {
        debug_assert!(alignment.is_power_of_two());
        let capacity = capacity & !(alignment - 1);
        let free = if capacity > 0 {
            vec![Range {
                offset: 0,
                size: capacity,
            }]
        } else {
            Vec::new()
        };
        Self {
            capacity,
            alignment,
            free,
            used: 0,
        }
    }

    /// Takes the first free range that fits `size` rounded up to the alignment. Every range
    /// boundary is a multiple of the alignment because every size is, so offsets are aligned.
    pub(crate) fn allocate(&mut self, size: usize) -> Result<Range, OutOfSpace> {
        debug_assert!(size > 0);
        let size = (size + self.alignment - 1) & !(self.alignment - 1);
        let Some(i) = self.free.iter().position(|r| r.size >= size) else {
            return Err(OutOfSpace {
                needed: size,
                largest_free: self.largest_free(),
            });
        };
        let range = Range {
            offset: self.free[i].offset,
            size,
        };
        self.free[i].offset += size;
        self.free[i].size -= size;
        if self.free[i].size == 0 {
            self.free.remove(i);
        }
        self.used += size;
        Ok(range)
    }

    /// Returns a range from [`RangeAllocator::allocate`], merging it with free neighbours.
    pub(crate) fn free(&mut self, range: Range) {
        let i = self.free.partition_point(|r| r.offset < range.offset);
        debug_assert!(
            i == 0 || self.free[i - 1].offset + self.free[i - 1].size <= range.offset,
            "double free or overlap at {range:?}"
        );
        debug_assert!(
            i == self.free.len() || range.offset + range.size <= self.free[i].offset,
            "double free or overlap at {range:?}"
        );
        self.free.insert(i, range);
        if i + 1 < self.free.len()
            && self.free[i].offset + self.free[i].size == self.free[i + 1].offset
        {
            self.free[i].size += self.free[i + 1].size;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].offset + self.free[i - 1].size == self.free[i].offset {
            self.free[i - 1].size += self.free[i].size;
            self.free.remove(i);
        }
        self.used -= range.size;
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn used(&self) -> usize {
        self.used
    }

    pub(crate) fn largest_free(&self) -> usize {
        self.free.iter().map(|r| r.size).max().unwrap_or(0)
    }

    #[cfg(test)]
    fn free_ranges(&self) -> usize {
        self.free.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_fit_reuses_the_freed_middle_range() {
        let mut a = RangeAllocator::new(1024, 16);
        let first = a.allocate(100).unwrap();
        let middle = a.allocate(100).unwrap();
        let last = a.allocate(100).unwrap();
        assert_eq!(first.offset, 0);
        assert_eq!(middle.offset, 112, "sizes round up to the alignment");
        assert_eq!(last.offset, 224);
        assert_eq!(a.used(), 336);

        a.free(middle);
        assert_eq!(a.free_ranges(), 2);
        let smaller = a.allocate(32).unwrap();
        assert_eq!(
            smaller.offset, 112,
            "first fit takes the hole before the tail"
        );
        assert_eq!(a.largest_free(), 1024 - 336);
        assert_eq!(a.used(), 336 - 112 + 32);
    }

    #[test]
    fn freeing_neighbours_coalesces_into_one_range() {
        let mut a = RangeAllocator::new(1024, 16);
        let first = a.allocate(256).unwrap();
        let middle = a.allocate(256).unwrap();
        let last = a.allocate(512).unwrap();
        assert_eq!(a.largest_free(), 0);

        a.free(middle);
        a.free(last);
        assert_eq!(a.free_ranges(), 1, "the later neighbour merges");
        assert_eq!(a.largest_free(), 768);
        a.free(first);
        assert_eq!(a.free_ranges(), 1, "the earlier neighbour merges");
        assert_eq!(a.largest_free(), 1024);
        assert_eq!(a.used(), 0);
    }

    #[test]
    fn fragmentation_shows_as_a_small_largest_free() {
        let mut a = RangeAllocator::new(1024, 16);
        let ranges: Vec<Range> = (0..8).map(|_| a.allocate(128).unwrap()).collect();
        for r in ranges.iter().step_by(2) {
            a.free(*r);
        }
        assert_eq!(a.used(), 512);
        assert_eq!(a.largest_free(), 128);
        assert_eq!(a.free_ranges(), 4);
        let err = a.allocate(256).unwrap_err();
        assert_eq!(
            err,
            OutOfSpace {
                needed: 256,
                largest_free: 128
            }
        );
    }

    #[test]
    fn exhaustion_is_an_error_with_the_largest_hole() {
        let mut a = RangeAllocator::new(64, 4);
        a.allocate(60).unwrap();
        let err = a.allocate(8).unwrap_err();
        assert_eq!(err.needed, 8);
        assert_eq!(err.largest_free, 4);
        assert!(a.allocate(4).is_ok());
        assert_eq!(a.allocate(1).unwrap_err().largest_free, 0);
    }

    #[test]
    fn offsets_respect_the_alignment() {
        let mut a = RangeAllocator::new(4096, 16);
        for size in [1, 3, 17, 33, 100, 16] {
            let r = a.allocate(size).unwrap();
            assert_eq!(r.offset % 16, 0, "{size} byte allocation at {}", r.offset);
            assert_eq!(r.size % 16, 0);
            assert!(r.size >= size);
        }
        let mut b = RangeAllocator::new(4096, 4);
        let r = b.allocate(6).unwrap();
        assert_eq!(r.size, 8);
        assert_eq!(b.allocate(1).unwrap().offset, 8);
    }

    #[test]
    fn capacity_rounds_down_and_zero_is_empty() {
        let a = RangeAllocator::new(1000, 16);
        assert_eq!(a.capacity(), 992);
        assert_eq!(a.largest_free(), 992);
        let mut z = RangeAllocator::new(0, 16);
        assert!(z.allocate(16).is_err());
    }
}
