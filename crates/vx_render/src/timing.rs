//! GPU pass timings from timestamp queries (engineering rules 4 and 5).
//!
//! Apple GPUs sample timestamps only at pass boundaries (`Features::TIMESTAMP_QUERY` without
//! `TIMESTAMP_QUERY_INSIDE_ENCODERS` or `_INSIDE_PASSES`), so every timed span starts and ends at
//! the boundary of a pass. Passes owned by other modules that open their own untimed passes are
//! timed from the end of a marker pass to the start of the next timed pass. Metal takes no
//! sample in an empty pass, so the marker dispatches one invocation of a no-op shader.
//!
//! A tile based GPU overlaps consecutive passes: a render pass's first sample is taken when its
//! vertex stage starts, which can be long before the previous pass's fragments finish, so a
//! render pass is reported as its exclusive time, from the later of its own start and the latest
//! end of the passes encoded before it, to its end. Compute passes have no such early sample and
//! are reported from their start to their end even when they run concurrently with an earlier
//! render pass, so their work is never hidden; the sum over passes can then exceed the frame.
//!
//! The queries of a frame are resolved and copied into one of a few readback buffers, mapped
//! after the submission and collected without blocking at the start of a later frame. A frame
//! that finds every readback buffer busy is not timed.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use crate::GpuTimings;

/// Upper bound on timestamps per frame (two per pass plus markers).
const MAX_QUERIES: u32 = 32;
const READBACK_SLOTS: usize = 3;

const MAP_PENDING: u8 = 0;
const MAP_OK: u8 = 1;
const MAP_FAILED: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    Copied,
    Mapping,
}

struct Slot {
    buffer: wgpu::Buffer,
    state: SlotState,
    mapped: Arc<AtomicU8>,
    frame: u64,
    query_count: u32,
    spans: Vec<Span>,
}

#[derive(Debug, Clone, Copy)]
struct Span {
    name: &'static str,
    begin: u32,
    end: u32,
    /// Render pass: measured from the latest earlier end (see the module documentation).
    exclusive: bool,
}

struct Queries {
    set: wgpu::QuerySet,
    marker: wgpu::ComputePipeline,
    resolve: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    period_ns: f64,
}

/// Timestamp queries around passes, read back a few frames late. Inert when the device has no
/// `TIMESTAMP_QUERY`.
pub(crate) struct GpuTimer {
    queries: Option<Queries>,
    slots: Vec<Slot>,
    /// Slot receiving this frame's queries, `None` when the frame is not timed.
    active: Option<usize>,
    next_query: u32,
    spans: Vec<Span>,
    frame: u64,
    latest: Option<(Vec<(&'static str, f64)>, u64)>,
}

impl GpuTimer {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let queries = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| Queries {
                marker: marker_pipeline(device),
                set: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("pass timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: MAX_QUERIES,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("timestamp resolve"),
                    size: u64::from(MAX_QUERIES) * wgpu::QUERY_SIZE as u64,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                period_ns: f64::from(queue.get_timestamp_period()),
            });
        let slots = if queries.is_some() {
            (0..READBACK_SLOTS)
                .map(|_| Slot {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("timestamp readback"),
                        size: u64::from(MAX_QUERIES) * wgpu::QUERY_SIZE as u64,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    }),
                    state: SlotState::Free,
                    mapped: Arc::new(AtomicU8::new(MAP_PENDING)),
                    frame: 0,
                    query_count: 0,
                    spans: Vec::new(),
                })
                .collect()
        } else {
            Vec::new()
        };
        Self {
            queries,
            slots,
            active: None,
            next_query: 0,
            spans: Vec::new(),
            frame: 0,
            latest: None,
        }
    }

    pub(crate) fn supported(&self) -> bool {
        self.queries.is_some()
    }

    /// Collects finished readbacks without blocking and starts a frame.
    pub(crate) fn begin_frame(&mut self, device: &wgpu::Device) {
        self.poll(device);
        self.frame += 1;
        self.next_query = 0;
        self.spans.clear();
        self.active = self.slots.iter().position(|s| s.state == SlotState::Free);
    }

    fn alloc(&mut self, count: u32) -> Option<u32> {
        self.active?;
        if self.next_query + count > MAX_QUERIES {
            return None;
        }
        let first = self.next_query;
        self.next_query += count;
        Some(first)
    }

    /// Timestamp writes for a compute pass timed as `name`.
    pub(crate) fn compute_writes(
        &mut self,
        name: &'static str,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let begin = self.alloc(2)?;
        self.spans.push(Span {
            name,
            begin,
            end: begin + 1,
            exclusive: false,
        });
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.queries.as_ref()?.set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(begin + 1),
        })
    }

    /// Timestamp writes for a render pass timed as `name`.
    pub(crate) fn render_writes(
        &mut self,
        name: &'static str,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let begin = self.alloc(2)?;
        self.spans.push(Span {
            name,
            begin,
            end: begin + 1,
            exclusive: true,
        });
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.queries.as_ref()?.set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(begin + 1),
        })
    }

    /// Encodes a minimal compute pass whose end is a timestamp, and returns its index. With
    /// [`GpuTimer::span_to_next`] it times work encoded by code that opens its own passes.
    pub(crate) fn marker(&mut self, encoder: &mut wgpu::CommandEncoder) -> Option<u32> {
        let index = self.alloc(1)?;
        let queries = self.queries.as_ref()?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("timestamp marker"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &queries.set,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: Some(index),
            }),
        });
        pass.set_pipeline(&queries.marker);
        pass.dispatch_workgroups(1, 1, 1);
        Some(index)
    }

    /// Records a span named `name` from the marker `begin` to the beginning of the next pass
    /// timed through this timer.
    pub(crate) fn span_to_next(&mut self, name: &'static str, begin: Option<u32>) {
        if let Some(begin) = begin
            && self.next_query < MAX_QUERIES
        {
            self.spans.push(Span {
                name,
                begin,
                end: self.next_query,
                exclusive: false,
            });
        }
    }

    /// Resolves this frame's queries into a readback buffer. Call once, after the last timed
    /// pass of the frame.
    pub(crate) fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let (Some(queries), Some(slot)) = (&self.queries, self.active) else {
            return;
        };
        let count = self.next_query;
        // A span ending at a query that was never allocated (no pass followed its marker) has
        // nothing to measure.
        self.spans.retain(|s| s.end < count && s.begin < count);
        if count == 0 || self.spans.is_empty() {
            return;
        }
        encoder.resolve_query_set(&queries.set, 0..count, &queries.resolve, 0);
        let slot = &mut self.slots[slot];
        encoder.copy_buffer_to_buffer(
            &queries.resolve,
            0,
            &slot.buffer,
            0,
            u64::from(count) * wgpu::QUERY_SIZE as u64,
        );
        slot.state = SlotState::Copied;
        slot.frame = self.frame;
        slot.query_count = count;
        slot.spans.clear();
        slot.spans.extend_from_slice(&self.spans);
        self.active = None;
    }

    /// Starts the map of the buffers copied this frame. Call after the submission.
    pub(crate) fn after_submit(&mut self) {
        for slot in self
            .slots
            .iter_mut()
            .filter(|s| s.state == SlotState::Copied)
        {
            slot.mapped.store(MAP_PENDING, Ordering::Release);
            let flag = slot.mapped.clone();
            slot.buffer
                .map_async(wgpu::MapMode::Read, .., move |result| {
                    flag.store(
                        if result.is_ok() { MAP_OK } else { MAP_FAILED },
                        Ordering::Release,
                    );
                });
            slot.state = SlotState::Mapping;
        }
    }

    /// Collects finished maps. Never waits: `PollType::Poll` only fires callbacks of completed
    /// submissions.
    pub(crate) fn poll(&mut self, device: &wgpu::Device) {
        let Some(queries) = &self.queries else {
            return;
        };
        if !self.slots.iter().any(|s| s.state == SlotState::Mapping) {
            return;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        for slot in self
            .slots
            .iter_mut()
            .filter(|s| s.state == SlotState::Mapping)
        {
            match slot.mapped.load(Ordering::Acquire) {
                MAP_PENDING => {}
                MAP_OK => {
                    let ticks: Option<Vec<u64>> = slot.buffer.get_mapped_range(..).ok().map(|v| {
                        bytemuck::cast_slice::<u8, u64>(
                            &v[..slot.query_count as usize * wgpu::QUERY_SIZE as usize],
                        )
                        .to_vec()
                    });
                    slot.buffer.unmap();
                    slot.state = SlotState::Free;
                    if let Some(ticks) = ticks
                        && self.latest.as_ref().is_none_or(|(_, f)| *f < slot.frame)
                    {
                        let passes = span_times(&slot.spans, &ticks, queries.period_ns);
                        self.latest = Some((passes, slot.frame));
                    }
                }
                _ => {
                    tracing::warn!("timestamp readback failed to map");
                    slot.state = SlotState::Free;
                }
            }
        }
    }

    pub(crate) fn timings(&self) -> Option<GpuTimings> {
        let (passes, frame) = self.latest.as_ref()?;
        Some(GpuTimings {
            passes: passes.clone(),
            frames_old: (self.frame - frame) as u32,
        })
    }
}

fn marker_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("timestamp marker"),
        source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("timestamp marker"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Milliseconds per span in encoding order, exclusive for render passes. A span whose end precedes
/// its begin (a reordered or missing sample) is dropped rather than reported as zero.
fn span_times(spans: &[Span], ticks: &[u64], period_ns: f64) -> Vec<(&'static str, f64)> {
    let mut frontier = 0;
    spans
        .iter()
        .filter_map(|s| {
            let begin = *ticks.get(s.begin as usize)?;
            let end = *ticks.get(s.end as usize)?;
            if end < begin || begin == 0 {
                return None;
            }
            let start = if s.exclusive {
                begin.max(frontier).min(end)
            } else {
                begin
            };
            frontier = frontier.max(end);
            Some((s.name, (end - start) as f64 * period_ns * 1.0e-6))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::tests::headless_device;

    /// The cull pass is timed from an empty marker pass to the next timed pass; an empty encoder
    /// must still produce its sample on this backend.
    #[test]
    fn a_marker_times_untimed_work_up_to_the_next_timed_pass() {
        let Some((device, queue)) = headless_device(wgpu::Features::TIMESTAMP_QUERY) else {
            eprintln!("no GPU adapter available, skipping a_marker_times_untimed_work");
            return;
        };
        let mut timer = GpuTimer::new(&device, &queue);
        if !timer.supported() {
            eprintln!("no timestamp queries, skipping a_marker_times_untimed_work");
            return;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("busy"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> out: array<f32>;
                 @compute @workgroup_size(64)
                 fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                     var x = f32(id.x);
                     for (var i = 0u; i < 4000u; i++) { x = sin(x) * 1.0001 + 0.1; }
                     out[id.x] = x;
                 }
                 @compute @workgroup_size(1)
                 fn consume() { out[0] = out[1] + 1.0; }"
                    .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("busy"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("busy out"),
            size: 64 * 1024 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let consume = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("consume"),
            layout: None,
            module: &module,
            entry_point: Some("consume"),
            compilation_options: Default::default(),
            cache: None,
        });
        let consume_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("consume"),
            layout: &consume.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("busy"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        for _ in 0..3 {
            timer.begin_frame(&device);
            let mut encoder = device.create_command_encoder(&Default::default());
            let start = timer.marker(&mut encoder);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1024, 1, 1);
            }
            timer.span_to_next("cull", start);
            // Like the terrain pass, which draws from the cull pass's indirect buffer, the next
            // timed pass depends on the marked work.
            let mut next = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("next"),
                timestamp_writes: timer.compute_writes("next"),
            });
            next.set_pipeline(&consume);
            next.set_bind_group(0, &consume_group, &[]);
            next.dispatch_workgroups(1, 1, 1);
            drop(next);
            timer.end_frame(&mut encoder);
            queue.submit([encoder.finish()]);
            timer.after_submit();
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll");
        }
        timer.poll(&device);
        let timings = timer.timings().expect("timings after three frames");
        let cull = timings
            .pass_ms("cull")
            .expect("the marker span is reported");
        eprintln!("marker span over untimed work: {cull:.3} ms ({timings:?})");
        let next = timings.pass_ms("next").expect("the next pass is reported");
        assert!(
            cull > 5.0 * next,
            "the marker span holds the untimed work: {cull} vs {next}"
        );
    }

    #[test]
    fn render_spans_are_exclusive_compute_spans_raw_and_reversed_spans_dropped() {
        let span = |name, begin, end, exclusive| Span {
            name,
            begin,
            end,
            exclusive,
        };
        let spans = [
            span("gbuffer", 0, 1, true),
            span("reversed", 2, 3, false),
            span("lut", 4, 5, false),
            span("lighting", 6, 7, true),
        ];
        // Milliseconds with a period of 1e6 ns per tick. The LUT runs during the G-buffer pass and
        // keeps its full duration; the lighting pass starts its vertex stage before the G-buffer
        // ends and is counted from that end.
        let ticks = [1, 4, 9, 8, 2, 3, 3, 6];
        let times = span_times(&spans, &ticks, 1.0e6);
        let names: Vec<_> = times.iter().map(|(n, _)| *n).collect();
        assert_eq!(names, ["gbuffer", "lut", "lighting"]);
        assert!((times[0].1 - 3.0).abs() < 1e-9);
        assert!((times[1].1 - 1.0).abs() < 1e-9);
        assert!((times[2].1 - 2.0).abs() < 1e-9, "{times:?}");
    }
}
