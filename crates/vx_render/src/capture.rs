use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// An RGBA8 copy of a presented frame (sRGB encoded when the surface is sRGB).
pub struct CapturedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

const MAP_PENDING: u8 = 0;
const MAP_OK: u8 = 1;
const MAP_FAILED: u8 = 2;

enum State {
    Idle,
    Requested,
    Copying {
        buffer: wgpu::Buffer,
        width: u32,
        height: u32,
        padded_bytes_per_row: u32,
    },
    Mapping {
        buffer: wgpu::Buffer,
        width: u32,
        height: u32,
        padded_bytes_per_row: u32,
        mapped: Arc<AtomicU8>,
    },
}

/// Surface readback through a mapped buffer, polled without blocking.
pub(crate) struct Capture {
    supported: bool,
    state: State,
}

impl Capture {
    pub(crate) fn new(supported: bool) -> Self {
        Self {
            supported,
            state: State::Idle,
        }
    }

    pub(crate) fn request(&mut self) -> Result<(), String> {
        if !self.supported {
            return Err("the surface does not support COPY_SRC".into());
        }
        if !matches!(self.state, State::Idle) {
            return Err("a capture is already in flight".into());
        }
        self.state = State::Requested;
        Ok(())
    }

    pub(crate) fn is_pending(&self) -> bool {
        !matches!(self.state, State::Idle)
    }

    pub(crate) fn encode_copy(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) {
        if !matches!(self.state, State::Requested) {
            return;
        }
        let (width, height) = (texture.width(), texture.height());
        let unpadded = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded.div_ceil(align) * align;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture readback"),
            size: u64::from(padded_bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.state = State::Copying {
            buffer,
            width,
            height,
            padded_bytes_per_row,
        };
    }

    pub(crate) fn after_submit(&mut self) {
        if !matches!(self.state, State::Copying { .. }) {
            return;
        }
        let State::Copying {
            buffer,
            width,
            height,
            padded_bytes_per_row,
        } = std::mem::replace(&mut self.state, State::Idle)
        else {
            unreachable!("state matched Copying above");
        };
        let mapped = Arc::new(AtomicU8::new(MAP_PENDING));
        let flag = mapped.clone();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            flag.store(
                if result.is_ok() { MAP_OK } else { MAP_FAILED },
                Ordering::Release,
            );
        });
        self.state = State::Mapping {
            buffer,
            width,
            height,
            padded_bytes_per_row,
            mapped,
        };
    }

    pub(crate) fn try_take(&mut self, format: wgpu::TextureFormat) -> Option<CapturedImage> {
        let State::Mapping { mapped, .. } = &self.state else {
            return None;
        };
        match mapped.load(Ordering::Acquire) {
            MAP_PENDING => return None,
            MAP_FAILED => {
                tracing::error!("capture readback failed to map");
                self.state = State::Idle;
                return None;
            }
            _ => {}
        }
        let State::Mapping {
            buffer,
            width,
            height,
            padded_bytes_per_row,
            ..
        } = std::mem::replace(&mut self.state, State::Idle)
        else {
            unreachable!("state matched Mapping above");
        };
        let bgra = matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        {
            let Ok(view) = buffer.get_mapped_range(..) else {
                tracing::error!("capture readback could not be read");
                buffer.unmap();
                return None;
            };
            for row in view.chunks_exact(padded_bytes_per_row as usize) {
                let row = &row[..(width * 4) as usize];
                if bgra {
                    for px in row.chunks_exact(4) {
                        rgba.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
                    }
                } else {
                    rgba.extend_from_slice(row);
                }
            }
        }
        buffer.unmap();
        Some(CapturedImage {
            width,
            height,
            rgba,
        })
    }
}
