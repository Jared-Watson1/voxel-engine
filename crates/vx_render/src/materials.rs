//! Material albedo texture array for the triplanar terrain shader (D19). Layers arrive as RGBA8
//! sRGB images; the mip chain is generated on the CPU and every level is uploaded, so the GPU
//! never runs a blit pass for materials.

use crate::{MaterialTextures, PreparedMaterialTextures};

/// Size of the built in debug layers.
const DEBUG_SIZE: u32 = 16;
/// Debug colours by layer: magenta for 0 (air, never a surface material), then grey, brown,
/// green and yellow, each with a faint checker so the tile repeat is visible.
const DEBUG_COLOURS: [[u8; 3]; 5] = [
    [255, 0, 255],
    [128, 128, 128],
    [120, 80, 50],
    [70, 130, 50],
    [200, 180, 60],
];

pub(crate) struct MaterialArray {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    bytes: usize,
}

impl MaterialArray {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain materials"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // Anisotropy needs linear filters on all three axes; four samples is enough for terrain
        // viewed at grazing angles without the cost of sixteen on an M1.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain materials"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 4,
            ..Default::default()
        });
        let debug = prepare(&debug_textures()).unwrap_or_else(|e| {
            // The built in pattern has a fixed valid shape; a failure here is a programming error.
            panic!("the built in material textures must prepare: {e}")
        });
        let (view, bytes) = upload(device, queue, &debug, "terrain materials debug");
        let bind_group = make_bind_group(device, &layout, &view, &sampler);
        Self {
            layout,
            bind_group,
            sampler,
            bytes,
        }
    }

    /// Replaces the array. On a set the device cannot hold, the previous array stays bound.
    pub(crate) fn set(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: &PreparedMaterialTextures,
    ) -> Result<(), String> {
        validate_limits(device, textures)?;
        let (view, bytes) = upload(device, queue, textures, "terrain materials");
        self.bind_group = make_bind_group(device, &self.layout, &view, &self.sampler);
        self.bytes = bytes;
        Ok(())
    }

    pub(crate) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Bytes of every level of every layer.
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
}

/// Shape checks and the CPU mip chains, without a device, so the work can run on a worker.
pub(crate) fn prepare(textures: &MaterialTextures) -> Result<PreparedMaterialTextures, String> {
    profiling::scope!("materials prepare");
    validate_shape(textures)?;
    let layers = textures
        .layers
        .iter()
        .map(|base| mip_chain(base, textures.size, textures.size))
        .collect();
    Ok(PreparedMaterialTextures {
        size: textures.size,
        layers,
    })
}

fn validate_limits(
    device: &wgpu::Device,
    textures: &PreparedMaterialTextures,
) -> Result<(), String> {
    let limits = device.limits();
    if textures.size > limits.max_texture_dimension_2d {
        return Err(format!(
            "material texture size {} is above {}",
            textures.size, limits.max_texture_dimension_2d
        ));
    }
    if textures.layers.len() as u32 > limits.max_texture_array_layers {
        return Err(format!(
            "material layer count {} is above {}",
            textures.layers.len(),
            limits.max_texture_array_layers
        ));
    }
    Ok(())
}

fn validate_shape(textures: &MaterialTextures) -> Result<(), String> {
    if textures.size == 0 {
        return Err("material texture size is 0".into());
    }
    if textures.layers.is_empty() {
        return Err("no material layers".into());
    }
    let expected = (textures.size * textures.size * 4) as usize;
    if let Some((i, layer)) = textures
        .layers
        .iter()
        .enumerate()
        .find(|(_, l)| l.len() != expected)
    {
        return Err(format!(
            "material layer {i} has {} bytes, expected {expected} for {size} by {size} RGBA8",
            layer.len(),
            size = textures.size
        ));
    }
    Ok(())
}

fn make_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("terrain materials"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Creates the sRGB array with a full mip chain and writes every level. Returns the view and the
/// total bytes resident on the GPU.
fn upload(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    textures: &PreparedMaterialTextures,
    label: &str,
) -> (wgpu::TextureView, usize) {
    let size = textures.size;
    let levels = mip_level_count(size);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: textures.layers.len() as u32,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut bytes = 0;
    for (layer, chain) in textures.layers.iter().enumerate() {
        for (level, (width, height, pixels)) in chain.iter().enumerate() {
            bytes += pixels.len();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(*height),
                },
                wgpu::Extent3d {
                    width: *width,
                    height: *height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (view, bytes)
}

/// Levels down to 1 by 1: floor(log2(size)) + 1.
fn mip_level_count(size: u32) -> u32 {
    debug_assert!(size > 0);
    u32::BITS - size.leading_zeros()
}

/// Every level from the base image down to 1 by 1 as (width, height, RGBA8 sRGB pixels).
fn mip_chain(base: &[u8], width: u32, height: u32) -> Vec<(u32, u32, Vec<u8>)> {
    let to_linear: [f32; 256] = std::array::from_fn(|i| srgb_to_linear(i as u8));
    let mut chain = vec![(width, height, base.to_vec())];
    while let Some((w, h, src)) = chain.last()
        && (*w > 1 || *h > 1)
    {
        let next = downsample(src, *w, *h, &to_linear);
        chain.push(next);
    }
    chain
}

/// Box filter over 2 by 2 source texels, averaged in linear light so dark and bright texels mix
/// as they would on screen (averaging sRGB codes directly darkens the result). Odd dimensions
/// clamp the second sample to the edge. Alpha is averaged as stored.
fn downsample(src: &[u8], width: u32, height: u32, to_linear: &[f32; 256]) -> (u32, u32, Vec<u8>) {
    let w = (width / 2).max(1);
    let h = (height / 2).max(1);
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let mut rgb = [0.0f32; 3];
            let mut alpha = 0.0f32;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let sx = (2 * x + dx).min(width - 1);
                let sy = (2 * y + dy).min(height - 1);
                let i = ((sy * width + sx) * 4) as usize;
                for c in 0..3 {
                    rgb[c] += to_linear[src[i + c] as usize];
                }
                alpha += f32::from(src[i + 3]);
            }
            out.extend(rgb.map(|v| linear_to_srgb(v / 4.0)));
            out.push((alpha / 4.0).round() as u8);
        }
    }
    (w, h, out)
}

fn srgb_to_linear(code: u8) -> f32 {
    let c = f32::from(code) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(linear: f32) -> u8 {
    let c = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0).round().clamp(0.0, 255.0) as u8
}

fn debug_textures() -> MaterialTextures {
    let layers = DEBUG_COLOURS
        .iter()
        .map(|colour| {
            let mut pixels = Vec::with_capacity((DEBUG_SIZE * DEBUG_SIZE * 4) as usize);
            for y in 0..DEBUG_SIZE {
                for x in 0..DEBUG_SIZE {
                    let dark = ((x / 8) + (y / 8)) % 2 == 1;
                    for &c in colour {
                        pixels.push(if dark { c - c / 8 } else { c });
                    }
                    pixels.push(255);
                }
            }
            pixels
        })
        .collect();
    MaterialTextures {
        size: DEBUG_SIZE,
        layers,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_level_count_covers_down_to_one_texel() {
        assert_eq!(mip_level_count(1), 1);
        assert_eq!(mip_level_count(2), 2);
        assert_eq!(mip_level_count(16), 5);
        assert_eq!(mip_level_count(20), 5);
        assert_eq!(mip_level_count(1024), 11);
    }

    #[test]
    fn mip_chain_halves_until_one_texel_and_averages_in_linear_light() {
        // A 4 by 4 image whose left half is black and right half is white.
        let mut base = Vec::new();
        for _y in 0..4 {
            for x in 0..4 {
                let v = if x < 2 { 0 } else { 255 };
                base.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let chain = mip_chain(&base, 4, 4);
        let dims: Vec<(u32, u32)> = chain.iter().map(|(w, h, _)| (*w, *h)).collect();
        assert_eq!(dims, [(4, 4), (2, 2), (1, 1)]);
        assert_eq!(chain.len() as u32, mip_level_count(4));
        let (_, _, level1) = &chain[1];
        assert_eq!(&level1[..4], &[0, 0, 0, 255], "left texel stays black");
        assert_eq!(
            &level1[4..8],
            &[255, 255, 255, 255],
            "right texel stays white"
        );
        let (_, _, level2) = &chain[2];
        // Half black and half white in linear light is sRGB 188, not 128.
        assert_eq!(level2[0], linear_to_srgb(0.5));
        assert_eq!(level2[0], 188);
        assert_eq!(level2[3], 255);
    }

    #[test]
    fn odd_sizes_clamp_at_the_edge() {
        let base = vec![100u8; 3 * 3 * 4];
        let chain = mip_chain(&base, 3, 3);
        let dims: Vec<(u32, u32)> = chain.iter().map(|(w, h, _)| (*w, *h)).collect();
        assert_eq!(dims, [(3, 3), (1, 1)]);
        assert_eq!(chain[1].2, vec![100, 100, 100, 100]);
    }

    #[test]
    fn srgb_round_trips() {
        for code in [0u8, 1, 10, 100, 128, 200, 254, 255] {
            assert_eq!(linear_to_srgb(srgb_to_linear(code)), code);
        }
    }

    #[test]
    fn debug_layers_are_well_formed() {
        let debug = debug_textures();
        assert_eq!(debug.layers.len(), DEBUG_COLOURS.len());
        for layer in &debug.layers {
            assert_eq!(layer.len(), (DEBUG_SIZE * DEBUG_SIZE * 4) as usize);
        }
        assert_eq!(&debug.layers[0][..4], &[255, 0, 255, 255]);
    }
}
