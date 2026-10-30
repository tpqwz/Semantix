//! Decode a gallery image into upright RGB, and write a JPEG thumbnail.

use crate::exif;
use anyhow::{anyhow, Result};
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ExtendedColorType, ImageReader, Limits, Rgb, RgbImage};
use std::io::Cursor;
use std::path::Path;

pub const EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff"];

pub struct Decoded {
    pub rgb: RgbImage,
    pub exif_json: String,
}

pub fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn decode(bytes: &[u8]) -> Result<Decoded> {
    let meta = if looks_like_jpeg(bytes) {
        exif::from_jpeg(bytes)
    } else {
        exif::Meta::default()
    };
    let mut limits = Limits::default();
    limits.max_image_width = Some(16_000);
    limits.max_image_height = Some(16_000);
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|err| anyhow!("image format: {err}"))?;
    reader.limits(limits);
    let image = reader.decode().map_err(|err| anyhow!("decode: {err}"))?;
    let upright = exif::apply_orientation(image, meta.orientation);
    Ok(Decoded {
        rgb: flatten(upright),
        exif_json: meta.json,
    })
}

fn looks_like_jpeg(bytes: &[u8]) -> bool {
    bytes.len() > 3 && bytes[0] == 0xFF && bytes[1] == 0xD8
}

fn flatten(image: DynamicImage) -> RgbImage {
    if !image.has_alpha() {
        return image.to_rgb8();
    }
    let rgba = image.to_rgba8();
    let mut out = RgbImage::new(rgba.width(), rgba.height());
    for (dst, src) in out.pixels_mut().zip(rgba.pixels()) {
        let alpha = src[3] as f32 / 255.0;
        for channel in 0..3 {
            let value = src[channel] as f32 * alpha + 255.0 * (1.0 - alpha);
            dst[channel] = value.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

pub fn fit(image: &RgbImage, max_side: u32) -> RgbImage {
    let (width, height) = image.dimensions();
    let long = width.max(height);
    if long <= max_side || long == 0 {
        return image.clone();
    }
    let scale = max_side as f32 / long as f32;
    let nw = ((width as f32) * scale).round().max(1.0) as u32;
    let nh = ((height as f32) * scale).round().max(1.0) as u32;
    image::imageops::resize(image, nw, nh, image::imageops::FilterType::Triangle)
}

pub fn encode_jpeg(image: &RgbImage, quality: u8) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut buf, quality);
    encoder
        .encode(image.as_raw(), image.width(), image.height(), ExtendedColorType::Rgb8)
        .map_err(|err| anyhow!("jpeg: {err}"))?;
    Ok(buf)
}

pub fn mtime_ns(meta: &std::fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|dur| i64::try_from(dur.as_nanos()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[allow(dead_code)]
pub fn solid(width: u32, height: u32, color: [u8; 3]) -> RgbImage {
    RgbImage::from_pixel(width, height, Rgb(color))
}
