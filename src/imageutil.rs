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

/// Stored thumbnail when the indexer has written one. Otherwise a small
/// decode of the original, so a scanned-but-not-indexed photo still previews.
pub fn cell_preview(thumb: Option<&Path>, original: &Path, max_side: u32) -> Option<RgbImage> {
    if let Some(rgb) = thumb.and_then(read_rgb) {
        return Some(fit(&rgb, max_side));
    }
    if !is_image(original) {
        return None;
    }
    let len = std::fs::metadata(original).ok()?.len();
    if len == 0 || len > 32 * 1024 * 1024 {
        return None;
    }
    match read_photo(original, 8) {
        PhotoRead::Bytes(bytes) => decode(&bytes).ok().map(|decoded| fit(&decoded.rgb, max_side)),
        PhotoRead::Unread | PhotoRead::RemoteFailed => None,
    }
}

pub(crate) enum PhotoRead {
    Bytes(Vec<u8>),
    /// The filesystem did not answer in time.
    Unread,
    /// The pCloud catalog has this file and the download did not return it.
    RemoteFailed,
}

/// Bytes for a photo. A pCloud file is downloaded over HTTPS. Anything else
/// is read from disk, and a stuck read is abandoned after `secs`.
pub(crate) fn read_photo(path: &Path, secs: u64) -> PhotoRead {
    match crate::pcloud::fetch(path) {
        crate::pcloud::Fetch::Bytes(bytes) => return PhotoRead::Bytes(bytes),
        crate::pcloud::Fetch::Failed => return PhotoRead::RemoteFailed,
        crate::pcloud::Fetch::NotCloud => {}
    }
    match read_within(path, secs) {
        Some(bytes) => PhotoRead::Bytes(bytes),
        None => PhotoRead::Unread,
    }
}

/// A local read that must not sit forever. pCloud files do not come through here.
pub(crate) fn read_within(path: &Path, secs: u64) -> Option<Vec<u8>> {
    if gnu_timeout() {
        return read_with_gnu_timeout(path, secs);
    }
    // Windows `timeout` is a different program, and macOS does not ship GNU
    // timeout. Abandon the read from this thread instead of failing the open.
    let path = path.to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let bytes = std::fs::read(&path).ok().filter(|bytes| !bytes.is_empty());
        let _ = tx.send(bytes);
    });
    rx.recv_timeout(std::time::Duration::from_secs(secs)).ok().flatten()
}

fn gnu_timeout() -> bool {
    static READY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *READY.get_or_init(|| {
        #[cfg(windows)]
        {
            false
        }
        #[cfg(not(windows))]
        {
            std::process::Command::new("timeout")
                .args(["--signal=KILL", "1", "true"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|status| status.success())
                .unwrap_or(false)
        }
    })
}

fn read_with_gnu_timeout(path: &Path, secs: u64) -> Option<Vec<u8>> {
    let output = std::process::Command::new("timeout")
        .args(["--signal=KILL", &secs.to_string(), "cat", "--"])
        .arg(path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() || output.stdout.is_empty() {
        return None;
    }
    Some(output.stdout)
}

fn read_rgb(path: &Path) -> Option<RgbImage> {
    let bytes = std::fs::read(path).ok()?;
    decode(&bytes).ok().map(|decoded| decoded.rgb)
}

/// A thumbnail already written under the data directory. Does not touch the original.
pub fn stored_thumb(path: &Path, max_side: u32) -> Option<RgbImage> {
    read_rgb(path).map(|rgb| fit(&rgb, max_side))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_preview_uses_the_original_when_the_thumb_is_missing() {
        let dir = std::env::temp_dir().join(format!("semantix-preview-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let original = dir.join("shot.png");
        solid(4, 4, [10, 20, 30]).save(&original).unwrap();
        let preview = cell_preview(Some(&dir.join("missing.jpg")), &original, 64).expect("preview");
        assert_eq!(preview.dimensions(), (4, 4));
        assert_eq!(preview.get_pixel(0, 0).0, [10, 20, 30]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stuck_file_does_not_hold_the_preview() {
        let dir = std::env::temp_dir().join(format!("semantix-stuck-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let fifo = dir.join("stuck.png");
        assert!(std::process::Command::new("mkfifo").arg(&fifo).status().unwrap().success());
        let started = std::time::Instant::now();
        assert!(read_within(&fifo, 1).is_none());
        assert!(started.elapsed() < std::time::Duration::from_secs(4));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
