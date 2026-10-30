//! Sample a few frames from a video with the system `ffmpeg`, then index those
//! JPEGs like any other picture. The app does not link ffmpeg.

use crate::db::ScanFile;
use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const EXTENSIONS: &[&str] = &["mp4", "mkv", "webm", "mov", "m4v", "avi"];
pub const FFMPEG_MISSING: &str = "ffmpeg is not usable";

pub fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Times, in seconds, spread through the clip. At most `max_frames` samples.
pub fn sample_times(duration_secs: f64, max_frames: usize) -> Vec<f64> {
    let max_frames = max_frames.max(1);
    if !duration_secs.is_finite() || duration_secs <= 1.0 {
        let at = if duration_secs.is_finite() && duration_secs > 0.0 {
            (duration_secs * 0.5).clamp(0.0, duration_secs)
        } else {
            0.5
        };
        return vec![at];
    }
    let count = max_frames
        .min((duration_secs / 8.0).ceil() as usize)
        .clamp(1, max_frames);
    if count == 1 {
        return vec![duration_secs * 0.5];
    }
    let start = duration_secs * 0.1;
    let end = duration_secs * 0.9;
    (0..count)
        .map(|index| {
            let t = start + (end - start) * (index as f64) / (count as f64 - 1.0);
            t.clamp(0.05, duration_secs - 0.05)
        })
        .collect()
}

/// JPEG frames for one video. Reuses a previous sample when the file is unchanged.
pub fn cached_frames(
    video: &Path,
    library_root: &Path,
    frames_root: &Path,
    size: u64,
    mtime_ns: i64,
) -> Result<Vec<ScanFile>> {
    let key = blake3::hash(format!("{}\n{size}\n{mtime_ns}", video.display()).as_bytes());
    let dir = frames_root.join(&key.to_hex().to_string()[..16]);
    let manifest = dir.join("manifest.txt");
    if let Some(files) = read_manifest(&manifest, &dir) {
        return Ok(files);
    }
    if !ffmpeg_available() {
        return Err(anyhow!(FFMPEG_MISSING));
    }
    std::fs::create_dir_all(&dir)?;
    let duration = probe_duration(video).unwrap_or(0.0);
    let times = sample_times(duration, 8);
    let rel_video = video.strip_prefix(library_root).unwrap_or(video);
    let stem = stem_token(video);
    let mut files = Vec::new();
    let mut lines = Vec::new();
    for (index, seconds) in times.iter().copied().enumerate() {
        let name = format!("{stem}-{index:02}-{}.jpg", (seconds * 10.0).round() as i64);
        let dest = dir.join(&name);
        if grab(video, seconds, &dest).is_err() {
            continue;
        }
        let rel = format!("{} @ {seconds:.1}s", rel_video.display());
        let rel_line = rel.replace(['\t', '\n', '\r'], " ");
        write_sidecar(&dest, video, seconds)?;
        let meta = dest.metadata()?;
        files.push(ScanFile {
            path: dest.to_string_lossy().into_owned(),
            rel,
            size: i64::try_from(meta.len()).unwrap_or(0),
            mtime_ns,
        });
        lines.push(format!("{name}\t{seconds:.3}\t{rel_line}\t{mtime_ns}"));
    }
    if files.is_empty() {
        return Err(anyhow!("no frames decoded"));
    }
    let mut manifest_file = File::create(&manifest)?;
    for line in &lines {
        writeln!(manifest_file, "{line}")?;
    }
    Ok(files)
}

pub fn source_note(image: &Path) -> Option<String> {
    let text = std::fs::read_to_string(sidecar_path(image)).ok()?;
    let mut lines = text.lines();
    let source = lines.next()?.trim();
    if source.is_empty() {
        return None;
    }
    let seconds = lines.next().unwrap_or("").trim();
    let source = serde_json::to_string(source).ok()?;
    Some(format!(
        "{{\"source\":{source},\"time\":\"{seconds}s\",\"orientation\":1}}"
    ))
}

pub fn remove_cached_frame(data_dir: &Path, path: &Path) {
    let frames = data_dir.join("frames");
    if !path.starts_with(&frames) {
        return;
    }
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(sidecar_path(path));
    if let Some(parent) = path.parent() {
        if parent.starts_with(&frames) {
            let _ = std::fs::remove_dir_all(parent);
        }
    }
}

fn read_manifest(manifest: &Path, dir: &Path) -> Option<Vec<ScanFile>> {
    let file = File::open(manifest).ok()?;
    let mut files = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.ok()?;
        let mut parts = line.split('\t');
        let name = parts.next()?.trim();
        let seconds: f64 = parts.next()?.trim().parse().ok()?;
        if name.is_empty() || name.contains("..") || name.contains('/') || name.contains('\\') {
            return None;
        }
        let dest = dir.join(name);
        if !dest.is_file() {
            return None;
        }
        let meta = dest.metadata().ok()?;
        let rel = parts
            .next()
            .map(str::trim)
            .filter(|rel| !rel.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("{name} @ {seconds:.1}s"));
        let mtime_ns = parts
            .next()
            .and_then(|text| text.trim().parse::<i64>().ok())
            .unwrap_or_else(|| crate::imageutil::mtime_ns(&meta));
        files.push(ScanFile {
            path: dest.to_string_lossy().into_owned(),
            rel,
            size: i64::try_from(meta.len()).unwrap_or(0),
            mtime_ns,
        });
    }
    if files.is_empty() {
        None
    } else {
        Some(files)
    }
}

fn probe_duration(video: &Path) -> Option<f64> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(video)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let duration: f64 = text.trim().parse().ok()?;
    duration.is_finite().then_some(duration).filter(|value| *value > 0.0)
}

fn grab(video: &Path, seconds: f64, dest: &Path) -> Result<()> {
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-ss")
        .arg(format!("{seconds:.3}"))
        .arg("-i")
        .arg(video)
        .args(["-frames:v", "1", "-q:v", "3"])
        .arg(dest)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|err| anyhow!("ffmpeg: {err}"))?;
    if !status.success() || !dest.is_file() {
        return Err(anyhow!("ffmpeg wrote no frame at {seconds:.1}s"));
    }
    Ok(())
}

fn write_sidecar(image: &Path, video: &Path, seconds: f64) -> Result<()> {
    std::fs::write(sidecar_path(image), format!("{}\n{seconds:.3}\n", video.display()))?;
    Ok(())
}

fn sidecar_path(image: &Path) -> PathBuf {
    let mut name = image.file_name().unwrap_or_default().to_os_string();
    name.push(".src");
    image.with_file_name(name)
}

fn stem_token(path: &Path) -> String {
    let stem = path.file_stem().and_then(|value| value.to_str()).unwrap_or("video");
    let mut out = String::new();
    for ch in stem.chars().take(40) {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("video");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_times_stay_inside_the_clip() {
        let times = sample_times(40.0, 8);
        assert!(times.len() >= 2 && times.len() <= 8, "{times:?}");
        assert!(times.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(times.iter().all(|time| *time > 0.0 && *time < 40.0));
        assert_eq!(sample_times(0.2, 8).len(), 1);
        assert_eq!(sample_times(100.0, 8).len(), 8);
    }

    #[test]
    fn ffmpeg_samples_a_short_clip() {
        if !ffmpeg_available() {
            eprintln!("skipping video sample: ffmpeg is not on PATH");
            return;
        }
        let dir = crate::db::scratch_dir("video");
        let video = dir.join("clip.mp4");
        let made = Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x64:d=1",
                "-frames:v",
                "8",
            ])
            .arg(&video)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if !made.map(|status| status.success()).unwrap_or(false) || !video.is_file() {
            eprintln!("skipping video sample: ffmpeg could not encode a test clip");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        let frames = cached_frames(&video, &dir, &dir.join("frames"), 10, 10).unwrap();
        assert!(!frames.is_empty());
        let image = PathBuf::from(&frames[0].path);
        assert!(image.is_file());
        let note = source_note(&image).unwrap();
        assert!(note.contains("clip.mp4"), "{note}");
        assert!(frames[0].rel.contains("clip.mp4"), "{}", frames[0].rel);
        let again = cached_frames(&video, &dir, &dir.join("frames"), 10, 10).unwrap();
        assert_eq!(again.len(), frames.len());
        assert_eq!(again[0].rel, frames[0].rel);
        assert_eq!(frames[0].mtime_ns, 10);
        assert_eq!(again[0].mtime_ns, 10);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reused_frames_keep_the_video_mtime() {
        let dir = crate::db::scratch_dir("video-mtime");
        let video = dir.join("clip.mp4");
        std::fs::write(&video, b"x").unwrap();
        let size = 9u64;
        let mtime = 42i64;
        let key = blake3::hash(format!("{}\n{size}\n{mtime}", video.display()).as_bytes());
        let hex = key.to_hex().to_string();
        let frame_dir = dir.join("frames").join(&hex[..16]);
        std::fs::create_dir_all(&frame_dir).unwrap();
        std::fs::write(frame_dir.join("clip-00-5.jpg"), b"jpeg").unwrap();
        std::fs::write(
            frame_dir.join("manifest.txt"),
            "clip-00-5.jpg\t0.500\tclip.mp4 @ 0.5s\t42\n",
        )
        .unwrap();
        let frames = cached_frames(&video, &dir, &dir.join("frames"), size, mtime).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].mtime_ns, 42);
        assert_eq!(frames[0].rel, "clip.mp4 @ 0.5s");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
