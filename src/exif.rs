//! JPEG APP1 EXIF reader. Orientation, camera, exposure, and capture time are
//! enough for the gallery. Other containers keep the default orientation.

use image::DynamicImage;
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct Meta {
    pub orientation: u16,
    pub json: String,
}

impl Default for Meta {
    fn default() -> Self {
        Self {
            orientation: 1,
            json: "{\"orientation\":1}".to_string(),
        }
    }
}

pub fn from_jpeg(bytes: &[u8]) -> Meta {
    if bytes.len() < 4 || bytes[0] != 0xFF || bytes[1] != 0xD8 {
        return Meta::default();
    }
    let mut i = 2usize;
    while i + 4 < bytes.len() {
        if bytes[i] != 0xFF {
            break;
        }
        while i < bytes.len() && bytes[i] == 0xFF {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let marker = bytes[i];
        i += 1;
        if marker == 0xD8 || marker == 0x01 || (0xD0..=0xD9).contains(&marker) {
            if marker == 0xD9 || marker == 0xDA {
                break;
            }
            continue;
        }
        if i + 2 > bytes.len() {
            break;
        }
        let len = u16::from_be_bytes([bytes[i], bytes[i + 1]]) as usize;
        if len < 2 || i + len > bytes.len() {
            break;
        }
        if marker == 0xE1 {
            let payload = &bytes[i + 2..i + len];
            if let Some(tiff) = payload.strip_prefix(b"Exif\0\0") {
                return parse_tiff(tiff);
            }
        }
        if marker == 0xDA {
            break;
        }
        i += len;
    }
    Meta::default()
}

pub fn parse_tiff(data: &[u8]) -> Meta {
    if data.len() < 8 {
        return Meta::default();
    }
    let le = match &data[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return Meta::default(),
    };
    if read_u16(data, 2, le) != Some(42) {
        return Meta::default();
    }
    let Some(offset) = read_u32(data, 4, le) else {
        return Meta::default();
    };
    let mut info = read_ifd(data, offset as usize, le);
    if let Some(sub) = info.exif_offset {
        let extra = read_ifd(data, sub as usize, le);
        if extra.iso.is_some() {
            info.iso = extra.iso;
        }
        if extra.exposure.is_some() {
            info.exposure = extra.exposure;
        }
        if extra.fnumber.is_some() {
            info.fnumber = extra.fnumber;
        }
        // DateTimeOriginal and DateTimeDigitized live in the Exif sub-IFD.
        // IFD0 DateTime is the last-modified stamp and stays the fallback.
        if extra.datetime_original.is_some() {
            info.datetime_original = extra.datetime_original;
        }
        if extra.datetime_digitized.is_some() {
            info.datetime_digitized = extra.datetime_digitized;
        }
        if info.datetime.is_none() {
            info.datetime = extra.datetime;
        }
    }
    let orientation = info.orientation.unwrap_or(1).clamp(1, 8);
    let datetime = chosen_datetime(&info);
    let mut body = serde_json::Map::new();
    body.insert("orientation".into(), json!(orientation));
    if let Some(make) = info.make {
        body.insert("make".into(), Value::String(make));
    }
    if let Some(model) = info.model {
        body.insert("model".into(), Value::String(model));
    }
    if let Some(datetime) = datetime {
        body.insert("datetime".into(), Value::String(datetime));
    }
    if let Some(iso) = info.iso {
        body.insert("iso".into(), json!(iso));
    }
    if let Some(exposure) = info.exposure {
        body.insert("exposure".into(), Value::String(exposure));
    }
    if let Some(fnumber) = info.fnumber {
        body.insert("fnumber".into(), json!(fnumber));
    }
    Meta {
        orientation,
        json: Value::Object(body).to_string(),
    }
}

struct Ifd {
    orientation: Option<u16>,
    make: Option<String>,
    model: Option<String>,
    datetime: Option<String>,
    datetime_original: Option<String>,
    datetime_digitized: Option<String>,
    exif_offset: Option<u32>,
    iso: Option<u32>,
    exposure: Option<String>,
    fnumber: Option<f64>,
}

fn chosen_datetime(info: &Ifd) -> Option<String> {
    info.datetime_original
        .clone()
        .or(info.datetime_digitized.clone())
        .or(info.datetime.clone())
}

/// Stored EXIF `datetime` as Unix nanoseconds. The stamp is read as UTC.
/// Missing, blank, and unparsable values return None so the caller can use
/// the file modification time. A leap second is clamped to `:59`.
pub fn taken_ns(json: &str) -> Option<i64> {
    let value: Value = serde_json::from_str(json).ok()?;
    let text = value.get("datetime")?.as_str()?;
    exif_datetime_ns(text)
}

fn exif_datetime_ns(text: &str) -> Option<i64> {
    let text = text.trim();
    let bytes = text.as_bytes();
    if bytes.len() < 19
        || bytes[4] != b':'
        || bytes[7] != b':'
        || bytes[10] != b' '
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return None;
    }
    let year: i32 = text[0..4].parse().ok()?;
    let month: u32 = text[5..7].parse().ok()?;
    let day: u32 = text[8..10].parse().ok()?;
    let hour: u32 = text[11..13].parse().ok()?;
    let minute: u32 = text[14..16].parse().ok()?;
    let mut second: u32 = text[17..19].parse().ok()?;
    if !(1..=9999).contains(&year) || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    if second == 60 {
        second = 59;
    }
    let days = days_from_civil(year, month, day)?;
    let unix = days
        .checked_mul(86_400)?
        .checked_add(i64::from(hour) * 3_600)?
        .checked_add(i64::from(minute) * 60)?
        .checked_add(i64::from(second))?;
    unix.checked_mul(1_000_000_000)
}

/// Days since 1970-01-01. Howard Hinnant's civil-from-days inverse.
fn days_from_civil(year: i32, month: u32, day: u32) -> Option<i64> {
    let max = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(year) => 29,
        2 => 28,
        _ => return None,
    };
    if day == 0 || day > max {
        return None;
    }
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp as u64 + 2) / 5 + day as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era as i64 * 146_097 + doe as i64 - 719_468)
}

fn is_leap(year: i32) -> bool {
    let extra = year % 400 == 0;
    year % 4 == 0 && (year % 100 != 0 || extra)
}

fn read_ifd(data: &[u8], offset: usize, le: bool) -> Ifd {
    let mut info = Ifd {
        orientation: None,
        make: None,
        model: None,
        datetime: None,
        datetime_original: None,
        datetime_digitized: None,
        exif_offset: None,
        iso: None,
        exposure: None,
        fnumber: None,
    };
    let Some(count) = read_u16(data, offset, le) else {
        return info;
    };
    for n in 0..count as usize {
        let at = offset + 2 + n * 12;
        let Some(tag) = read_u16(data, at, le) else { continue };
        let Some(kind) = read_u16(data, at + 2, le) else { continue };
        let Some(nvals) = read_u32(data, at + 4, le) else { continue };
        match tag {
            0x0112 => info.orientation = read_short(data, at, kind, nvals, le),
            0x010F => info.make = read_ascii(data, at, kind, nvals, le),
            0x0110 => info.model = read_ascii(data, at, kind, nvals, le),
            0x0132 => info.datetime = datetime_text(data, at, kind, nvals, le),
            0x9003 => info.datetime_original = datetime_text(data, at, kind, nvals, le),
            0x9004 => info.datetime_digitized = datetime_text(data, at, kind, nvals, le),
            0x8769 => info.exif_offset = read_long(data, at, kind, nvals, le),
            0x8827 => info.iso = read_long(data, at, kind, nvals, le).or_else(|| {
                read_short(data, at, kind, nvals, le).map(u32::from)
            }),
            0x829A => info.exposure = read_rational(data, at, kind, nvals, le).map(|(n, d)| {
                if d == 0 { format!("{n}") } else { format!("{n}/{d}") }
            }),
            0x829D => {
                if let Some((n, d)) = read_rational(data, at, kind, nvals, le) {
                    if d != 0 {
                        info.fnumber = Some(n as f64 / d as f64);
                    }
                }
            }
            _ => {}
        }
    }
    info
}

fn read_short(data: &[u8], entry: usize, kind: u16, count: u32, le: bool) -> Option<u16> {
    if kind != 3 || count < 1 {
        return None;
    }
    let inline = count * 2 <= 4;
    let at = if inline { entry + 8 } else { read_u32(data, entry + 8, le)? as usize };
    read_u16(data, at, le)
}

fn read_long(data: &[u8], entry: usize, kind: u16, count: u32, le: bool) -> Option<u32> {
    if count < 1 {
        return None;
    }
    if kind == 4 {
        read_u32(data, entry + 8, le)
    } else if kind == 3 {
        read_short(data, entry, kind, count, le).map(u32::from)
    } else {
        None
    }
}

fn read_rational(data: &[u8], entry: usize, kind: u16, count: u32, le: bool) -> Option<(u32, u32)> {
    if kind != 5 || count < 1 {
        return None;
    }
    let at = read_u32(data, entry + 8, le)? as usize;
    Some((read_u32(data, at, le)?, read_u32(data, at + 4, le)?))
}

fn datetime_text(data: &[u8], entry: usize, kind: u16, count: u32, le: bool) -> Option<String> {
    read_ascii(data, entry, kind, count, le).filter(|text| !text.is_empty())
}

fn read_ascii(data: &[u8], entry: usize, kind: u16, count: u32, le: bool) -> Option<String> {
    if kind != 2 || count == 0 {
        return None;
    }
    let at = if count <= 4 {
        entry + 8
    } else {
        read_u32(data, entry + 8, le)? as usize
    };
    let end = at.saturating_add(count as usize).min(data.len());
    if at >= data.len() {
        return None;
    }
    let bytes = &data[at..end];
    let bytes = bytes.split(|b| *b == 0).next().unwrap_or(bytes);
    Some(String::from_utf8_lossy(bytes).trim().to_string())
}

fn read_u16(data: &[u8], at: usize, le: bool) -> Option<u16> {
    let b = data.get(at..at + 2)?;
    Some(if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) })
}

fn read_u32(data: &[u8], at: usize, le: bool) -> Option<u32> {
    let b = data.get(at..at + 4)?;
    Some(if le {
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    } else {
        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    })
}

pub fn apply_orientation(img: DynamicImage, orientation: u16) -> DynamicImage {
    match orientation {
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.rotate90().fliph(),
        6 => img.rotate90(),
        7 => img.rotate270().fliph(),
        8 => img.rotate270(),
        _ => img,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn le_ifd() -> Vec<u8> {
        let mut data = vec![0u8; 48];
        data[0..2].copy_from_slice(b"II");
        data[2..4].copy_from_slice(&42u16.to_le_bytes());
        data[4..8].copy_from_slice(&8u32.to_le_bytes());
        data[8..10].copy_from_slice(&2u16.to_le_bytes());
        // Orientation = 6, SHORT, count 1, inline.
        data[10..12].copy_from_slice(&0x0112u16.to_le_bytes());
        data[12..14].copy_from_slice(&3u16.to_le_bytes());
        data[14..18].copy_from_slice(&1u32.to_le_bytes());
        data[18..20].copy_from_slice(&6u16.to_le_bytes());
        // Make = "Semantix", ASCII, stored after the IFD.
        data[22..24].copy_from_slice(&0x010Fu16.to_le_bytes());
        data[24..26].copy_from_slice(&2u16.to_le_bytes());
        data[26..30].copy_from_slice(&9u32.to_le_bytes());
        data[30..34].copy_from_slice(&38u32.to_le_bytes());
        data[38..47].copy_from_slice(b"Semantix\0");
        data
    }

    #[test]
    fn reads_little_endian_orientation_and_make() {
        let meta = parse_tiff(&le_ifd());
        assert_eq!(meta.orientation, 6);
        assert!(meta.json.contains("Semantix"), "{}", meta.json);
        assert!(meta.json.contains("\"orientation\":6"), "{}", meta.json);
    }

    /// `stamps` are `(tag, text, in_sub_ifd)`. ASCII values sit after both IFDs.
    fn tiff_datetime(stamps: &[(u16, &str, bool)]) -> Vec<u8> {
        let ifd0_count = stamps.iter().filter(|stamp| !stamp.2).count()
            + usize::from(stamps.iter().any(|stamp| stamp.2));
        let sub_count = stamps.iter().filter(|stamp| stamp.2).count();
        let ifd0_at = 8usize;
        let after_ifd0 = ifd0_at + 2 + ifd0_count * 12 + 4;
        let sub_at = after_ifd0;
        let after_sub = if sub_count == 0 {
            after_ifd0
        } else {
            sub_at + 2 + sub_count * 12 + 4
        };
        let mut cursor = after_sub;
        let mut placed = Vec::with_capacity(stamps.len());
        let mut strings = Vec::with_capacity(stamps.len());
        for (_, text, _) in stamps {
            let mut bytes = text.as_bytes().to_vec();
            bytes.push(0);
            placed.push((cursor, bytes.len() as u32));
            cursor += bytes.len();
            strings.push(bytes);
        }
        let mut data = vec![0u8; cursor];
        data[0..2].copy_from_slice(b"II");
        data[2..4].copy_from_slice(&42u16.to_le_bytes());
        data[4..8].copy_from_slice(&(ifd0_at as u32).to_le_bytes());
        data[ifd0_at..ifd0_at + 2].copy_from_slice(&(ifd0_count as u16).to_le_bytes());
        let mut ifd0_n = 0usize;
        let mut sub_n = 0usize;
        if sub_count > 0 {
            let at = ifd0_at + 2 + (ifd0_count - 1) * 12;
            data[at..at + 2].copy_from_slice(&0x8769u16.to_le_bytes());
            data[at + 2..at + 4].copy_from_slice(&4u16.to_le_bytes());
            data[at + 4..at + 8].copy_from_slice(&1u32.to_le_bytes());
            data[at + 8..at + 12].copy_from_slice(&(sub_at as u32).to_le_bytes());
            data[sub_at..sub_at + 2].copy_from_slice(&(sub_count as u16).to_le_bytes());
        }
        for (n, (tag, _, sub)) in stamps.iter().enumerate() {
            let at = if *sub {
                let at = sub_at + 2 + sub_n * 12;
                sub_n += 1;
                at
            } else {
                let at = ifd0_at + 2 + ifd0_n * 12;
                ifd0_n += 1;
                at
            };
            let (offset, count) = placed[n];
            data[at..at + 2].copy_from_slice(&tag.to_le_bytes());
            data[at + 2..at + 4].copy_from_slice(&2u16.to_le_bytes());
            data[at + 4..at + 8].copy_from_slice(&count.to_le_bytes());
            data[at + 8..at + 12].copy_from_slice(&(offset as u32).to_le_bytes());
            data[offset..offset + strings[n].len()].copy_from_slice(&strings[n]);
        }
        data
    }

    #[test]
    fn capture_time_prefers_original_then_digitized_then_modified() {
        let original = parse_tiff(&tiff_datetime(&[
            (0x0132, "2020:01:02 03:04:05", false),
            (0x9003, "2019:05:06 07:08:09", true),
            (0x9004, "2018:01:01 00:00:00", true),
        ]));
        assert!(original.json.contains("2019:05:06 07:08:09"), "{}", original.json);
        assert!(!original.json.contains("2020:01:02"), "{}", original.json);
        let digitized = parse_tiff(&tiff_datetime(&[
            (0x0132, "2020:01:02 03:04:05", false),
            (0x9004, "2018:03:04 05:06:07", true),
        ]));
        assert!(digitized.json.contains("2018:03:04 05:06:07"), "{}", digitized.json);
        let modified = parse_tiff(&tiff_datetime(&[(0x0132, "2020:01:02 03:04:05", false)]));
        assert!(modified.json.contains("2020:01:02 03:04:05"), "{}", modified.json);
        let spaces = " ".repeat(19);
        let blank = parse_tiff(&tiff_datetime(&[
            (0x0132, "2020:01:02 03:04:05", false),
            (0x9003, &spaces, true),
            (0x9004, "2018:03:04 05:06:07", true),
        ]));
        assert!(blank.json.contains("2018:03:04 05:06:07"), "{}", blank.json);
    }

    #[test]
    fn taken_ns_parses_exif_datetimes() {
        assert_eq!(taken_ns(r#"{"datetime":"1970:01:01 00:00:00"}"#), Some(0));
        assert_eq!(
            taken_ns(r#"{"datetime":"2020:01:01 00:00:00"}"#),
            Some(1_577_836_800_000_000_000)
        );
        assert_eq!(
            taken_ns(r#"{"datetime":"1969:12:31 23:59:59"}"#),
            Some(-1_000_000_000)
        );
        assert_eq!(
            taken_ns(r#"{"datetime":"2020:01:01 00:00:00.50"}"#),
            taken_ns(r#"{"datetime":"2020:01:01 00:00:00"}"#)
        );
        assert_eq!(
            taken_ns(r#"{"datetime":"1972:06:30 23:59:60"}"#),
            taken_ns(r#"{"datetime":"1972:06:30 23:59:59"}"#)
        );
        assert!(taken_ns(r#"{"datetime":"2020:02:29 00:00:00"}"#).is_some());
        assert_eq!(taken_ns(r#"{"datetime":"2019:02:29 00:00:00"}"#), None);
        assert_eq!(taken_ns(r#"{"datetime":"0000:00:00 00:00:00"}"#), None);
        assert_eq!(taken_ns(r#"{"orientation":1}"#), None);
        assert_eq!(taken_ns(""), None);
        assert_eq!(taken_ns(r#"{"time":"0.400s"}"#), None);
    }

    #[test]
    fn orientation_six_swaps_axes() {
        let mut raw = RgbImage::new(2, 1);
        raw.put_pixel(0, 0, Rgb([1, 0, 0]));
        raw.put_pixel(1, 0, Rgb([2, 0, 0]));
        let turned = apply_orientation(DynamicImage::ImageRgb8(raw), 6);
        assert_eq!((turned.width(), turned.height()), (1, 2));
    }
}
