//! PP-OCRv5 detection and recognition through ONNX Runtime.
//!
//! Detector and recognizer inputs are BGR, matching the OpenCV images the
//! graphs were trained on. Detection uses ImageNet mean and std on that BGR
//! order. Recognition and the angle classifier use `(x / 255 - 0.5) / 0.5`.

use crate::geom::{self, corners, min_area_rect, orient_horizontal, unclip, warp};
use crate::runtime::{self, positive_dim, tensor_shape};
use crate::models;
use anyhow::{anyhow, Result};
use image::imageops::FilterType;
use image::RgbImage;
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use std::collections::BTreeMap;
use std::path::Path;

pub const DICT_ID: &str = "ppocrv5-en-e025a66d";
const DROP_SCORE: f32 = 0.5;

pub struct Ocr {
    det: Session,
    rec: Session,
    cls: Option<Session>,
    det_name: String,
    rec_name: String,
    cls_name: String,
    rec_height: usize,
    rec_width: Option<usize>,
    charset: Vec<String>,
    det_limit: u32,
}

impl Ocr {
    pub fn load(env: &ort::environment::Environment, dir: &Path, cuda: bool, det_limit: u32) -> Result<Self> {
        let det = runtime::open_session(env, &dir.join("det.onnx"), cuda)?;
        let rec = runtime::open_session(env, &dir.join("rec.onnx"), cuda)?;
        let cls_path = dir.join("cls.onnx");
        let cls = if cls_path.is_file() {
            match runtime::open_session(env, &cls_path, cuda) {
                Ok(session) => Some(session),
                Err(err) => {
                    eprintln!("semantix: angle classifier skipped: {err}");
                    None
                }
            }
        } else {
            None
        };
        let det_name = input_name(&det)?;
        let rec_name = input_name(&rec)?;
        let cls_name = match &cls {
            Some(session) => input_name(session)?,
            None => String::new(),
        };
        let rec_shape = tensor_shape(rec.inputs()[0].dtype());
        let rec_height = positive_dim(&rec_shape, 2).unwrap_or(48).max(8);
        let rec_width = positive_dim(&rec_shape, 3);
        let out_shape = tensor_shape(rec.outputs()[0].dtype());
        let classes = positive_dim(&out_shape, out_shape.len().saturating_sub(1)).ok_or_else(|| {
            anyhow!("recognition output has no fixed class count ({out_shape:?})")
        })?;
        let charset = load_charset(&dict_text(dir), classes)?;
        Ok(Self {
            det,
            rec,
            cls,
            det_name,
            rec_name,
            cls_name,
            rec_height,
            rec_width,
            charset,
            det_limit: det_limit.clamp(32, 4096),
        })
    }

    pub fn stamp(dir: &Path) -> String {
        let dict = if dir.join("dict.txt").is_file() {
            models::file_sig(&dir.join("dict.txt"))
        } else {
            DICT_ID.to_string()
        };
        format!(
            "det={};rec={};cls={};dict={dict}",
            models::file_sig(&dir.join("det.onnx")),
            models::file_sig(&dir.join("rec.onnx")),
            models::file_sig(&dir.join("cls.onnx")),
        )
    }

    pub fn set_det_limit(&mut self, limit: u32) {
        self.det_limit = limit.clamp(32, 4096);
    }

    pub fn read(&mut self, image: &RgbImage) -> Result<String> {
        Ok(self
            .lines(image)?
            .into_iter()
            .map(|(text, _)| text)
            .collect::<Vec<_>>()
            .join("\n"))
    }

    pub fn lines(&mut self, image: &RgbImage) -> Result<Vec<(String, f32)>> {
        if image.width() < 2 || image.height() < 2 {
            return Ok(Vec::new());
        }
        let quads = self.detect(image)?;
        let mut pending = Vec::new();
        let mut lines = Vec::new();
        for quad in quads {
            let width = dist(quad[0], quad[1]).round().max(2.0) as u32;
            let height = dist(quad[0], quad[3]).round().max(2.0) as u32;
            let Some(crop) = warp(image, quad, width, height) else {
                continue;
            };
            let crop = self.maybe_rotate(crop);
            pending.push(crop);
            if pending.len() == 8 {
                lines.extend(self.recognize(&pending)?);
                pending.clear();
            }
        }
        if !pending.is_empty() {
            lines.extend(self.recognize(&pending)?);
        }
        Ok(lines)
    }

    fn detect(&mut self, image: &RgbImage) -> Result<Vec<[(f32, f32); 4]>> {
        let (rw, rh) = det_hw(image.width(), image.height(), self.det_limit);
        let resized = image::imageops::resize(image, rw, rh, FilterType::Triangle);
        let tensor = det_tensor(&resized);
        let input = Tensor::from_array(tensor).map_err(|err| anyhow!("det tensor: {err}"))?;
        let name = self.det_name.clone();
        let outputs = self
            .det
            .run(ort::inputs![name.as_str() => input])
            .map_err(|err| anyhow!("text detection: {err}"))?;
        let (shape, data) = runtime::extract_f32(&outputs[0])?;
        let (out_h, out_w, map) = probability_map(&shape, &data)
            .ok_or_else(|| anyhow!("unexpected detection output {shape:?}"))?;
        let scale_x = image.width() as f32 / out_w as f32;
        let scale_y = image.height() as f32 / out_h as f32;
        let mut mask = vec![0u8; out_w * out_h];
        for (i, value) in map.iter().enumerate() {
            if *value > 0.3 {
                mask[i] = 1;
            }
        }
        let mut boxes = Vec::new();
        for pixels in components(&mask, out_w, out_h) {
            if pixels.len() < 3 {
                continue;
            }
            let score = pixels.iter().map(|(x, y)| map[y * out_w + x]).sum::<f32>() / pixels.len() as f32;
            if score < 0.6 {
                continue;
            }
            let mut rows: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
            for (x, y) in pixels {
                let span = rows.entry(y).or_insert((x, x));
                if x < span.0 {
                    span.0 = x;
                }
                if x > span.1 {
                    span.1 = x;
                }
            }
            let mut points = Vec::new();
            for (y, (left, right)) in rows {
                points.push((left as f32 * scale_x, y as f32 * scale_y));
                if right != left {
                    points.push((right as f32 * scale_x, y as f32 * scale_y));
                }
            }
            let Some(rect) = min_area_rect(&points) else {
                continue;
            };
            if rect.w.min(rect.h) < 3.0 {
                continue;
            }
            let rect = unclip(orient_horizontal(rect), 1.5);
            let rect = orient_horizontal(rect);
            if rect.w.min(rect.h) < 5.0 {
                continue;
            }
            boxes.push((top_left_first(corners(&rect)), score));
        }
        boxes.sort_by(|a, b| b.1.total_cmp(&a.1));
        boxes.truncate(1000);
        geom::sort_reading_order(&mut boxes);
        Ok(boxes.into_iter().map(|(quad, _)| quad).collect())
    }

    fn maybe_rotate(&mut self, crop: RgbImage) -> RgbImage {
        let Some(cls) = self.cls.as_mut() else {
            return crop;
        };
        let resized = image::imageops::resize(&crop, 192, 48, FilterType::Triangle);
        let mut tensor = Array4::<f32>::zeros((1, 3, 48, 192));
        fill_bgr_half(&mut tensor, 0, &resized);
        let Ok(input) = Tensor::from_array(tensor) else {
            return crop;
        };
        let name = self.cls_name.clone();
        let Ok(outputs) = cls.run(ort::inputs![name.as_str() => input]) else {
            return crop;
        };
        let Ok((_, data)) = runtime::extract_f32(&outputs[0]) else {
            return crop;
        };
        if data.len() < 2 {
            return crop;
        }
        let (up, down) = pair_probs(data[0], data[1]);
        if down > up && down > 0.9 {
            image::imageops::rotate180(&crop)
        } else {
            crop
        }
    }

    fn recognize(&mut self, crops: &[RgbImage]) -> Result<Vec<(String, f32)>> {
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        let height = self.rec_height;
        let widths: Vec<usize> = crops
            .iter()
            .map(|crop| rec_width_for(crop, height, self.rec_width))
            .collect();
        let max_w = widths.iter().copied().max().unwrap_or(8).max(8);
        let mut tensor = Array4::<f32>::zeros((crops.len(), 3, height, max_w));
        for (index, crop) in crops.iter().enumerate() {
            let resized = image::imageops::resize(crop, widths[index] as u32, height as u32, FilterType::Triangle);
            fill_bgr_half(&mut tensor, index, &resized);
        }
        let input = Tensor::from_array(tensor).map_err(|err| anyhow!("rec tensor: {err}"))?;
        let name = self.rec_name.clone();
        let outputs = self
            .rec
            .run(ort::inputs![name.as_str() => input])
            .map_err(|err| anyhow!("text recognition: {err}"))?;
        let (shape, data) = runtime::extract_f32(&outputs[0])?;
        let (steps, classes) = match shape.as_slice() {
            [n, steps, classes] if *n == crops.len() => (*steps, *classes),
            [steps, classes] if crops.len() == 1 => (*steps, *classes),
            other => return Err(anyhow!("unexpected recognition output {other:?}")),
        };
        if classes != self.charset.len() {
            return Err(anyhow!(
                "recognition produced {classes} classes but the dictionary covers {}",
                self.charset.len()
            ));
        }
        let mut lines = Vec::new();
        for index in 0..crops.len() {
            let start = index * steps * classes;
            let end = start + steps * classes;
            if end > data.len() {
                break;
            }
            let (text, confidence) = ctc_decode(&data[start..end], classes, &self.charset);
            if confidence >= DROP_SCORE && !text.trim().is_empty() {
                lines.push((text, confidence));
            }
        }
        Ok(lines)
    }
}

fn input_name(session: &Session) -> Result<String> {
    session
        .inputs()
        .first()
        .map(|input| input.name().to_string())
        .ok_or_else(|| anyhow!("model has no inputs"))
}

fn dict_text(dir: &Path) -> String {
    let path = dir.join("dict.txt");
    if let Ok(text) = std::fs::read_to_string(&path) {
        if !text.trim().is_empty() {
            return text;
        }
    }
    include_str!("../assets/ppocrv5_en_dict.txt").to_string()
}

pub fn load_charset(dict_text: &str, classes: usize) -> Result<Vec<String>> {
    let mut glyphs = Vec::new();
    for (index, line) in dict_text.split('\n').enumerate() {
        let line = if index == 0 {
            line.strip_prefix('\u{feff}').unwrap_or(line)
        } else {
            line
        };
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        glyphs.push(line.to_string());
    }
    if classes == glyphs.len() + 1 {
        let mut charset = vec!["<blank>".to_string()];
        charset.extend(glyphs);
        Ok(charset)
    } else if classes == glyphs.len() + 2 {
        let mut charset = vec!["<blank>".to_string()];
        charset.extend(glyphs);
        charset.push(" ".to_string());
        Ok(charset)
    } else {
        Err(anyhow!(
            "dictionary has {} glyphs but the model has {classes} classes (expected {} or {})",
            glyphs.len(),
            glyphs.len() + 1,
            glyphs.len() + 2
        ))
    }
}

pub fn ctc_collapse(ids: &[usize], charset: &[String]) -> String {
    let mut text = String::new();
    let mut prev = 0usize;
    for &id in ids {
        if id != 0 && id != prev {
            if let Some(glyph) = charset.get(id) {
                if glyph != "<blank>" {
                    text.push_str(glyph);
                }
            }
        }
        prev = id;
    }
    text
}

fn ctc_decode(row: &[f32], classes: usize, charset: &[String]) -> (String, f32) {
    if classes == 0 || row.len() < classes {
        return (String::new(), 0.0);
    }
    let steps = row.len() / classes;
    let mut ids = Vec::with_capacity(steps);
    let mut confidence = 0.0f32;
    let mut kept = 0u32;
    let mut prev = 0usize;
    for step_index in 0..steps {
        let step = &row[step_index * classes..(step_index + 1) * classes];
        let probs = if step_is_prob(step) { step.to_vec() } else { softmax(step) };
        let mut best_i = 0usize;
        let mut best_p = f32::MIN;
        for (index, value) in probs.iter().enumerate() {
            if *value > best_p {
                best_p = *value;
                best_i = index;
            }
        }
        if best_i != 0 && best_i != prev {
            confidence += best_p;
            kept += 1;
        }
        ids.push(best_i);
        prev = best_i;
    }
    let text = ctc_collapse(&ids, charset);
    let mean = if kept == 0 { 0.0 } else { confidence / kept as f32 };
    (text, mean)
}

fn step_is_prob(step: &[f32]) -> bool {
    let mut sum = 0.0f32;
    for value in step {
        if *value < -0.01 || *value > 1.01 {
            return false;
        }
        sum += *value;
    }
    (sum - 1.0).abs() < 0.05
}

fn softmax(step: &[f32]) -> Vec<f32> {
    let max = step.iter().copied().fold(f32::MIN, f32::max);
    let mut out = vec![0.0f32; step.len()];
    let mut sum = 0.0f32;
    for (index, value) in step.iter().enumerate() {
        let exp = (value - max).exp();
        out[index] = exp;
        sum += exp;
    }
    let sum = sum.max(1e-12);
    for value in &mut out {
        *value /= sum;
    }
    out
}

fn pair_probs(a: f32, b: f32) -> (f32, f32) {
    if a < -0.01 || b < -0.01 || a > 1.01 || b > 1.01 || (a + b - 1.0).abs() > 0.05 {
        let max = a.max(b);
        let e0 = (a - max).exp();
        let e1 = (b - max).exp();
        let sum = (e0 + e1).max(1e-12);
        (e0 / sum, e1 / sum)
    } else {
        (a, b)
    }
}

pub fn py_round(value: f32) -> i32 {
    if !value.is_finite() {
        return 0;
    }
    let negative = value < 0.0;
    let value = value.abs();
    let floor = value.floor();
    let diff = value - floor;
    let mut rounded = if (diff - 0.5).abs() <= 1e-4 {
        let even = floor as i32;
        if even % 2 == 0 { even } else { even + 1 }
    } else if diff > 0.5 {
        floor as i32 + 1
    } else {
        floor as i32
    };
    if negative {
        rounded = -rounded;
    }
    rounded
}

pub fn det_hw(width: u32, height: u32, limit: u32) -> (u32, u32) {
    let limit = limit.max(32) as f32;
    let width = width.max(1) as f32;
    let height = height.max(1) as f32;
    let long = width.max(height);
    let ratio = if long > limit { limit / long } else { 1.0 };
    (align32(width * ratio), align32(height * ratio))
}

fn align32(side: f32) -> u32 {
    let blocks = py_round(side / 32.0).max(1) as u32;
    blocks * 32
}

fn det_tensor(image: &RgbImage) -> Array4<f32> {
    let height = image.height() as usize;
    let width = image.width() as usize;
    let mut dest = Array4::<f32>::zeros((1, 3, height, width));
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32);
            let bgr = [pixel[2] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[0] as f32 / 255.0];
            for channel in 0..3 {
                dest[[0, channel, y, x]] = (bgr[channel] - MEAN[channel]) / STD[channel];
            }
        }
    }
    dest
}

fn fill_bgr_half(dest: &mut Array4<f32>, index: usize, image: &RgbImage) {
    let height = image.height() as usize;
    let width = image.width() as usize;
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32);
            dest[[index, 0, y, x]] = (pixel[2] as f32 / 255.0 - 0.5) / 0.5;
            dest[[index, 1, y, x]] = (pixel[1] as f32 / 255.0 - 0.5) / 0.5;
            dest[[index, 2, y, x]] = (pixel[0] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
}

fn rec_width_for(crop: &RgbImage, height: usize, fixed: Option<usize>) -> usize {
    if let Some(width) = fixed {
        return width.max(8);
    }
    let ratio = crop.width() as f32 / crop.height().max(1) as f32;
    let mut width = (height as f32 * ratio).ceil() as usize;
    width = width.max(8);
    width = width.div_ceil(8) * 8;
    width.min(960)
}

fn probability_map(shape: &[usize], data: &[f32]) -> Option<(usize, usize, Vec<f32>)> {
    let (height, width) = match shape {
        [_, _, height, width] => (*height, *width),
        [_, height, width] => (*height, *width),
        [height, width] => (*height, *width),
        _ => return None,
    };
    if height == 0 || width == 0 || data.len() < height * width {
        return None;
    }
    let mut map = data[..height * width].to_vec();
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for value in &map {
        min = min.min(*value);
        max = max.max(*value);
    }
    if min < -0.01 || max > 1.01 {
        for value in &mut map {
            *value = 1.0 / (1.0 + (-*value).exp());
        }
    }
    Some((height, width, map))
}

fn components(mask: &[u8], width: usize, height: usize) -> Vec<Vec<(usize, usize)>> {
    let mut seen = vec![0u8; width * height];
    let mut found = Vec::new();
    let mut stack = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let start = y * width + x;
            if mask[start] == 0 || seen[start] != 0 {
                continue;
            }
            seen[start] = 1;
            stack.push((x, y));
            let mut pixels = Vec::new();
            while let Some((cx, cy)) = stack.pop() {
                pixels.push((cx, cy));
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = cx as i32 + dx;
                        let ny = cy as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                            continue;
                        }
                        let next = ny as usize * width + nx as usize;
                        if mask[next] == 1 && seen[next] == 0 {
                            seen[next] = 1;
                            stack.push((nx as usize, ny as usize));
                        }
                    }
                }
            }
            found.push(pixels);
        }
    }
    found
}

fn top_left_first(quad: [(f32, f32); 4]) -> [(f32, f32); 4] {
    let mut best = 0usize;
    for index in 1..4 {
        let better_y = quad[index].1 < quad[best].1 - 1.0;
        let same_y = (quad[index].1 - quad[best].1).abs() <= 1.0 && quad[index].0 < quad[best].0;
        if better_y || same_y {
            best = index;
        }
    }
    let mut out = [(0.0, 0.0); 4];
    for index in 0..4 {
        out[index] = quad[(best + index) % 4];
    }
    out
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_blank_separates_repeats() {
        let charset = vec!["<blank>".to_string(), "a".to_string()];
        assert_eq!(ctc_collapse(&[1, 1, 0, 1], &charset), "aa");
    }

    #[test]
    fn det_resize_uses_half_even() {
        assert_eq!(py_round(48.5), 48);
        assert_eq!(py_round(47.5), 48);
        assert_eq!(py_round(2.5), 2);
        assert_eq!(py_round(1.5), 2);
        assert_eq!(det_hw(960, 720, 960), (960, 704));
    }

    #[test]
    fn bundled_dict_matches_v5_head() {
        let text = include_str!("../assets/ppocrv5_en_dict.txt");
        let charset = load_charset(text, 438).unwrap();
        assert_eq!(charset.len(), 438);
        assert_eq!(charset[0], "<blank>");
        assert_eq!(charset[437], " ");
    }
}
