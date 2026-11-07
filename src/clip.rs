//! OpenAI CLIP ViT-B/32: byte-level BPE text encoder and image encoder.
//!
//! The graphs are the MIT-licensed Xenova ONNX export. MobileCLIP is not used;
//! its license is not a standard permissive grant.

use crate::models;
use crate::runtime::{self, positive_dim, tensor_shape};
use anyhow::{anyhow, Result};
use image::imageops::FilterType;
use image::RgbImage;
use ndarray::{Array2, Array4};
use ort::session::Session;
use ort::value::{Tensor, TensorElementType};
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

const CONTEXT: usize = 77;

pub struct Clip {
    vision: Session,
    text: Session,
    vision_name: String,
    text_name: String,
    vision_out: String,
    text_out: String,
    tokenizer: Tokenizer,
    size: u32,
    dim: usize,
}

impl Clip {
    pub fn load(env: &ort::environment::Environment, dir: &Path, cuda: bool) -> Result<Self> {
        let vision = runtime::open_session(env, &dir.join("vision.onnx"), cuda)?;
        let text = runtime::open_session(env, &dir.join("text.onnx"), cuda)?;
        let vision_name = first_name(vision.inputs()).ok_or_else(|| anyhow!("CLIP vision has no inputs"))?;
        let text_name = first_name(text.inputs()).ok_or_else(|| anyhow!("CLIP text has no inputs"))?;
        let text_type = text.inputs()[0].dtype().tensor_type();
        if text_type != Some(TensorElementType::Int64) {
            return Err(anyhow!("CLIP text input must be int64, found {text_type:?}"));
        }
        let in_shape = tensor_shape(vision.inputs()[0].dtype());
        let side_h = positive_dim(&in_shape, 2);
        let side_w = positive_dim(&in_shape, 3);
        let size = match (side_h, side_w) {
            (Some(h), Some(w)) if h == w => h as u32,
            _ => 224,
        };
        let vision_out = embed_name(vision.outputs()).unwrap_or_else(|| vision_name.clone());
        let text_out = embed_name(text.outputs()).unwrap_or_else(|| text_name.clone());
        let out_shape = tensor_shape(vision.outputs().iter().find(|o| o.name() == vision_out).unwrap_or(&vision.outputs()[0]).dtype());
        let dim = positive_dim(&out_shape, 1).unwrap_or(512);
        let tokenizer = Tokenizer::load(&dir.join("vocab.json"), &dir.join("merges.txt"))?;
        Ok(Self {
            vision,
            text,
            vision_name,
            text_name,
            vision_out,
            text_out,
            tokenizer,
            size,
            dim,
        })
    }

    pub fn stamp(dir: &Path) -> String {
        format!(
            "vision={};text={};vocab={};merges={}",
            models::file_sig(&dir.join("vision.onnx")),
            models::file_sig(&dir.join("text.onnx")),
            models::file_sig(&dir.join("vocab.json")),
            models::file_sig(&dir.join("merges.txt")),
        )
    }

    pub fn dim(&self) -> u32 {
        self.dim as u32
    }

    pub fn text(&mut self, query: &str) -> Result<Vec<f32>> {
        let ids = self.tokenizer.encode(query);
        let array = Array2::from_shape_vec((1, CONTEXT), ids)?;
        let tensor = Tensor::from_array(array).map_err(|err| anyhow!("text tensor: {err}"))?;
        let name = self.text_name.clone();
        let outputs = self
            .text
            .run(ort::inputs![name.as_str() => tensor])
            .map_err(|err| anyhow!("CLIP text: {err}"))?;
        let value = outputs.get(&self.text_out).unwrap_or(&outputs[0]);
        let (_shape, data) = runtime::extract_f32(value)?;
        take_normalized(&data, self.dim)
    }

    pub fn image(&mut self, image: &RgbImage) -> Result<Vec<f32>> {
        let cropped = clip_image(image, self.size);
        let tensor = clip_tensor(&cropped);
        let input = Tensor::from_array(tensor).map_err(|err| anyhow!("image tensor: {err}"))?;
        let name = self.vision_name.clone();
        let outputs = self
            .vision
            .run(ort::inputs![name.as_str() => input])
            .map_err(|err| anyhow!("CLIP image: {err}"))?;
        let value = outputs.get(&self.vision_out).unwrap_or(&outputs[0]);
        let (_shape, data) = runtime::extract_f32(value)?;
        take_normalized(&data, self.dim)
    }
}

fn first_name(outlets: &[ort::value::Outlet]) -> Option<String> {
    outlets.first().map(|outlet| outlet.name().to_string())
}

fn embed_name(outlets: &[ort::value::Outlet]) -> Option<String> {
    outlets
        .iter()
        .find(|outlet| outlet.name().to_ascii_lowercase().contains("embed"))
        .map(|outlet| outlet.name().to_string())
}

fn take_normalized(data: &[f32], dim: usize) -> Result<Vec<f32>> {
    if data.len() < dim || dim == 0 {
        return Err(anyhow!("CLIP embedding length {} is shorter than {dim}", data.len()));
    }
    Ok(l2_normalize(data[..dim].to_vec()))
}

fn l2_normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let mut sum = 0.0f32;
    for value in &vector {
        sum += value * value;
    }
    let norm = sum.sqrt();
    if norm > 1e-12 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

fn clip_image(image: &RgbImage, size: u32) -> RgbImage {
    let (width, height) = image.dimensions();
    let short = width.min(height).max(1) as f32;
    let scale = size as f32 / short;
    let mut nw = (width as f32 * scale).round().max(1.0) as u32;
    let mut nh = (height as f32 * scale).round().max(1.0) as u32;
    if nw < size {
        nw = size;
    }
    if nh < size {
        nh = size;
    }
    let resized = image::imageops::resize(image, nw, nh, FilterType::CatmullRom);
    let x = resized.width().saturating_sub(size) / 2;
    let y = resized.height().saturating_sub(size) / 2;
    image::imageops::crop_imm(&resized, x, y, size.min(resized.width()), size.min(resized.height())).to_image()
}

fn clip_tensor(image: &RgbImage) -> Array4<f32> {
    let height = image.height() as usize;
    let width = image.width() as usize;
    let mut dest = Array4::<f32>::zeros((1, 3, height, width));
    // Published OpenCLIP / OpenAI normalization. The extra digits are the
    // constants the model was trained with, kept as written.
    #[allow(clippy::excessive_precision)]
    const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x as u32, y as u32);
            for channel in 0..3 {
                let value = pixel[channel] as f32 / 255.0;
                dest[[0, channel, y, x]] = (value - MEAN[channel]) / STD[channel];
            }
        }
    }
    dest
}

pub struct Tokenizer {
    encoder: HashMap<String, i64>,
    ranks: HashMap<(String, String), usize>,
    bytes: Vec<char>,
    sot: i64,
    eot: i64,
    cache: HashMap<String, Vec<String>>,
}

impl Tokenizer {
    pub fn load(vocab: &Path, merges: &Path) -> Result<Self> {
        let vocab_text = std::fs::read_to_string(vocab).map_err(|err| anyhow!("reading {}: {err}", vocab.display()))?;
        let merges_text = std::fs::read_to_string(merges).map_err(|err| anyhow!("reading {}: {err}", merges.display()))?;
        Self::from_text(&vocab_text, &merges_text)
    }

    fn from_text(vocab_text: &str, merges_text: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(vocab_text)?;
        let object = value.as_object().ok_or_else(|| anyhow!("vocab.json is not an object"))?;
        let mut encoder = HashMap::with_capacity(object.len());
        for (key, id) in object {
            let id = id.as_i64().ok_or_else(|| anyhow!("vocab id for {key} is not an integer"))?;
            encoder.insert(key.clone(), id);
        }
        let sot = *encoder
            .get("<|startoftext|>")
            .ok_or_else(|| anyhow!("vocab is missing <|startoftext|>"))?;
        let eot = *encoder
            .get("<|endoftext|>")
            .ok_or_else(|| anyhow!("vocab is missing <|endoftext|>"))?;
        let mut ranks = HashMap::new();
        for line in merges_text.lines() {
            let line = line.trim().trim_start_matches('\u{feff}');
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split(' ');
            let Some(left) = parts.next() else { continue };
            let Some(right) = parts.next() else { continue };
            if parts.next().is_some() || left.is_empty() || right.is_empty() {
                continue;
            }
            let rank = ranks.len();
            ranks.insert((left.to_string(), right.to_string()), rank);
        }
        Ok(Self {
            encoder,
            ranks,
            bytes: bytes_to_unicode(),
            sot,
            eot,
            cache: HashMap::new(),
        })
    }

    pub fn encode(&mut self, text: &str) -> Vec<i64> {
        let cleaned = whitespace_clean(&basic_clean(text)).to_lowercase();
        let mut pieces = Vec::new();
        for token in token_pattern().find_iter(&cleaned) {
            let mapped: String = token.as_str().bytes().map(|byte| self.bytes[byte as usize]).collect();
            for piece in self.bpe(&mapped) {
                if let Some(id) = self.encoder.get(&piece) {
                    pieces.push(*id);
                }
            }
        }
        let mut ids = Vec::with_capacity(CONTEXT);
        ids.push(self.sot);
        if pieces.len() > CONTEXT - 2 {
            ids.extend(pieces.into_iter().take(CONTEXT - 2));
        } else {
            ids.extend(pieces);
        }
        ids.push(self.eot);
        ids.resize(CONTEXT, 0);
        ids
    }

    fn bpe(&mut self, token: &str) -> Vec<String> {
        if let Some(cached) = self.cache.get(token) {
            return cached.clone();
        }
        let mut word: Vec<String> = token.chars().map(|ch| ch.to_string()).collect();
        if word.is_empty() {
            return Vec::new();
        }
        let last = word.len() - 1;
        word[last].push_str("</w>");
        if word.len() > 1 {
            loop {
                let mut best_rank = usize::MAX;
                let mut best_pair: Option<(String, String)> = None;
                for index in 0..word.len() - 1 {
                    let pair = (word[index].clone(), word[index + 1].clone());
                    if let Some(rank) = self.ranks.get(&pair) {
                        if *rank < best_rank {
                            best_rank = *rank;
                            best_pair = Some(pair);
                        }
                    }
                }
                let Some((first, second)) = best_pair else {
                    break;
                };
                let mut merged = Vec::with_capacity(word.len());
                let mut index = 0;
                while index < word.len() {
                    if index + 1 < word.len() && word[index] == first && word[index + 1] == second {
                        merged.push(format!("{first}{second}"));
                        index += 2;
                    } else {
                        merged.push(word[index].clone());
                        index += 1;
                    }
                }
                word = merged;
                if word.len() == 1 {
                    break;
                }
            }
        }
        self.cache.insert(token.to_string(), word.clone());
        word
    }
}

pub fn bytes_to_unicode() -> Vec<char> {
    let mut bs: Vec<u32> = (b'!'..=b'~').map(|byte| byte as u32).collect();
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);
    let mut cs = bs.clone();
    let mut extra = 0u32;
    for byte in 0..256u32 {
        if !bs.contains(&byte) {
            bs.push(byte);
            cs.push(256 + extra);
            extra += 1;
        }
    }
    let mut out = vec!['\0'; 256];
    for (byte, code) in bs.into_iter().zip(cs) {
        out[byte as usize] = char::from_u32(code).unwrap_or('\0');
    }
    out
}

fn token_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")
            .expect("CLIP token pattern")
    })
}

fn basic_clean(text: &str) -> String {
    html_unescape(&html_unescape(text)).trim().to_string()
}

fn html_unescape(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn whitespace_clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_encoder_covers_every_byte() {
        let bytes = bytes_to_unicode();
        assert_eq!(bytes.len(), 256);
        assert_eq!(bytes[b'A' as usize], 'A');
        assert_eq!(bytes.iter().filter(|ch| **ch == '\0').count(), 0);
    }

    #[test]
    fn tiny_vocab_wraps_a_word() {
        let vocab = r#"{"<|startoftext|>":10,"<|endoftext|>":11,"a</w>":1,"a":2}"#;
        let mut tokenizer = Tokenizer::from_text(vocab, "").unwrap();
        let ids = tokenizer.encode("a");
        assert_eq!(ids.len(), 77);
        assert_eq!(ids[0], 10);
        assert_eq!(ids[1], 1);
        assert_eq!(ids[2], 11);
        assert!(ids[3..].iter().all(|id| *id == 0));
    }
}
