//! Memory-mapped int8 embedding file.
//!
//! Header is 32 bytes: magic `SEMX`, version 1, dimension, capacity. Slot 0 is
//! unused so a SQLite file id is the slot index. Each record is a little-endian
//! f32 scale followed by `dim` int8 values. A scale of 0 means the slot is empty.
//!
//! Quantization of an L2-normalized vector stores `scale = max(abs) / 127` and
//! `q = round(x / scale)`. Search dots the float query with `q * scale`.

use anyhow::{anyhow, Result};
use memmap2::MmapMut;
use std::fs::File;
use std::path::Path;

const MAGIC: &[u8; 4] = b"SEMX";
const VERSION: u32 = 1;
const HEADER: usize = 32;

#[derive(Clone, Debug)]
pub struct Group {
    pub ids: Vec<i64>,
    pub score: f32,
}

pub struct Store {
    file: File,
    mmap: MmapMut,
    dim: u32,
    capacity: u32,
    pub rebuilt: bool,
}

impl Store {
    pub fn open(path: &Path, dim: u32) -> Result<Self> {
        if dim == 0 {
            return Err(anyhow!("embedding dimension is unknown"));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut rebuilt = false;
        if path.is_file() {
            if let Ok(existing) = read_header(path) {
                if existing.0 != dim {
                    backup(path)?;
                    rebuilt = true;
                }
            } else {
                backup(path)?;
                rebuilt = true;
            }
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let mut store = Self {
            file,
            mmap: MmapMut::map_anon(HEADER)?,
            dim,
            capacity: 0,
            rebuilt,
        };
        if store.file.metadata()?.len() < HEADER as u64 || rebuilt {
            store.create_fresh(1024)?;
        } else {
            let (file_dim, capacity) = read_header(path)?;
            if file_dim != dim {
                store.create_fresh(1024)?;
                store.rebuilt = true;
            } else {
                let bytes = HEADER + store.record_len() * capacity as usize;
                if store.file.metadata()?.len() < bytes as u64 {
                    store.file.set_len(bytes as u64)?;
                }
                store.capacity = capacity;
                store.remap()?;
            }
        }
        Ok(store)
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    pub fn put(&mut self, id: i64, vector: &[f32]) -> Result<()> {
        if id <= 0 {
            return Err(anyhow!("embedding id must be positive"));
        }
        if vector.len() != self.dim as usize {
            return Err(anyhow!("embedding dim {} does not match {}", vector.len(), self.dim));
        }
        self.ensure(id)?;
        let (scale, quantized) = quantize(vector);
        let at = self.offset(id);
        self.mmap[at..at + 4].copy_from_slice(&scale.to_le_bytes());
        for (i, value) in quantized.iter().enumerate() {
            self.mmap[at + 4 + i] = *value as u8;
        }
        self.mmap.flush()?;
        Ok(())
    }

    pub fn copy_slot(&mut self, from: i64, to: i64) -> Result<bool> {
        if from <= 0 || to <= 0 {
            return Ok(false);
        }
        self.ensure(from.max(to))?;
        let len = self.record_len();
        let src = self.offset(from);
        let bytes = self.mmap[src..src + len].to_vec();
        let scale = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if scale <= 0.0 {
            return Ok(false);
        }
        let dst = self.offset(to);
        self.mmap[dst..dst + len].copy_from_slice(&bytes);
        self.mmap.flush()?;
        Ok(true)
    }

    pub fn clear(&mut self, id: i64) -> Result<()> {
        if id <= 0 || id as u32 >= self.capacity {
            return Ok(());
        }
        let at = self.offset(id);
        let len = self.record_len();
        self.mmap[at..at + len].fill(0);
        self.mmap.flush()?;
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize, exclude: Option<i64>) -> Vec<(i64, f32)> {
        if k == 0 || query.len() != self.dim as usize {
            return Vec::new();
        }
        let mut best: Vec<(i64, f32)> = Vec::new();
        for id in 1..self.capacity as i64 {
            if exclude == Some(id) {
                continue;
            }
            let Some(score) = self.score(id, query) else {
                continue;
            };
            if best.len() < k {
                best.push((id, score));
                best.sort_by(|a, b| b.1.total_cmp(&a.1));
            } else {
                let last = best.len() - 1;
                if score > best[last].1 {
                    best[last] = (id, score);
                    best.sort_by(|a, b| b.1.total_cmp(&a.1));
                }
            }
        }
        best
    }

    /// Near-duplicate groups. Cosine is `scale_a * scale_b * sum(qa * qb)`.
    /// A group needs at least two occupied slots at or above `threshold`.
    /// `score` is the strongest link inside the group.
    pub fn groups(&self, threshold: f32) -> Vec<Group> {
        let dim = self.dim as usize;
        let mut slots: Vec<(i64, f32, Vec<i8>)> = Vec::new();
        for id in 1..self.capacity as i64 {
            let at = self.offset(id);
            let len = self.record_len();
            if at + len > self.mmap.len() {
                break;
            }
            let Ok(scale_bytes) = self.mmap[at..at + 4].try_into() else {
                continue;
            };
            let scale = f32::from_le_bytes(scale_bytes);
            if scale <= 0.0 {
                continue;
            }
            let mut quantized = Vec::with_capacity(dim);
            for i in 0..dim {
                quantized.push(self.mmap[at + 4 + i] as i8);
            }
            slots.push((id, scale, quantized));
        }
        let count = slots.len();
        let mut parent: Vec<usize> = (0..count).collect();
        let mut rank = vec![0u8; count];
        let mut best = vec![0.0f32; count];
        for left in 0..count {
            for right in (left + 1)..count {
                let score = pair_cosine(slots[left].1, &slots[left].2, slots[right].1, &slots[right].2);
                if score >= threshold {
                    unite(&mut parent, &mut rank, &mut best, left, right, score);
                }
            }
        }
        let mut buckets: Vec<Vec<i64>> = vec![Vec::new(); count];
        for (index, slot) in slots.iter().enumerate() {
            buckets[find(&mut parent, index)].push(slot.0);
        }
        let mut groups = Vec::new();
        for (root, mut ids) in buckets.into_iter().enumerate() {
            if ids.len() < 2 {
                continue;
            }
            ids.sort_unstable();
            groups.push(Group { ids, score: best[root] });
        }
        groups.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.ids[0].cmp(&b.ids[0])));
        groups
    }

    fn score(&self, id: i64, query: &[f32]) -> Option<f32> {
        let at = self.offset(id);
        if at + self.record_len() > self.mmap.len() {
            return None;
        }
        let scale = f32::from_le_bytes(self.mmap[at..at + 4].try_into().ok()?);
        if scale <= 0.0 {
            return None;
        }
        let mut dot = 0.0f32;
        for (i, component) in query.iter().enumerate() {
            let q = self.mmap[at + 4 + i] as i8 as f32;
            dot += component * q;
        }
        Some(dot * scale)
    }

    fn create_fresh(&mut self, capacity: u32) -> Result<()> {
        let bytes = HEADER + self.record_len() * capacity as usize;
        self.file.set_len(bytes as u64)?;
        self.capacity = capacity;
        self.remap()?;
        self.mmap[..HEADER].fill(0);
        self.mmap[0..4].copy_from_slice(MAGIC);
        self.mmap[4..8].copy_from_slice(&VERSION.to_le_bytes());
        self.mmap[8..12].copy_from_slice(&self.dim.to_le_bytes());
        self.mmap[12..16].copy_from_slice(&capacity.to_le_bytes());
        self.mmap.flush()?;
        Ok(())
    }

    fn ensure(&mut self, id: i64) -> Result<()> {
        if id < self.capacity as i64 {
            return Ok(());
        }
        let mut capacity = self.capacity.max(16);
        while id >= capacity as i64 {
            capacity = capacity.saturating_mul(2).max(capacity + 1);
        }
        let bytes = HEADER + self.record_len() * capacity as usize;
        self.mmap.flush()?;
        self.file.set_len(bytes as u64)?;
        self.capacity = capacity;
        self.remap()?;
        self.mmap[12..16].copy_from_slice(&capacity.to_le_bytes());
        self.mmap.flush()?;
        Ok(())
    }

    fn remap(&mut self) -> Result<()> {
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        Ok(())
    }

    fn record_len(&self) -> usize {
        4 + self.dim as usize
    }

    fn offset(&self, id: i64) -> usize {
        HEADER + self.record_len() * id as usize
    }
}

fn read_header(path: &Path) -> Result<(u32, u32)> {
    let mut file = File::open(path)?;
    let mut bytes = [0u8; HEADER];
    std::io::Read::read_exact(&mut file, &mut bytes)?;
    if &bytes[0..4] != MAGIC {
        return Err(anyhow!("not an embedding file"));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != VERSION {
        return Err(anyhow!("unsupported embedding version {version}"));
    }
    let dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let capacity = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if dim == 0 || capacity == 0 {
        return Err(anyhow!("empty embedding header"));
    }
    Ok((dim, capacity))
}

fn backup(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let bak = path.with_extension("bin.bak");
    let dest = if bak.exists() {
        path.with_extension(format!("bin.bak.{}", std::process::id()))
    } else {
        bak
    };
    std::fs::rename(path, dest)?;
    Ok(())
}

pub fn quantize(vector: &[f32]) -> (f32, Vec<i8>) {
    let max = vector.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if max < 1e-12 {
        return (0.0, vec![0; vector.len()]);
    }
    let scale = max / 127.0;
    let quantized = vector
        .iter()
        .map(|value| {
            let rounded = (value / scale).round().clamp(-127.0, 127.0);
            rounded as i8
        })
        .collect();
    (scale, quantized)
}

fn pair_cosine(scale_a: f32, a: &[i8], scale_b: f32, b: &[i8]) -> f32 {
    let mut dot: i64 = 0;
    for (left, right) in a.iter().zip(b) {
        dot += *left as i64 * *right as i64;
    }
    scale_a * scale_b * dot as f32
}

fn find(parent: &mut [usize], mut index: usize) -> usize {
    while parent[index] != index {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    index
}

fn unite(parent: &mut [usize], rank: &mut [u8], best: &mut [f32], a: usize, b: usize, score: f32) {
    let mut a = find(parent, a);
    let mut b = find(parent, b);
    if a == b {
        best[a] = best[a].max(score);
        return;
    }
    if rank[a] < rank[b] {
        std::mem::swap(&mut a, &mut b);
    }
    parent[b] = a;
    if rank[a] == rank[b] {
        rank[a] = rank[a].saturating_add(1);
    }
    best[a] = best[a].max(best[b]).max(score);
}

pub fn cosine_i8(query: &[f32], quantized: &[i8], scale: f32) -> f32 {
    let mut dot = 0.0f32;
    for (left, right) in query.iter().zip(quantized) {
        dot += left * (*right as f32);
    }
    dot * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(values: &[f32]) -> Vec<f32> {
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        values.iter().map(|v| v / norm).collect()
    }

    #[test]
    fn int8_cosine_matches_float() {
        let vector = unit(&[1.0, 0.0, -0.5, 0.25]);
        let (scale, quantized) = quantize(&vector);
        let score = cosine_i8(&vector, &quantized, scale);
        assert!(scale > 0.0);
        assert!((score - 1.0).abs() < 0.02, "{score}");
        let other = unit(&[0.0, 1.0, 0.0, 0.0]);
        let cross = cosine_i8(&other, &quantized, scale);
        assert!(cross < score);
    }

    #[test]
    fn store_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "semantix-embed-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("embeddings.bin");
        let vector = unit(&[0.2, -0.4, 0.8, 0.1]);
        {
            let mut store = Store::open(&path, 4).unwrap();
            store.put(3, &vector).unwrap();
        }
        let store = Store::open(&path, 4).unwrap();
        assert!(!store.rebuilt);
        let hits = store.search(&vector, 2, None);
        assert_eq!(hits[0].0, 3);
        assert!((hits[0].1 - 1.0).abs() < 0.02, "{hits:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn near_copies_form_one_group() {
        let dir = std::env::temp_dir().join(format!(
            "semantix-groups-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let vector = unit(&[1.0, 0.2, -0.4, 0.1]);
        let other = unit(&[0.0, 1.0, 0.0, 0.0]);
        let mut store = Store::open(&dir.join("embeddings.bin"), 4).unwrap();
        store.put(1, &vector).unwrap();
        store.put(2, &vector).unwrap();
        store.put(4, &other).unwrap();
        let groups = store.groups(0.95);
        assert_eq!(groups.len(), 1, "{groups:?}");
        assert_eq!(groups[0].ids, vec![1, 2]);
        assert!(groups[0].score >= 0.95, "{}", groups[0].score);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
