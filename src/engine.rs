//! Folder scan, per-file OCR and embeddings, and the background worker.
//!
//! The worker thread is the only writer and the only place ONNX sessions run.
//! Search requests are handled between files so the window stays responsive.

use crate::clip::Clip;
use crate::config::{self, Config};
use crate::db::{Library, ScanFile};
use crate::embed::Store;
use crate::imageutil::{self, Decoded};
use crate::models;
use crate::ocr::Ocr;
use crate::runtime;
use ort::environment::Environment;
use crate::search::{self, Hit, Mode};
use crate::video;
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use walkdir::WalkDir;

#[derive(Clone, Debug)]
pub struct Progress {
    pub done: u32,
    pub total: u32,
    pub path: String,
    pub paused: bool,
}

#[derive(Clone, Debug)]
pub struct Job {
    pub root_id: i64,
    pub pending: Vec<i64>,
    pub scanned: usize,
}

#[derive(Clone, Debug)]
pub struct ReadyInfo {
    pub data_dir: PathBuf,
    pub model_dir: PathBuf,
    pub db_path: PathBuf,
    pub notes: Vec<String>,
    pub ocr: bool,
    pub clip: bool,
    pub runtime: String,
    pub cuda: bool,
}

pub struct Engine {
    cfg: Config,
    db: Library,
    env: Option<Environment>,
    ocr: Option<Ocr>,
    clip: Option<Clip>,
    store: Option<Store>,
    notes: Vec<String>,
    ocr_stamp: String,
    clip_stamp: String,
    /// A failed CUDA session must not be followed by another session in-process.
    frozen: bool,
}

impl Engine {
    pub fn open(cfg: Config) -> Result<Self> {
        cfg.ensure_dirs()?;
        let db = Library::open(&cfg.db_path())?;
        let mut engine = Self {
            cfg,
            db,
            env: None,
            ocr: None,
            clip: None,
            store: None,
            notes: Vec::new(),
            ocr_stamp: String::new(),
            clip_stamp: String::new(),
            frozen: false,
        };
        engine.load_models();
        Ok(engine)
    }

    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    pub fn ready_info(&self) -> ReadyInfo {
        ReadyInfo {
            data_dir: self.cfg.data_dir.clone(),
            model_dir: self.cfg.model_dir.clone(),
            db_path: self.cfg.db_path(),
            notes: self.notes.clone(),
            ocr: self.ocr.is_some(),
            clip: self.clip.is_some(),
            runtime: runtime::runtime_label(),
            cuda: self.cfg.cuda,
        }
    }

    pub fn begin(&mut self, folder: &Path) -> Result<Job> {
        if !folder.is_dir() {
            return Err(anyhow!("not a folder: {}", folder.display()));
        }
        let folder = folder.canonicalize().unwrap_or_else(|_| folder.to_path_buf());
        let root_id = self.db.add_root(&folder.to_string_lossy())?;
        let mut files = Vec::new();
        let mut videos = Vec::new();
        let frames_root = std::fs::canonicalize(self.cfg.frames_dir()).unwrap_or_else(|_| self.cfg.frames_dir());
        let thumb_root = std::fs::canonicalize(self.cfg.thumb_dir()).unwrap_or_else(|_| self.cfg.thumb_dir());
        let walker = WalkDir::new(&folder).follow_links(false).into_iter().filter_entry(|entry| {
            if entry.depth() != 0 && entry.file_name().to_string_lossy().starts_with('.') {
                return false;
            }
            let path = entry.path();
            !path.starts_with(&frames_root) && !path.starts_with(&thumb_root)
        });
        for entry in walker {
            let Ok(entry) = entry else { continue };
            if !entry.file_type().is_file() {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if imageutil::is_image(entry.path()) {
                let rel = entry.path().strip_prefix(&folder).unwrap_or(entry.path());
                files.push(ScanFile {
                    path: entry.path().to_string_lossy().into_owned(),
                    rel: rel.to_string_lossy().into_owned(),
                    size: i64::try_from(meta.len()).unwrap_or(i64::MAX),
                    mtime_ns: imageutil::mtime_ns(&meta),
                });
            } else if video::is_video(entry.path()) {
                videos.push((entry.path().to_path_buf(), meta.len(), imageutil::mtime_ns(&meta)));
            }
        }
        if !videos.is_empty() {
            let mut sampled = 0usize;
            let mut missing_ffmpeg = false;
            for (path, size, mtime_ns) in videos {
                match video::cached_frames(&path, &folder, &frames_root, size, mtime_ns) {
                    Ok(frames) => {
                        sampled += frames.len();
                        files.extend(frames);
                    }
                    Err(err) if err.to_string() == video::FFMPEG_MISSING => missing_ffmpeg = true,
                    Err(err) => self.note(format!("{}: {err}", path.display())),
                }
            }
            if missing_ffmpeg {
                self.note("ffmpeg is not usable. Videos were not sampled.");
            }
            if sampled > 0 {
                self.note(format!("sampled {sampled} video frames"));
            }
        }
        let scanned = files.len();
        self.db.rescan(root_id, &files)?;
        if self.ocr.is_some() {
            self.db.invalidate_ocr(&self.ocr_stamp)?;
        }
        if self.clip.is_some() {
            self.db.invalidate_embed(&self.clip_stamp)?;
        }
        let pending = self.db.pending(Some(root_id), self.ocr.is_some(), self.clip.is_some())?;
        Ok(Job { root_id, pending, scanned })
    }

    pub fn index_one(&mut self, id: i64) -> Result<String> {
        let row = self.db.get(id)?;
        let path = row.path.clone();
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(err) => {
                self.db.mark_failed(id, &err.to_string())?;
                return Ok(path);
            }
        };
        let hash = blake3::hash(&bytes).to_hex().to_string();
        if let Some(old) = self.db.absent_hash(&hash, id)? {
            self.adopt(id, &old)?;
        }
        let row = self.db.get(id)?;
        let need_meta = row.hash.as_deref() != Some(hash.as_str()) || row.width.is_none();
        let need_ocr = self.ocr.is_some() && row.ocr_status != "done";
        let need_embed = self.clip.is_some() && self.store.is_some() && row.embed_status != "done";
        if !need_meta && !need_ocr && !need_embed {
            return Ok(path);
        }
        let mut decoded = match imageutil::decode(&bytes) {
            Ok(decoded) => decoded,
            Err(err) => {
                let message = err.to_string();
                if need_meta {
                    self.db.mark_failed(id, &message)?;
                } else {
                    if need_ocr {
                        self.db.set_ocr_error(id, &message)?;
                    }
                    if need_embed {
                        self.db.set_embed_error(id, &message)?;
                    }
                }
                return Ok(path);
            }
        };
        if let Some(note) = video::source_note(Path::new(&path)) {
            decoded.exif_json = merge_exif(&decoded.exif_json, &note);
        }
        if need_meta {
            if let Err(err) = self.write_thumb(id, &hash, &decoded) {
                self.db.mark_failed(id, &err.to_string())?;
                return Ok(path);
            }
        }
        if need_ocr {
            let stamp = self.ocr_stamp.clone();
            match self.ocr.as_mut().unwrap().read(&decoded.rgb) {
                Ok(text) => self.db.set_ocr(id, &text, &stamp)?,
                Err(err) => self.db.set_ocr_error(id, &err.to_string())?,
            }
        }
        if need_embed {
            let stamp = self.clip_stamp.clone();
            match self.clip.as_mut().unwrap().image(&decoded.rgb) {
                Ok(vector) => {
                    if let Some(store) = self.store.as_mut() {
                        store.put(id, &vector)?;
                    }
                    self.db.set_embed_done(id, &stamp)?;
                }
                Err(err) => self.db.set_embed_error(id, &err.to_string())?,
            }
        }
        Ok(path)
    }

    pub fn finish_root(&mut self, root_id: i64) -> Result<()> {
        let absent = self.db.absent_ids(root_id)?;
        for id in absent {
            self.forget_file(id)?;
        }
        Ok(())
    }

    pub fn index_folder(&mut self, folder: &Path) -> Result<()> {
        let noted = self.notes.len();
        let job = self.begin(folder)?;
        for note in self.notes.iter().skip(noted) {
            eprintln!("{note}");
        }
        let total = job.pending.len();
        eprintln!("scanned {} images, {total} to update", job.scanned);
        for (index, id) in job.pending.iter().enumerate() {
            let path = self.index_one(*id)?;
            eprintln!("[{}/{}] {path}", index + 1, total);
        }
        self.finish_root(job.root_id)?;
        Ok(())
    }

    pub fn search_text(&self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        let rows = self.db.search_fts(&search::fts_query(query), limit)?;
        Ok(rows.into_iter().map(hit_from_text).collect())
    }

    pub fn search_semantic(&mut self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let vector = {
            let clip = self.clip.as_mut().ok_or_else(|| anyhow!("CLIP is not loaded"))?;
            clip.text(query)?
        };
        self.neighbors(&vector, limit, None)
    }

    pub fn search_hybrid(&mut self, query: &str, limit: usize) -> Result<Vec<Hit>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let literal = self.search_text(query, 100)?;
        let semantic = match self.search_semantic(query, 100) {
            Ok(hits) => hits,
            Err(err) if self.clip.is_none() => {
                let _ = err;
                Vec::new()
            }
            Err(err) => return Err(err),
        };
        if semantic.is_empty() {
            return Ok(literal.into_iter().take(limit).collect());
        }
        let mut fused = search::fuse(&literal, &semantic);
        fused.truncate(limit);
        Ok(fused)
    }

    pub fn similar_path(&mut self, path: &Path, limit: usize) -> Result<Vec<Hit>> {
        let bytes = std::fs::read(path).map_err(|err| anyhow!("reading {}: {err}", path.display()))?;
        let decoded = imageutil::decode(&bytes)?;
        let vector = {
            let clip = self.clip.as_mut().ok_or_else(|| anyhow!("CLIP is not loaded"))?;
            clip.image(&decoded.rgb)?
        };
        let exclude = self.lookup_path(path);
        self.neighbors(&vector, limit, exclude)
    }

    pub fn similar_id(&mut self, id: i64, limit: usize) -> Result<Vec<Hit>> {
        let row = self.db.get(id)?;
        let mut hits = self.similar_path(Path::new(&row.path), limit + 1)?;
        hits.retain(|hit| hit.id != id);
        hits.truncate(limit);
        Ok(hits)
    }

    pub fn fetch_models(&mut self) -> Result<Vec<String>> {
        let mut messages = Vec::new();
        let dir = self.cfg.model_dir.clone();
        models::fetch(&dir, |msg| messages.push(msg.to_string()))?;
        let before = self.notes.len();
        if self.frozen {
            self.note("Models are on disk. Restart the app to load them.");
        } else {
            self.load_models();
        }
        messages.extend(self.notes[before..].iter().cloned());
        Ok(messages)
    }

    pub fn retry_errors(&mut self) -> Result<usize> {
        self.db.retry_errors()
    }

    pub fn set_det_limit(&mut self, limit: u32) {
        self.cfg.det_limit = limit.clamp(32, 4096);
        if let Some(ocr) = self.ocr.as_mut() {
            ocr.set_det_limit(self.cfg.det_limit);
        }
    }

    pub fn set_cuda(&mut self, cuda: bool) -> Result<()> {
        if cuda == self.cfg.cuda {
            return Ok(());
        }
        if !cuda {
            self.cfg.cuda = false;
            self.note("CPU mode starts on the next launch. This process keeps its current sessions.");
            return Ok(());
        }
        if self.frozen {
            self.note("Restart the app before switching to CUDA.");
            return Ok(());
        }
        if self.env.is_none() {
            self.cfg.cuda = true;
            self.load_models();
            return Ok(());
        }
        self.frozen = true;
        runtime::preload_cuda();
        let env = self.env.clone().expect("env checked");
        let dir = self.cfg.model_dir.clone();
        let status = models::status(&dir);
        let ocr = if status.ocr {
            match Ocr::load(&env, &dir, true, self.cfg.det_limit) {
                Ok(ocr) => Some(ocr),
                Err(err) => {
                    self.note(format!("CUDA OCR failed: {err}. Kept the current sessions. Restart to change device."));
                    return Ok(());
                }
            }
        } else {
            None
        };
        let clip = if status.clip {
            match Clip::load(&env, &dir, true) {
                Ok(clip) => Some(clip),
                Err(err) => {
                    self.note(format!("CUDA CLIP failed: {err}. Kept the current sessions. Restart to change device."));
                    return Ok(());
                }
            }
        } else {
            None
        };
        if let Some(ocr) = ocr {
            self.ocr_stamp = Ocr::stamp(&dir);
            self.db.invalidate_ocr(&self.ocr_stamp)?;
            self.ocr = Some(ocr);
        }
        if let Some(clip) = clip {
            self.attach_store(clip)?;
        }
        self.cfg.cuda = true;
        self.frozen = false;
        self.note("Switched inference to CUDA.");
        Ok(())
    }

    pub fn remove_root(&mut self, id: i64) -> Result<()> {
        let ids = self.db.file_ids_for_root(id)?;
        for file_id in ids {
            self.forget_file(file_id)?;
        }
        self.db.delete_root(id)?;
        Ok(())
    }

    pub fn duplicates(&self, threshold: f32, limit_groups: usize) -> Result<Vec<Hit>> {
        let threshold = if threshold.is_finite() {
            threshold.clamp(0.0, 1.0)
        } else {
            0.95
        };
        let groups = self
            .store
            .as_ref()
            .ok_or_else(|| anyhow!("CLIP is not loaded"))?
            .groups(threshold);
        let mut hits = Vec::new();
        let mut shown = 0usize;
        for group in groups {
            if limit_groups == 0 || shown >= limit_groups {
                break;
            }
            let mut members = Vec::new();
            for id in group.ids {
                let Ok(row) = self.db.get(id) else { continue };
                if row.present {
                    members.push(row);
                }
            }
            if members.len() < 2 {
                continue;
            }
            shown += 1;
            let snippet = format!("group {shown} · {:.3}", group.score);
            for row in members {
                hits.push(Hit {
                    id: row.id,
                    path: row.path,
                    rel: row.rel,
                    thumb: row.thumb,
                    snippet: snippet.clone(),
                    score: group.score,
                    semantic: true,
                    width: row.width,
                    height: row.height,
                });
            }
        }
        Ok(hits)
    }

    pub fn save_search(&mut self, name: &str, mode: Mode, query: &str) -> Result<()> {
        self.db.save_search(name, mode.key(), query)
    }

    pub fn delete_saved_search(&mut self, id: i64) -> Result<()> {
        self.db.delete_saved_search(id)
    }

    pub fn add_to_collection(&mut self, name: &str, file_ids: &[i64]) -> Result<i64> {
        self.db.add_to_collection(name, file_ids)
    }

    pub fn add_paths_to_collection(&mut self, name: &str, paths: &[PathBuf]) -> Result<i64> {
        let mut ids = Vec::with_capacity(paths.len());
        for path in paths {
            let id = self
                .lookup_path(path)
                .ok_or_else(|| anyhow!("not indexed: {}", path.display()))?;
            ids.push(id);
        }
        self.db.add_to_collection(name, &ids)
    }

    pub fn delete_collection(&mut self, id: i64) -> Result<()> {
        self.db.delete_collection(id)
    }

    pub fn remove_from_collection(&mut self, collection_id: i64, file_id: i64) -> Result<usize> {
        self.db.remove_from_collection(collection_id, file_id)
    }

    pub fn remove_paths_from_collection(&mut self, name: &str, paths: &[PathBuf]) -> Result<i64> {
        let id = self
            .db
            .collection_id(name)?
            .ok_or_else(|| anyhow!("no collection named {name}"))?;
        for path in paths {
            let file_id = self
                .lookup_path(path)
                .ok_or_else(|| anyhow!("not indexed: {}", path.display()))?;
            self.db.remove_from_collection(id, file_id)?;
        }
        Ok(id)
    }

    pub fn collection_id(&self, name: &str) -> Result<Option<i64>> {
        self.db.collection_id(name)
    }

    pub fn collection_items(&self, id: i64) -> Result<Vec<crate::db::Item>> {
        self.db.collection_items(id)
    }

    pub fn shelf(&self) -> Result<Shelf> {
        Ok(Shelf {
            searches: self.db.saved_searches()?,
            collections: self.db.collections()?,
        })
    }

    pub fn roots(&self) -> Result<Vec<(i64, String)>> {
        self.db.roots()
    }

    fn forget_file(&mut self, id: i64) -> Result<()> {
        let path = self.db.get(id).ok().map(|row| PathBuf::from(row.path));
        if let Some(store) = self.store.as_mut() {
            store.clear(id)?;
        }
        self.db.delete_file(id)?;
        if let Some(path) = path {
            video::remove_cached_frame(&self.cfg.data_dir, &path);
        }
        Ok(())
    }

    fn write_thumb(&mut self, id: i64, hash: &str, decoded: &Decoded) -> Result<()> {
        let prefix = &hash[..2.min(hash.len())];
        let rel = format!("{prefix}/{hash}.jpg");
        let dest = self.cfg.thumb_dir().join(prefix).join(format!("{hash}.jpg"));
        if !dest.is_file() {
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let thumb = imageutil::fit(&decoded.rgb, 384);
            let bytes = imageutil::encode_jpeg(&thumb, 82)?;
            std::fs::write(&dest, bytes)?;
        }
        self.db.set_decoded(
            id,
            hash,
            decoded.rgb.width() as i64,
            decoded.rgb.height() as i64,
            &decoded.exif_json,
            &rel,
        )?;
        Ok(())
    }

    fn adopt(&mut self, id: i64, old: &crate::db::FileRow) -> Result<()> {
        if let (Some(hash), Some(width), Some(height)) = (&old.hash, old.width, old.height) {
            let thumb = old.thumb.clone().unwrap_or_default();
            let exif = old.exif_json.clone().unwrap_or_else(|| "{\"orientation\":1}".into());
            self.db.set_decoded(id, hash, width, height, &exif, &thumb)?;
        }
        if old.ocr_status == "done" {
            if let Some(body) = self.db.ocr_body(old.id)? {
                let model = old.ocr_model.clone().unwrap_or_default();
                self.db.set_ocr(id, &body, &model)?;
            }
        }
        let mut copied = false;
        if old.embed_status == "done" {
            if let Some(store) = self.store.as_mut() {
                copied = store.copy_slot(old.id, id)?;
                if copied {
                    store.clear(old.id)?;
                }
            }
        }
        if copied {
            let model = old.embed_model.clone().unwrap_or_default();
            self.db.set_embed_done(id, &model)?;
        }
        self.db.delete_file(old.id)?;
        Ok(())
    }

    fn neighbors(&mut self, vector: &[f32], limit: usize, exclude: Option<i64>) -> Result<Vec<Hit>> {
        let ranked = {
            let store = self.store.as_ref().ok_or_else(|| anyhow!("embedding store is not open"))?;
            store.search(vector, limit.max(1), exclude)
        };
        let mut hits = Vec::new();
        for (id, score) in ranked {
            let Ok(row) = self.db.get(id) else { continue };
            if !row.present {
                continue;
            }
            let snippet = self
                .db
                .ocr_body(id)?
                .unwrap_or_default()
                .chars()
                .take(120)
                .collect::<String>();
            hits.push(Hit {
                id,
                path: row.path,
                rel: row.rel,
                thumb: row.thumb,
                snippet,
                score,
                semantic: true,
                width: row.width,
                height: row.height,
            });
        }
        Ok(hits)
    }

    fn lookup_path(&self, path: &Path) -> Option<i64> {
        if let Ok(Some(id)) = self.db.id_by_path(&path.to_string_lossy()) {
            return Some(id);
        }
        let canonical = path.canonicalize().ok()?;
        self.db.id_by_path(&canonical.to_string_lossy()).ok().flatten()
    }

    fn attach_store(&mut self, clip: Clip) -> Result<()> {
        let dim = clip.dim();
        let store = Store::open(&self.cfg.embed_path(), dim)?;
        if store.rebuilt {
            self.db.reset_embed_status()?;
            self.note("Embedding file dimension changed. Images will be embedded again.");
        }
        self.clip_stamp = Clip::stamp(&self.cfg.model_dir);
        self.db.invalidate_embed(&self.clip_stamp)?;
        self.clip = Some(clip);
        self.store = Some(store);
        Ok(())
    }

    fn load_models(&mut self) {
        if self.frozen {
            self.note("ONNX sessions were left unchanged after a CUDA error. Restart the app to switch devices.");
            return;
        }
        let status = models::status(&self.cfg.model_dir);
        if !status.ocr && !status.clip {
            self.note("OCR and CLIP models are not installed. Use Download models, or run `semantix fetch-models`.");
            return;
        }
        let env = match self.env.clone() {
            Some(env) => env,
            None => match runtime::environment(self.cfg.cuda) {
                Ok(env) => {
                    if let Some(note) = runtime::acquired_note() {
                        self.note(note);
                    }
                    env
                }
                Err(err) => {
                    self.frozen = true;
                    self.note(format!(
                        "ONNX Runtime failed to load ({err}). Indexing can store files without OCR or embeddings."
                    ));
                    return;
                }
            },
        };
        self.env = Some(env.clone());
        if status.ocr && self.ocr.is_none() {
            match Ocr::load(&env, &self.cfg.model_dir, self.cfg.cuda, self.cfg.det_limit) {
                Ok(ocr) => {
                    self.ocr_stamp = Ocr::stamp(&self.cfg.model_dir);
                    let _ = self.db.invalidate_ocr(&self.ocr_stamp);
                    self.ocr = Some(ocr);
                }
                Err(err) => {
                    self.note(format!("OCR model failed to load: {err}"));
                    if self.cfg.cuda {
                        self.frozen = true;
                        self.note("CUDA was requested. This process will not open a replacement CPU session. Restart without --cuda.");
                        return;
                    }
                }
            }
        } else if !status.ocr {
            self.note("OCR models are missing (det.onnx and rec.onnx). Text search stays empty until they are downloaded.");
        }
        if self.frozen {
            return;
        }
        if status.clip && self.clip.is_none() {
            match Clip::load(&env, &self.cfg.model_dir, self.cfg.cuda) {
                Ok(clip) => {
                    if let Err(err) = self.attach_store(clip) {
                        self.note(format!("embedding store: {err}"));
                    }
                }
                Err(err) => {
                    self.note(format!("CLIP model failed to load: {err}"));
                    if self.cfg.cuda {
                        self.frozen = true;
                        self.note("CUDA was requested. Restart without --cuda. This process will not open a replacement CPU session.");
                    }
                }
            }
        } else if !status.clip {
            self.note("CLIP models are missing (vision.onnx, text.onnx, vocab.json, merges.txt). Semantic search is off.");
        }
        if status.ocr && !status.cls {
            self.note("Angle classifier is absent. Recognition continues without 180-degree correction.");
        }
        self.note(format!("ONNX Runtime: {}", runtime::runtime_label()));
    }

    fn note(&mut self, message: impl Into<String>) {
        let message = message.into();
        if !self.notes.iter().any(|existing| existing == &message) {
            self.notes.push(message);
        }
    }
}

fn hit_from_text(row: crate::db::TextHit) -> Hit {
    Hit {
        id: row.id,
        path: row.path,
        rel: row.rel,
        thumb: row.thumb,
        snippet: row.snippet,
        score: row.score,
        semantic: false,
        width: row.width,
        height: row.height,
    }
}

pub enum Cmd {
    IndexPath(PathBuf),
    IndexAll,
    RemoveRoot(i64),
    Semantic { gen: u64, query: String, mode: Mode },
    Similar { gen: u64, path: PathBuf },
    SimilarId { gen: u64, id: i64 },
    Duplicates { gen: u64, threshold: f32, limit: usize },
    SaveSearch { name: String, mode: Mode, query: String },
    DeleteSearch(i64),
    SaveCollection { name: String, ids: Vec<i64> },
    DeleteCollection(i64),
    RemoveFromCollection { collection: i64, file: i64 },
    FetchModels,
    RetryErrors,
    SetDetLimit(u32),
    SetCuda(bool),
    Shutdown,
}

#[derive(Clone, Debug)]
pub struct Shelf {
    pub searches: Vec<crate::db::SavedSearch>,
    pub collections: Vec<crate::db::Collection>,
}

pub enum Event {
    Ready(ReadyInfo),
    Progress(Progress),
    Hits { gen: u64, hits: Vec<Hit> },
    Note(String),
    Error(String),
    Idle,
    Roots(Vec<(i64, String)>),
    Shelf(Shelf),
}

pub struct Handle {
    tx: Sender<Cmd>,
    rx: Receiver<Event>,
    pause: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
}

impl Handle {
    pub fn send(&self, cmd: Cmd) {
        let _ = self.tx.send(cmd);
    }

    pub fn try_recv(&self) -> Option<Event> {
        self.rx.try_recv().ok()
    }

    pub fn set_paused(&self, paused: bool) {
        self.pause.store(paused, Ordering::Relaxed);
    }

    pub fn paused(&self) -> bool {
        self.pause.load(Ordering::Relaxed)
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

pub fn spawn(cfg: Config) -> Handle {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (ev_tx, ev_rx) = mpsc::channel();
    let pause = Arc::new(AtomicBool::new(false));
    let cancel = Arc::new(AtomicBool::new(false));
    let pause_worker = pause.clone();
    let cancel_worker = cancel.clone();
    thread::spawn(move || worker(cfg, cmd_rx, ev_tx, pause_worker, cancel_worker));
    Handle {
        tx: cmd_tx,
        rx: ev_rx,
        pause,
        cancel,
    }
}

fn worker(cfg: Config, rx: Receiver<Cmd>, ev: Sender<Event>, pause: Arc<AtomicBool>, cancel: Arc<AtomicBool>) {
    config::lower_priority();
    let mut engine = match Engine::open(cfg) {
        Ok(engine) => engine,
        Err(err) => {
            let _ = ev.send(Event::Error(format!("{err:#}")));
            return;
        }
    };
    let _ = ev.send(Event::Ready(engine.ready_info()));
    let _ = send_roots(&engine, &ev);
    send_shelf(&engine, &ev);
    let mut deferred = std::collections::VecDeque::new();
    loop {
        let cmd = if let Some(cmd) = deferred.pop_front() {
            cmd
        } else {
            match rx.recv() {
                Ok(cmd) => cmd,
                Err(_) => break,
            }
        };
        if matches!(cmd, Cmd::Shutdown) {
            break;
        }
        dispatch(&mut engine, cmd, &rx, &ev, &pause, &cancel, &mut deferred);
    }
}

fn dispatch(
    engine: &mut Engine,
    cmd: Cmd,
    rx: &Receiver<Cmd>,
    ev: &Sender<Event>,
    pause: &AtomicBool,
    cancel: &AtomicBool,
    deferred: &mut std::collections::VecDeque<Cmd>,
) {
    match cmd {
        Cmd::Shutdown => {}
        Cmd::IndexPath(path) => {
            cancel.store(false, Ordering::Relaxed);
            if let Err(err) = index_controlled(engine, std::slice::from_ref(&path), rx, ev, pause, cancel, deferred) {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
            let _ = ev.send(Event::Idle);
            let _ = send_roots(engine, ev);
            send_shelf(engine, ev);
        }
        Cmd::IndexAll | Cmd::RetryErrors => {
            if matches!(cmd, Cmd::RetryErrors) {
                match engine.retry_errors() {
                    Ok(count) => {
                        let _ = ev.send(Event::Note(format!("queued {count} failed files")));
                    }
                    Err(err) => {
                        let _ = ev.send(Event::Error(format!("{err:#}")));
                        let _ = ev.send(Event::Idle);
                        send_shelf(engine, ev);
                        return;
                    }
                }
            }
            cancel.store(false, Ordering::Relaxed);
            let roots = engine.roots().unwrap_or_default();
            let paths: Vec<PathBuf> = roots.into_iter().map(|(_, path)| PathBuf::from(path)).collect();
            if let Err(err) = index_controlled(engine, &paths, rx, ev, pause, cancel, deferred) {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
            let _ = ev.send(Event::Idle);
            let _ = send_roots(engine, ev);
            send_shelf(engine, ev);
        }
        Cmd::Semantic { gen, query, mode } => match run_search(engine, mode, &query, 100) {
            Ok(hits) => {
                let _ = ev.send(Event::Hits { gen, hits });
            }
            Err(err) => {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
        },
        Cmd::Similar { gen, path } => match engine.similar_path(&path, 48) {
            Ok(hits) => {
                let _ = ev.send(Event::Hits { gen, hits });
            }
            Err(err) => {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
        },
        Cmd::SimilarId { gen, id } => match engine.similar_id(id, 48) {
            Ok(hits) => {
                let _ = ev.send(Event::Hits { gen, hits });
            }
            Err(err) => {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
        },
        Cmd::Duplicates { gen, threshold, limit } => match engine.duplicates(threshold, limit) {
            Ok(hits) => {
                let _ = ev.send(Event::Hits { gen, hits });
            }
            Err(err) => {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
        },
        Cmd::SaveSearch { name, mode, query } => {
            if let Err(err) = engine.save_search(&name, mode, &query) {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
            send_shelf(engine, ev);
        }
        Cmd::DeleteSearch(id) => {
            if let Err(err) = engine.delete_saved_search(id) {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
            send_shelf(engine, ev);
        }
        Cmd::SaveCollection { name, ids } => {
            if let Err(err) = engine.add_to_collection(&name, &ids) {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
            send_shelf(engine, ev);
        }
        Cmd::DeleteCollection(id) => {
            if let Err(err) = engine.delete_collection(id) {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
            send_shelf(engine, ev);
        }
        Cmd::RemoveFromCollection { collection, file } => {
            if let Err(err) = engine.remove_from_collection(collection, file) {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
            send_shelf(engine, ev);
        }
        Cmd::FetchModels => {
            match engine.fetch_models() {
                Ok(messages) => {
                    for message in messages {
                        let _ = ev.send(Event::Note(message));
                    }
                }
                Err(err) => {
                    let _ = ev.send(Event::Error(format!("{err:#}")));
                }
            }
            let _ = ev.send(Event::Ready(engine.ready_info()));
            let _ = ev.send(Event::Idle);
            send_shelf(engine, ev);
        }
        Cmd::SetDetLimit(limit) => engine.set_det_limit(limit),
        Cmd::SetCuda(cuda) => {
            if let Err(err) = engine.set_cuda(cuda) {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
            let _ = ev.send(Event::Ready(engine.ready_info()));
        }
        Cmd::RemoveRoot(id) => {
            if let Err(err) = engine.remove_root(id) {
                let _ = ev.send(Event::Error(format!("{err:#}")));
            }
            let _ = send_roots(engine, ev);
            let _ = ev.send(Event::Idle);
            send_shelf(engine, ev);
        }
    }
}

fn index_controlled(
    engine: &mut Engine,
    folders: &[PathBuf],
    rx: &Receiver<Cmd>,
    ev: &Sender<Event>,
    pause: &AtomicBool,
    cancel: &AtomicBool,
    deferred: &mut std::collections::VecDeque<Cmd>,
) -> Result<()> {
    for folder in folders {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let noted = engine.notes().len();
        let job = engine.begin(folder)?;
        for note in engine.notes().iter().skip(noted) {
            let _ = ev.send(Event::Note(note.clone()));
        }
        let total = job.pending.len() as u32;
        let root_id = job.root_id;
        for (index, id) in job.pending.into_iter().enumerate() {
            while pause.load(Ordering::Relaxed) && !cancel.load(Ordering::Relaxed) {
                drain_interactive(engine, rx, ev, cancel, deferred);
                thread::sleep(Duration::from_millis(40));
            }
            if cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            drain_interactive(engine, rx, ev, cancel, deferred);
            if cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            let path = engine.index_one(id)?;
            let _ = ev.send(Event::Progress(Progress {
                done: index as u32 + 1,
                total,
                path,
                paused: false,
            }));
        }
        if !cancel.load(Ordering::Relaxed) {
            engine.finish_root(root_id)?;
        }
    }
    Ok(())
}

fn drain_interactive(
    engine: &mut Engine,
    rx: &Receiver<Cmd>,
    ev: &Sender<Event>,
    cancel: &AtomicBool,
    deferred: &mut std::collections::VecDeque<Cmd>,
) {
    loop {
        match rx.try_recv() {
            Ok(Cmd::Shutdown) => {
                cancel.store(true, Ordering::Relaxed);
                deferred.push_back(Cmd::Shutdown);
                break;
            }
            Ok(Cmd::Semantic { gen, query, mode }) => match run_search(engine, mode, &query, 100) {
                Ok(hits) => {
                    let _ = ev.send(Event::Hits { gen, hits });
                }
                Err(err) => {
                    let _ = ev.send(Event::Error(format!("{err:#}")));
                }
            },
            Ok(Cmd::Similar { gen, path }) => match engine.similar_path(&path, 48) {
                Ok(hits) => {
                    let _ = ev.send(Event::Hits { gen, hits });
                }
                Err(err) => {
                    let _ = ev.send(Event::Error(format!("{err:#}")));
                }
            },
            Ok(Cmd::SimilarId { gen, id }) => match engine.similar_id(id, 48) {
                Ok(hits) => {
                    let _ = ev.send(Event::Hits { gen, hits });
                }
                Err(err) => {
                    let _ = ev.send(Event::Error(format!("{err:#}")));
                }
            },
            Ok(Cmd::Duplicates { gen, threshold, limit }) => match engine.duplicates(threshold, limit) {
                Ok(hits) => {
                    let _ = ev.send(Event::Hits { gen, hits });
                }
                Err(err) => {
                    let _ = ev.send(Event::Error(format!("{err:#}")));
                }
            },
            Ok(Cmd::SaveSearch { name, mode, query }) => {
                if let Err(err) = engine.save_search(&name, mode, &query) {
                    let _ = ev.send(Event::Error(format!("{err:#}")));
                }
                send_shelf(engine, ev);
            }
            Ok(Cmd::DeleteSearch(id)) => {
                if let Err(err) = engine.delete_saved_search(id) {
                    let _ = ev.send(Event::Error(format!("{err:#}")));
                }
                send_shelf(engine, ev);
            }
            Ok(Cmd::SaveCollection { name, ids }) => {
                if let Err(err) = engine.add_to_collection(&name, &ids) {
                    let _ = ev.send(Event::Error(format!("{err:#}")));
                }
                send_shelf(engine, ev);
            }
            Ok(Cmd::DeleteCollection(id)) => {
                if let Err(err) = engine.delete_collection(id) {
                    let _ = ev.send(Event::Error(format!("{err:#}")));
                }
                send_shelf(engine, ev);
            }
            Ok(Cmd::RemoveFromCollection { collection, file }) => {
                if let Err(err) = engine.remove_from_collection(collection, file) {
                    let _ = ev.send(Event::Error(format!("{err:#}")));
                }
                send_shelf(engine, ev);
            }
            Ok(other) => deferred.push_back(other),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                cancel.store(true, Ordering::Relaxed);
                break;
            }
        }
    }
}

fn run_search(engine: &mut Engine, mode: Mode, query: &str, limit: usize) -> Result<Vec<Hit>> {
    match mode {
        Mode::Text => engine.search_text(query, limit),
        Mode::Semantic => engine.search_semantic(query, limit),
        Mode::Hybrid => engine.search_hybrid(query, limit),
    }
}

fn send_roots(engine: &Engine, ev: &Sender<Event>) -> Result<()> {
    ev.send(Event::Roots(engine.roots()?)).map_err(|err| anyhow!("{err}"))?;
    Ok(())
}

fn send_shelf(engine: &Engine, ev: &Sender<Event>) {
    match engine.shelf() {
        Ok(shelf) => {
            let _ = ev.send(Event::Shelf(shelf));
        }
        Err(err) => {
            let _ = ev.send(Event::Error(format!("{err:#}")));
        }
    }
}

fn merge_exif(exif_json: &str, note: &str) -> String {
    let Ok(extra) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(note) else {
        return exif_json.to_string();
    };
    let mut base = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(exif_json).unwrap_or_default();
    for (key, value) in extra {
        base.insert(key, value);
    }
    serde_json::Value::Object(base).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live_models() -> Option<PathBuf> {
        let mut dirs = Vec::new();
        if let Some(dir) = std::env::var_os("SEMANTIX_MODELS") {
            dirs.push(PathBuf::from(dir));
        }
        dirs.push(PathBuf::from("/tmp/semantix-models"));
        dirs.into_iter().find(|dir| dir.join("det.onnx").is_file() && dir.join("text.onnx").is_file())
    }

    fn render_receipt(path: &Path) -> bool {
        let script = r#"
from PIL import Image, ImageDraw, ImageFont
import sys
img = Image.new("RGB", (1200, 480), "white")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 72)
draw.text((60, 70), "RECEIPT 1042", fill=(0, 0, 0), font=font)
draw.text((60, 230), "ALICE CHEN", fill=(0, 0, 0), font=font)
img.save(sys.argv[1])
"#;
        if std::process::Command::new("python3")
            .arg("-c")
            .arg(script)
            .arg(path)
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
        {
            return true;
        }
        let Some(path_str) = path.to_str() else { return false };
        std::process::Command::new("convert")
            .args([
                "-size",
                "1200x480",
                "xc:white",
                "-font",
                "DejaVu-Sans",
                "-pointsize",
                "72",
                "-fill",
                "black",
                "-annotate",
                "+60+150",
                "RECEIPT 1042",
                "-annotate",
                "+60+300",
                "ALICE CHEN",
                path_str,
            ])
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    #[test]
    fn live_receipt_ocr_and_clip_text() {
        let Some(models) = live_models() else {
            eprintln!("skipping live model test: models not found");
            return;
        };
        if runtime::runtime_lib_dir().is_none() {
            eprintln!("skipping live model test: ONNX Runtime not found");
            return;
        }
        let data = crate::db::scratch_dir("live");
        let photos = data.join("photos");
        std::fs::create_dir_all(&photos).unwrap();
        let cfg = Config::resolve(Some(data.join("library")), Some(models), false);
        let mut engine = Engine::open(cfg).unwrap();
        assert!(engine.ocr.is_some(), "OCR failed to load: {:?}", engine.notes);
        assert!(engine.clip.is_some(), "CLIP failed to load: {:?}", engine.notes);

        let dog = engine.clip.as_mut().unwrap().text("a photo of a dog").unwrap();
        let city = engine.clip.as_mut().unwrap().text("skyscraper").unwrap();
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm(&dog) - 1.0).abs() < 1e-3, "{}", norm(&dog));
        assert!((norm(&city) - 1.0).abs() < 1e-3, "{}", norm(&city));
        let cosine = dog.iter().zip(&city).map(|(a, b)| a * b).sum::<f32>();
        assert!(cosine < 0.95, "embeddings did not differ: {cosine}");

        let receipt = photos.join("receipt.png");
        if !render_receipt(&receipt) {
            eprintln!("skipping OCR assertion: could not render a receipt");
            let _ = std::fs::remove_dir_all(&data);
            return;
        }
        let job = engine.begin(&photos).unwrap();
        assert!(!job.pending.is_empty());
        for id in &job.pending {
            engine.index_one(*id).unwrap();
        }
        engine.finish_root(job.root_id).unwrap();
        let hits = engine.search_text("1042", 10).unwrap();
        if hits.is_empty() {
            let items = engine.db.gallery(10).unwrap();
            for item in &items {
                eprintln!("ocr {:?} {:?}", item.path, engine.db.ocr_body(item.id).unwrap());
            }
            panic!("expected an OCR hit for 1042");
        }
        let named = engine.search_text("Alice", 10).unwrap();
        assert!(!named.is_empty(), "expected an OCR hit for Alice");
        let again = engine.begin(&photos).unwrap();
        assert!(again.pending.is_empty(), "second index should not redo OCR");

        let renamed = photos.join("renamed.png");
        std::fs::rename(&receipt, &renamed).unwrap();
        let moved = engine.begin(&photos).unwrap();
        for id in &moved.pending {
            engine.index_one(*id).unwrap();
        }
        engine.finish_root(moved.root_id).unwrap();
        let hits = engine.search_text("1042", 10).unwrap();
        assert!(hits.iter().any(|hit| hit.path.ends_with("renamed.png")), "{hits:?}");
        let _ = std::fs::remove_dir_all(&data);
    }
}
