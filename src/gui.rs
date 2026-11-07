//! Gallery window. The worker owns indexing and ONNX; this thread only reads
//! SQLite and draws.

use crate::config::{self, Config};
use crate::db::{self, Detail, Item, Library, Stats};
use crate::engine::{self, Cmd, Event, Handle, Progress, Shelf};
use crate::imageutil;
use crate::search::{self, Hit, Mode};
use anyhow::{anyhow, Result};
use eframe::egui;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const SEARCH_ID: &str = "semantix-search";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Library,
    Text,
    Semantic,
    Hybrid,
    Duplicates,
}

impl Tab {
    fn label(self) -> &'static str {
        match self {
            Self::Library => "Library",
            Self::Text => "Text",
            Self::Semantic => "Semantic",
            Self::Hybrid => "Hybrid",
            Self::Duplicates => "Duplicates",
        }
    }

    fn all() -> [Tab; 5] {
        [Self::Library, Self::Text, Self::Semantic, Self::Hybrid, Self::Duplicates]
    }
}

#[derive(Clone)]
struct Card {
    id: i64,
    path: String,
    rel: String,
    thumb: Option<String>,
    snippet: String,
    score: Option<f32>,
}

struct TexCache {
    map: HashMap<i64, egui::TextureHandle>,
    order: VecDeque<i64>,
}

pub fn run(cfg: Config) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1240.0, 820.0])
            .with_title("Semantix"),
        ..Default::default()
    };
    eframe::run_native(
        "Semantix",
        options,
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(GalleryApp::new(cfg)))
        }),
    )
    .map_err(|err| anyhow!("{err}"))
}

struct GalleryApp {
    handle: Handle,
    db: Option<Library>,
    db_path: PathBuf,
    data_dir: PathBuf,
    model_dir: PathBuf,
    notes: Vec<String>,
    ocr_ready: bool,
    clip_ready: bool,
    runtime: String,
    tab: Tab,
    query: String,
    folder: String,
    cards: Vec<Card>,
    gallery: Vec<Item>,
    textures: TexCache,
    selected: Option<i64>,
    detail: Option<Detail>,
    viewer: bool,
    viewer_tex: Option<egui::TextureHandle>,
    viewer_for: Option<i64>,
    progress: Option<Progress>,
    running: bool,
    gen: u64,
    query_at: Instant,
    query_dirty: bool,
    showing_hits: bool,
    grid: bool,
    stats: Stats,
    roots: Vec<(i64, String)>,
    error: Option<String>,
    det_limit: u32,
    cols: usize,
    last_refresh: Instant,
    searches: Vec<db::SavedSearch>,
    collections: Vec<db::Collection>,
    shelf_name: String,
    open_collection: Option<i64>,
    view_note: Option<String>,
    dup_threshold: f32,
    show_frames: bool,
    folder_filter: Option<i64>,
}

impl GalleryApp {
    fn new(cfg: Config) -> Self {
        let db_path = cfg.db_path();
        let data_dir = cfg.data_dir.clone();
        let model_dir = cfg.model_dir.clone();
        Self {
            handle: engine::spawn(cfg),
            db: None,
            db_path,
            data_dir,
            model_dir,
            notes: Vec::new(),
            ocr_ready: false,
            clip_ready: false,
            runtime: String::new(),
            tab: Tab::Library,
            query: String::new(),
            folder: String::new(),
            cards: Vec::new(),
            gallery: Vec::new(),
            textures: TexCache { map: HashMap::new(), order: VecDeque::new() },
            selected: None,
            detail: None,
            viewer: false,
            viewer_tex: None,
            viewer_for: None,
            progress: None,
            running: false,
            gen: 0,
            query_at: Instant::now(),
            query_dirty: false,
            showing_hits: false,
            grid: true,
            stats: Stats::default(),
            roots: Vec::new(),
            error: None,
            det_limit: 960,
            cols: 1,
            last_refresh: Instant::now(),
            searches: Vec::new(),
            collections: Vec::new(),
            shelf_name: String::new(),
            open_collection: None,
            view_note: None,
            dup_threshold: 0.95,
            show_frames: false,
            folder_filter: None,
        }
    }

    fn pump(&mut self) {
        while let Some(event) = self.handle.try_recv() {
            match event {
                Event::Ready(info) => {
                    self.notes = info.notes;
                    self.ocr_ready = info.ocr;
                    self.clip_ready = info.clip;
                    self.runtime = info.runtime;
                    self.data_dir = info.data_dir;
                    self.model_dir = info.model_dir;
                    self.db_path = info.db_path;
                    self.db = Library::open_read(&self.db_path).ok();
                    self.reload();
                    if !self.showing_hits {
                        self.rebuild_cards();
                    }
                }
                Event::Progress(progress) => {
                    self.running = true;
                    self.progress = Some(progress);
                    if self.last_refresh.elapsed() > Duration::from_millis(400) {
                        self.reload();
                        if !self.showing_hits {
                            self.rebuild_cards();
                        }
                        self.last_refresh = Instant::now();
                    }
                }
                Event::Hits { gen, hits } => {
                    if gen == self.gen && matches!(self.tab, Tab::Semantic | Tab::Hybrid | Tab::Duplicates) {
                        self.cards = hits.into_iter().map(card_from_hit).collect();
                        self.showing_hits = true;
                    }
                }
                Event::Note(note) => {
                    if !self.notes.iter().any(|existing| existing == &note) {
                        self.notes.push(note);
                    }
                }
                Event::Error(message) => self.error = Some(message),
                Event::Idle => {
                    self.running = false;
                    self.reload();
                    if !self.showing_hits {
                        self.rebuild_cards();
                    }
                }
                Event::Roots(roots) => self.roots = roots,
                Event::Shelf(shelf) => {
                    self.apply_shelf(shelf);
                    self.refresh_open_collection();
                }
            }
        }
    }

    fn apply_shelf(&mut self, shelf: Shelf) {
        self.searches = shelf.searches;
        self.collections = shelf.collections;
    }

    fn leave_special_view(&mut self) {
        self.open_collection = None;
        self.view_note = None;
    }

    fn refresh_open_collection(&mut self) {
        let Some(id) = self.open_collection else { return };
        if !self.collections.iter().any(|collection| collection.id == id) {
            self.leave_special_view();
            self.showing_hits = false;
            self.rebuild_cards();
            return;
        }
        self.reload_collection(id);
    }

    fn reload(&mut self) {
        if self.db.is_none() {
            self.db = Library::open_read(&self.db_path).ok();
        }
        let Some(db) = self.db.as_ref() else { return };
        self.gallery = db.gallery(20_000).unwrap_or_default();
        self.stats = db.stats().unwrap_or_default();
        if let Some(id) = self.selected {
            self.detail = db.detail(id).ok().flatten();
        }
    }

    fn rebuild_cards(&mut self) {
        match self.tab {
            Tab::Library => {
                let needle = self.query.to_lowercase();
                self.cards = self
                    .gallery
                    .iter()
                    .filter(|item| self.visible_in_library(item))
                    .filter(|item| {
                        needle.is_empty()
                            || item.path.to_lowercase().contains(&needle)
                            || item.rel.to_lowercase().contains(&needle)
                    })
                    .cloned()
                    .map(card_from_item)
                    .collect();
            }
            Tab::Text => {
                if self.query.trim().is_empty() {
                    self.cards = self
                        .gallery
                        .iter()
                        .filter(|item| self.visible_in_library(item))
                        .cloned()
                        .map(card_from_item)
                        .collect();
                } else if let Some(db) = self.db.as_ref() {
                    match db.search_fts(&search::fts_query(&self.query), 200) {
                        Ok(hits) => {
                            self.error = None;
                            self.cards = hits.into_iter().map(card_from_text).collect();
                        }
                        Err(err) => self.error = Some(err.to_string()),
                    }
                }
            }
            Tab::Semantic | Tab::Hybrid => {
                if self.query.trim().is_empty() && !self.showing_hits {
                    self.cards = self
                        .gallery
                        .iter()
                        .filter(|item| self.visible_in_library(item))
                        .cloned()
                        .map(card_from_item)
                        .collect();
                }
            }
            Tab::Duplicates => {}
        }
    }

    fn show_collection(&mut self, id: i64) {
        let name = self
            .collections
            .iter()
            .find(|collection| collection.id == id)
            .map(|collection| collection.name.clone());
        self.open_collection = Some(id);
        self.view_note = name.map(|name| format!("Collection · {name}"));
        self.tab = Tab::Library;
        self.viewer = false;
        self.reload_collection(id);
    }

    fn reload_collection(&mut self, id: i64) {
        let Some(db) = self.db.as_ref() else { return };
        let Ok(items) = db.collection_items(id) else { return };
        self.cards = items.into_iter().map(card_from_item).collect();
        self.showing_hits = true;
        if self.selected.is_some_and(|selected| !self.cards.iter().any(|card| card.id == selected)) {
            self.selected = None;
            self.detail = None;
            self.viewer = false;
            self.viewer_for = None;
        }
    }

    fn visible_in_library(&self, item: &Item) -> bool {
        if self.folder_filter.is_some_and(|id| item.root_id != id) {
            return false;
        }
        self.show_frames || !frame_rel(&item.rel)
    }

    fn select_folder(&mut self, id: i64) {
        self.folder_filter = if self.folder_filter == Some(id) { None } else { Some(id) };
        self.tab = Tab::Library;
        self.showing_hits = false;
        self.query_dirty = false;
        self.leave_special_view();
        self.rebuild_cards();
    }

    fn show_clip(&mut self, source: &str) {
        let Some(db) = self.db.as_ref() else { return };
        let Ok(frames) = db.clip_frames(source) else { return };
        if frames.is_empty() {
            return;
        }
        self.cards = frames.into_iter().map(card_from_frame).collect();
        self.showing_hits = true;
        self.open_collection = None;
        self.view_note = Some(format!("Clip · {source}"));
        self.tab = Tab::Library;
        self.query_dirty = false;
        self.viewer = false;
    }

    fn fire_duplicates(&mut self) {
        self.gen += 1;
        self.showing_hits = true;
        self.query_dirty = false;
        self.leave_special_view();
        self.cards.clear();
        self.handle.send(Cmd::Duplicates {
            gen: self.gen,
            threshold: self.dup_threshold,
            limit: 80,
        });
    }

    fn run_saved(&mut self, mode: &str, query: &str) {
        let Some(mode) = Mode::parse(mode) else { return };
        self.query = query.to_string();
        self.tab = match mode {
            Mode::Text => Tab::Text,
            Mode::Semantic => Tab::Semantic,
            Mode::Hybrid => Tab::Hybrid,
        };
        self.viewer = false;
        self.leave_special_view();
        match mode {
            Mode::Text => {
                self.showing_hits = false;
                self.rebuild_cards();
            }
            Mode::Semantic | Mode::Hybrid => self.fire_search(),
        }
    }

    fn fire_search(&mut self) {
        let mode = match self.tab {
            Tab::Semantic => Mode::Semantic,
            Tab::Hybrid => Mode::Hybrid,
            _ => return,
        };
        self.gen += 1;
        self.showing_hits = true;
        self.query_dirty = false;
        self.handle.send(Cmd::Semantic {
            gen: self.gen,
            query: self.query.clone(),
            mode,
        });
    }

    fn load_detail(&mut self, id: i64) {
        self.selected = Some(id);
        if let Some(db) = self.db.as_ref() {
            self.detail = db.detail(id).ok().flatten();
        }
    }

    fn open_frame(&mut self, id: i64) {
        self.load_detail(id);
        if self.viewer {
            self.viewer_for = None;
        }
    }

    /// Step through samples of the current video. Returns false when this picture is not a frame.
    fn clip_step(&mut self, delta: isize) -> bool {
        let Some(detail) = self.detail.clone() else { return false };
        let Some(source) = frame_source(&detail.exif_json) else { return false };
        let frames = {
            let Some(db) = self.db.as_ref() else { return false };
            match db.clip_frames(&source) {
                Ok(frames) if frames.len() > 1 => frames,
                _ => return false,
            }
        };
        let Some(pos) = frames.iter().position(|frame| frame.id == detail.id) else {
            return false;
        };
        let next = (pos as isize + delta).clamp(0, frames.len() as isize - 1) as usize;
        let id = frames[next].id;
        if id != detail.id {
            self.open_frame(id);
        }
        true
    }

    fn move_selection(&mut self, delta: isize) {
        if self.cards.is_empty() {
            return;
        }
        let current = self
            .selected
            .and_then(|id| self.cards.iter().position(|card| card.id == id))
            .unwrap_or(0);
        let next = (current as isize + delta).clamp(0, self.cards.len() as isize - 1) as usize;
        let id = self.cards[next].id;
        self.load_detail(id);
        if self.viewer {
            self.viewer_for = None;
        }
    }

    fn dropped(&mut self, ctx: &egui::Context) {
        let files = ctx.input(|input| input.raw.dropped_files.clone());
        for file in files {
            let path = file.path().to_path_buf();
            if path.as_os_str().is_empty() {
                continue;
            }
            if path.is_dir() {
                self.folder = path.display().to_string();
                self.running = true;
                self.handle.send(Cmd::IndexPath(path));
            } else if imageutil::is_image(&path) {
                self.gen += 1;
                self.tab = Tab::Semantic;
                self.showing_hits = true;
                self.leave_special_view();
                self.handle.send(Cmd::Similar { gen: self.gen, path });
            }
        }
    }

    fn texture(&mut self, ctx: &egui::Context, id: i64, rel: &Option<String>) -> Option<egui::TextureId> {
        if let Some(texture) = self.textures.map.get(&id) {
            return Some(texture.id());
        }
        let rel = rel.as_ref()?;
        let bytes = std::fs::read(self.data_dir.join("thumbs").join(rel)).ok()?;
        let image = image::load_from_memory(&bytes).ok()?.into_rgb8();
        let color = egui::ColorImage::from_rgb([image.width() as usize, image.height() as usize], image.as_raw());
        let handle = ctx.load_texture(format!("thumb-{id}"), color, egui::TextureOptions::LINEAR);
        let texture_id = handle.id();
        self.textures.order.push_back(id);
        self.textures.map.insert(id, handle);
        while self.textures.order.len() > 180 {
            if let Some(old) = self.textures.order.pop_front() {
                self.textures.map.remove(&old);
            }
        }
        Some(texture_id)
    }

    fn ensure_viewer(&mut self, ctx: &egui::Context) {
        let Some(id) = self.selected else { return };
        if self.viewer_for == Some(id) {
            return;
        }
        let path = self
            .detail
            .as_ref()
            .filter(|detail| detail.id == id)
            .map(|detail| detail.path.clone())
            .or_else(|| self.cards.iter().find(|card| card.id == id).map(|card| card.path.clone()));
        let Some(path) = path else { return };
        let Ok(bytes) = std::fs::read(&path) else { return };
        let Ok(decoded) = imageutil::decode(&bytes) else { return };
        let fitted = imageutil::fit(&decoded.rgb, 1500);
        let color = egui::ColorImage::from_rgb([fitted.width() as usize, fitted.height() as usize], fitted.as_raw());
        self.viewer_tex = Some(ctx.load_texture(format!("view-{id}"), color, egui::TextureOptions::LINEAR));
        self.viewer_for = Some(id);
    }
}

impl eframe::App for GalleryApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.pump();
        self.dropped(ctx);
        let semantic = matches!(self.tab, Tab::Semantic | Tab::Hybrid);
        if self.query_dirty && semantic {
            if self.query.trim().is_empty() {
                self.query_dirty = false;
                self.showing_hits = false;
                self.rebuild_cards();
            } else if self.query_at.elapsed() >= Duration::from_millis(250) {
                self.fire_search();
            } else {
                ctx.request_repaint_after(Duration::from_millis(80));
            }
        }
        if self.running {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if ui.input_mut(|input| input.consume_key(egui::Modifiers::COMMAND, egui::Key::F)) {
            ui.memory_mut(|memory| memory.request_focus(egui::Id::unique(SEARCH_ID)));
        }
        let frame = egui::Frame::central_panel(ui.style()).inner_margin(egui::Margin::same(12));
        egui::CentralPanel::default().frame(frame).show(ui, |ui| {
            self.banner(ui);
            self.toolbar(ui);
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                for tab in Tab::all() {
                    if ui.selectable_label(self.tab == tab, tab.label()).clicked() {
                        let changed = self.tab != tab;
                        if tab == Tab::Duplicates {
                            self.tab = tab;
                            self.query_at = Instant::now();
                            self.fire_duplicates();
                        } else if changed {
                            self.tab = tab;
                            self.query_at = Instant::now();
                            self.showing_hits = false;
                            self.leave_special_view();
                            self.query_dirty = matches!(tab, Tab::Semantic | Tab::Hybrid) && !self.query.trim().is_empty();
                            self.rebuild_cards();
                        }
                    }
                }
                ui.separator();
                if ui.selectable_label(self.grid, "Grid").clicked() {
                    self.grid = true;
                }
                if ui.selectable_label(!self.grid, "List").clicked() {
                    self.grid = false;
                }
                if ui.selectable_label(self.show_frames, "Frames").clicked() {
                    self.show_frames = !self.show_frames;
                    if !self.showing_hits {
                        self.rebuild_cards();
                    }
                }
            });
            let hint = match self.tab {
                Tab::Library => "Filter paths",
                Tab::Text => "Receipt number, name, or phrase",
                Tab::Semantic => "Describe an image",
                Tab::Hybrid => "Text and meaning",
                Tab::Duplicates => "Near duplicates use embeddings",
            };
            let mut query = self.query.clone();
            let search = ui.add(
                egui::TextEdit::singleline(&mut query)
                    .id(egui::Id::unique(SEARCH_ID))
                    .hint_text(hint)
                    .desired_width(f32::INFINITY),
            );
            if search.changed() {
                self.query = query;
                self.query_at = Instant::now();
                if !matches!(self.tab, Tab::Duplicates) {
                    self.query_dirty = true;
                    self.showing_hits = false;
                    self.leave_special_view();
                }
                if matches!(self.tab, Tab::Library | Tab::Text) {
                    self.rebuild_cards();
                }
            } else {
                self.query = query;
            }
            let focused = search.has_focus();
            if search.lost_focus()
                && ui.input(|input| input.key_pressed(egui::Key::Enter))
                && matches!(self.tab, Tab::Semantic | Tab::Hybrid)
                && !self.query.trim().is_empty()
            {
                self.fire_search();
            }
            if self.tab == Tab::Duplicates {
                let response = ui.add(
                    egui::Slider::new(&mut self.dup_threshold, 0.80..=0.99)
                        .text("cosine")
                        .fixed_decimals(2),
                );
                let commit = response.drag_stopped() || (response.changed() && !response.dragged());
                if commit {
                    self.fire_duplicates();
                }
            }
            ui.add_space(4.0);
            ui.horizontal_top(|ui| {
                let side = 320.0;
                let grid_width = (ui.available_width() - side - 8.0).max(200.0);
                ui.allocate_ui_with_layout(
                    egui::vec2(grid_width, ui.available_height()),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        if self.viewer {
                            self.viewer_ui(ui);
                        } else {
                            self.browser(ui);
                        }
                    },
                );
                ui.separator();
                ui.allocate_ui_with_layout(
                    egui::vec2(side, ui.available_height()),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| self.side(ui),
                );
            });
            self.keys(ui, focused);
        });
    }
}

impl GalleryApp {
    fn banner(&mut self, ui: &mut egui::Ui) {
        let missing = !self.ocr_ready || !self.clip_ready;
        if self.notes.is_empty() && self.error.is_none() && !missing {
            return;
        }
        egui::Frame::NONE
            .fill(egui::Color32::from_rgb(48, 36, 20))
            .inner_margin(egui::Margin::same(8))
            .corner_radius(6.0)
            .show(ui, |ui| {
                if let Some(error) = &self.error {
                    ui.colored_label(egui::Color32::from_rgb(255, 160, 120), error);
                }
                for note in self.notes.iter().take(4) {
                    ui.label(note);
                }
                if missing && ui.button("Download models").clicked() {
                    self.running = true;
                    self.handle.send(Cmd::FetchModels);
                }
            });
        ui.add_space(6.0);
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Folder");
            ui.add(egui::TextEdit::singleline(&mut self.folder).desired_width(360.0).hint_text("Photos, screenshots, scans"));
            if ui.button("Browse").clicked() {
                if let Some(path) = rfd::FileDialog::new().set_title("Index folder").pick_folder() {
                    self.folder = path.display().to_string();
                }
            }
            if ui.add_enabled(!self.folder.trim().is_empty() && !self.running, egui::Button::new("Index")).clicked() {
                self.running = true;
                self.progress = None;
                self.handle.send(Cmd::IndexPath(PathBuf::from(self.folder.trim())));
            }
            let pause_label = if self.handle.paused() { "Resume" } else { "Pause" };
            if ui.add_enabled(self.running, egui::Button::new(pause_label)).clicked() {
                self.handle.set_paused(!self.handle.paused());
            }
            if ui.add_enabled(self.running, egui::Button::new("Cancel")).clicked() {
                self.handle.cancel();
            }
            if ui.button("Retry errors").clicked() {
                self.running = true;
                self.handle.send(Cmd::RetryErrors);
            }
        });
        let progress = self.progress.as_ref().map(|progress| {
            format!(
                "  ·  {}/{} {}",
                progress.done,
                progress.total,
                Path::new(&progress.path)
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default()
            )
        });
        let photos = (self.stats.images - self.stats.frames).max(0);
        ui.label(format!(
            "{photos} photos · {} frames · {} with text · {} embedded{}",
            self.stats.frames,
            self.stats.with_text,
            self.stats.embedded,
            progress.unwrap_or_default()
        ));
    }

    fn browser(&mut self, ui: &mut egui::Ui) {
        let folder_note = self.folder_filter.and_then(|id| {
            if self.showing_hits || self.tab != Tab::Library {
                return None;
            }
            self.roots.iter().find(|(root, _)| *root == id).map(|(_, path)| format!("Folder · {path}"))
        });
        if let Some(note) = self.view_note.as_deref().or(folder_note.as_deref()) {
            ui.label(note);
        }
        if self.cards.is_empty() {
            let hidden_frames = self.gallery.iter().any(|item| {
                frame_rel(&item.rel) && self.folder_filter.is_none_or(|id| item.root_id == id)
            });
            let message = if self.tab == Tab::Duplicates {
                format!("No near-duplicates at {:.2} cosine.", self.dup_threshold)
            } else if self.open_collection.is_some() {
                "This collection is empty.".to_string()
            } else if self.tab == Tab::Library && !self.show_frames && hidden_frames {
                "No photos yet. Turn on Frames to see video samples.".to_string()
            } else if self.tab == Tab::Library && self.folder_filter.is_some() {
                "Nothing in this folder.".to_string()
            } else {
                "No images yet. Choose a folder and index it.".to_string()
            };
            ui.label(message);
            return;
        }
        let width = ui.available_width();
        let cols = if self.grid { (width / 168.0).floor().max(1.0) as usize } else { 1 };
        self.cols = cols;
        let rows = self.cards.len().div_ceil(cols);
        let annotated = self.cards.iter().any(|card| !card.snippet.is_empty());
        let row_h = if self.grid {
            if annotated { 208.0 } else { 188.0 }
        } else {
            28.0
        };
        egui::ScrollArea::vertical().auto_shrink([false, false]).show_rows(ui, row_h, rows, |ui, range| {
            for row in range {
                if self.grid {
                    ui.horizontal(|ui| {
                        for col in 0..cols {
                            let index = row * cols + col;
                            if index < self.cards.len() {
                                self.cell(ui, index, annotated);
                            }
                        }
                    });
                } else if row < self.cards.len() {
                    self.list_row(ui, row);
                }
            }
        });
    }

    fn cell(&mut self, ui: &mut egui::Ui, index: usize, annotated: bool) {
        let Some(card) = self.cards.get(index).cloned() else { return };
        let selected = self.selected == Some(card.id);
        let height = if annotated { 204.0 } else { 184.0 };
        ui.allocate_ui_with_layout(
            egui::vec2(164.0, height),
            egui::Layout::top_down(egui::Align::Center),
            |ui| {
                let (rect, response) = ui.allocate_exact_size(egui::vec2(148.0, 148.0), egui::Sense::click());
                if let Some(texture) = self.texture(ui.ctx(), card.id, &card.thumb) {
                    ui.painter().image(
                        texture,
                        rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                } else {
                    ui.painter().rect_filled(rect, egui::CornerRadius::same(4), egui::Color32::from_gray(32));
                }
                if let Some(group) = group_number(&card.snippet) {
                    let color = GROUP_COLORS[(group - 1) % GROUP_COLORS.len()];
                    let bar = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), 4.0));
                    ui.painter().rect_filled(bar, 0.0, color);
                }
                if selected {
                    ui.painter().rect_stroke(
                        rect,
                        egui::CornerRadius::same(4),
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 170, 255)),
                        egui::StrokeKind::Outside,
                    );
                }
                ui.add(egui::Label::new(card_name(&card.path, &card.rel)).truncate());
                if annotated {
                    ui.add(egui::Label::new(egui::RichText::new(card_note(&card)).small().weak()).truncate());
                }
                if response.double_clicked() {
                    self.load_detail(card.id);
                    self.viewer = true;
                    self.viewer_for = None;
                } else if response.clicked() {
                    self.load_detail(card.id);
                }
            },
        );
    }

    fn list_row(&mut self, ui: &mut egui::Ui, index: usize) {
        let Some(card) = self.cards.get(index).cloned() else { return };
        let selected = self.selected == Some(card.id);
        let title = if frame_rel(&card.rel) {
            card_name(&card.path, &card.rel)
        } else {
            card.path.clone()
        };
        let label = if card.snippet.is_empty() {
            title
        } else if let Some(score) = card.score {
            format!("{score:.3}  {title}  {}", card.snippet)
        } else {
            format!("{title}  {}", card.snippet)
        };
        let response = ui.selectable_label(selected, label);
        if response.double_clicked() {
            self.load_detail(card.id);
            self.viewer = true;
            self.viewer_for = None;
        } else if response.clicked() {
            self.load_detail(card.id);
        }
    }

    fn viewer_ui(&mut self, ui: &mut egui::Ui) {
        self.ensure_viewer(ui.ctx());
        let clip = self
            .detail
            .as_ref()
            .and_then(|detail| frame_source(&detail.exif_json))
            .is_some();
        ui.horizontal(|ui| {
            if ui.button("Back").clicked() {
                self.viewer = false;
            }
            ui.label(if clip {
                "Esc closes · left and right step through the clip"
            } else {
                "Esc closes · left and right move"
            });
        });
        let Some(texture) = self.viewer_tex.clone() else {
            ui.label("No preview");
            return;
        };
        let size = texture.size_vec2();
        if size.x < 1.0 || size.y < 1.0 {
            return;
        }
        let avail = ui.available_size();
        let scale = (avail.x / size.x).min(avail.y / size.y).clamp(0.01, 1.0);
        let draw = size * scale;
        let (rect, _) = ui.allocate_exact_size(draw, egui::Sense::hover());
        ui.painter().image(
            texture.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }

    fn side(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().id_salt("side").auto_shrink([false, false]).show(ui, |ui| {
            if let Some(detail) = self.detail.clone() {
                ui.strong(card_name(&detail.path, &detail.rel));
                ui.label(&detail.path);
                let source = frame_source(&detail.exif_json);
                let clip = source.as_ref().and_then(|source| {
                    self.db.as_ref().and_then(|db| db.clip_frames(source).ok())
                });
                if let Some(source) = &source {
                    ui.label(format!("Video: {source}"));
                    ui.horizontal(|ui| {
                        if ui.button("Show clip").clicked() {
                            self.show_clip(source);
                        }
                        if ui.button("Reveal video").clicked() {
                            config::reveal(Path::new(source));
                        }
                    });
                }
                if let Some(clip) = clip.as_ref().filter(|frames| frames.len() > 1) {
                    ui.horizontal_wrapped(|ui| {
                        for frame in clip {
                            let label = format!("{:.1}s", frame.seconds);
                            if ui.selectable_label(frame.id == detail.id, label).clicked() {
                                self.open_frame(frame.id);
                            }
                        }
                    });
                }
                if let (Some(w), Some(h)) = (detail.width, detail.height) {
                    ui.label(format!("{w}×{h} · OCR {} · embed {}", detail.ocr_status, detail.embed_status));
                }
                if let Some(error) = &detail.error {
                    ui.colored_label(egui::Color32::from_rgb(255, 160, 120), error);
                }
                ui.separator();
                ui.strong("Text");
                let mut ocr = detail.ocr.clone();
                ui.add(
                    egui::TextEdit::multiline(&mut ocr)
                        .interactive(false)
                        .desired_width(f32::INFINITY)
                        .desired_rows(8),
                );
                ui.strong("EXIF");
                for line in exif_lines(&detail.exif_json) {
                    ui.label(line);
                }
                ui.horizontal(|ui| {
                    if ui.add_enabled(self.clip_ready, egui::Button::new("Find similar")).clicked() {
                        self.gen += 1;
                        self.tab = Tab::Semantic;
                        self.showing_hits = true;
                        self.leave_special_view();
                        self.handle.send(Cmd::SimilarId { gen: self.gen, id: detail.id });
                    }
                    if ui.button("Reveal").clicked() {
                        config::reveal(Path::new(&detail.path));
                    }
                });
            } else {
                ui.label("Select an image.");
            }
            ui.separator();
            ui.label(format!("Models: {}", self.model_dir.display()));
            ui.label(format!("Data: {}", self.data_dir.display()));
            if !self.runtime.is_empty() {
                ui.label(format!("Runtime: {}", self.runtime));
            }
            let mut limit = self.det_limit as i32;
            if ui.add(egui::Slider::new(&mut limit, 640..=1600).text("det long side")).changed() {
                self.det_limit = limit as u32;
                self.handle.send(Cmd::SetDetLimit(self.det_limit));
            }
            ui.separator();
            ui.strong("Saved");
            ui.add(egui::TextEdit::singleline(&mut self.shelf_name).hint_text("Name").desired_width(f32::INFINITY));
            ui.horizontal(|ui| {
                let can_save = !self.shelf_name.trim().is_empty()
                    && !self.query.trim().is_empty()
                    && !matches!(self.tab, Tab::Duplicates);
                if ui.add_enabled(can_save, egui::Button::new("Save search")).clicked() {
                    let mode = match self.tab {
                        Tab::Semantic => Mode::Semantic,
                        Tab::Hybrid => Mode::Hybrid,
                        _ => Mode::Text,
                    };
                    self.handle.send(Cmd::SaveSearch {
                        name: self.shelf_name.trim().to_string(),
                        mode,
                        query: self.query.trim().to_string(),
                    });
                }
                let ids: Vec<i64> = if let Some(id) = self.selected {
                    vec![id]
                } else {
                    self.cards.iter().map(|card| card.id).collect()
                };
                let can_collect = !self.shelf_name.trim().is_empty() && !ids.is_empty();
                if ui.add_enabled(can_collect, egui::Button::new("Collect")).clicked() {
                    self.handle.send(Cmd::SaveCollection {
                        name: self.shelf_name.trim().to_string(),
                        ids,
                    });
                }
            });
            if let Some(collection) = self.open_collection {
                if ui
                    .add_enabled(self.selected.is_some(), egui::Button::new("Remove from collection"))
                    .clicked()
                {
                    if let Some(file) = self.selected {
                        self.handle.send(Cmd::RemoveFromCollection { collection, file });
                    }
                }
            }
            let searches = self.searches.clone();
            for search in searches {
                ui.horizontal(|ui| {
                    if ui.button(&search.name).clicked() {
                        self.run_saved(&search.mode, &search.query);
                    }
                    ui.weak(format!("{} · {}", search.mode, search.query));
                    if ui.small_button("×").clicked() {
                        self.handle.send(Cmd::DeleteSearch(search.id));
                    }
                });
            }
            ui.strong("Collections");
            let collections = self.collections.clone();
            for collection in collections {
                ui.horizontal(|ui| {
                    let label = format!("{} ({})", collection.name, collection.count);
                    if ui.button(label).clicked() {
                        self.show_collection(collection.id);
                    }
                    if ui.small_button("×").clicked() {
                        self.handle.send(Cmd::DeleteCollection(collection.id));
                    }
                });
            }
            ui.separator();
            ui.strong("Folders");
            let roots = self.roots.clone();
            for (id, path) in roots {
                ui.horizontal(|ui| {
                    if ui.selectable_label(self.folder_filter == Some(id), &path).clicked() {
                        self.select_folder(id);
                    }
                    if ui.small_button("Forget").clicked() {
                        if self.folder_filter == Some(id) {
                            self.folder_filter = None;
                        }
                        self.handle.send(Cmd::RemoveRoot(id));
                    }
                });
            }
        });
    }

    fn keys(&mut self, ui: &mut egui::Ui, search_focused: bool) {
        if search_focused {
            return;
        }
        if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.viewer = false;
        }
        if ui.input(|input| input.key_pressed(egui::Key::Enter)) && self.selected.is_some() && !self.viewer {
            self.viewer = true;
            self.viewer_for = None;
        }
        let cols = self.cols.max(1) as isize;
        if self.viewer {
            if ui.input(|input| input.key_pressed(egui::Key::ArrowLeft)) && !self.clip_step(-1) {
                self.move_selection(-1);
            }
            if ui.input(|input| input.key_pressed(egui::Key::ArrowRight)) && !self.clip_step(1) {
                self.move_selection(1);
            }
        } else {
            if ui.input(|input| input.key_pressed(egui::Key::ArrowLeft)) {
                self.move_selection(-1);
            }
            if ui.input(|input| input.key_pressed(egui::Key::ArrowRight)) {
                self.move_selection(1);
            }
            if ui.input(|input| input.key_pressed(egui::Key::ArrowUp)) {
                self.move_selection(-cols);
            }
            if ui.input(|input| input.key_pressed(egui::Key::ArrowDown)) {
                self.move_selection(cols);
            }
        }
    }
}

fn card_from_item(item: Item) -> Card {
    Card {
        id: item.id,
        path: item.path,
        rel: item.rel,
        thumb: item.thumb,
        snippet: String::new(),
        score: None,
    }
}

fn card_from_hit(hit: Hit) -> Card {
    Card {
        id: hit.id,
        path: hit.path,
        rel: hit.rel,
        thumb: hit.thumb,
        snippet: hit.snippet,
        score: Some(hit.score),
    }
}

fn card_from_frame(frame: db::ClipFrame) -> Card {
    Card {
        id: frame.id,
        path: frame.path,
        rel: frame.rel,
        thumb: frame.thumb,
        snippet: format!("{:.1}s", frame.seconds),
        score: None,
    }
}

fn card_from_text(hit: db::TextHit) -> Card {
    Card {
        id: hit.id,
        path: hit.path,
        rel: hit.rel,
        thumb: hit.thumb,
        snippet: hit.snippet,
        score: Some(hit.score),
    }
}

const GROUP_COLORS: [egui::Color32; 6] = [
    egui::Color32::from_rgb(90, 170, 255),
    egui::Color32::from_rgb(255, 170, 70),
    egui::Color32::from_rgb(120, 210, 140),
    egui::Color32::from_rgb(210, 130, 220),
    egui::Color32::from_rgb(255, 120, 140),
    egui::Color32::from_rgb(240, 220, 90),
];

fn card_name(path: &str, rel: &str) -> String {
    if frame_rel(rel) {
        return rel.to_string();
    }
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| path.to_string())
}

fn frame_rel(rel: &str) -> bool {
    let Some((name, time)) = rel.rsplit_once(" @ ") else {
        return false;
    };
    !name.is_empty()
        && time.ends_with('s')
        && time[..time.len() - 1].chars().all(|ch| ch.is_ascii_digit() || ch == '.')
        && time[..time.len() - 1].contains('.')
}

fn card_note(card: &Card) -> String {
    if card.snippet.is_empty() {
        return String::new();
    }
    if card.snippet.starts_with("group ") {
        return card.snippet.clone();
    }
    match card.score {
        Some(score) => format!("{score:.2}  {}", card.snippet),
        None => card.snippet.clone(),
    }
}

fn group_number(snippet: &str) -> Option<usize> {
    let rest = snippet.strip_prefix("group ")?;
    let number = rest.split_whitespace().next()?;
    let number: usize = number.parse().ok()?;
    (number > 0).then_some(number)
}

fn exif_lines(json: &str) -> Vec<String> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(json) else {
        return vec![json.to_string()];
    };
    let Some(object) = value.as_object() else {
        return vec![json.to_string()];
    };
    object.iter().map(|(key, value)| format!("{key}: {value}")).collect()
}

fn frame_source(exif_json: &str) -> Option<String> {
    let value = serde_json::from_str::<serde_json::Value>(exif_json).ok()?;
    let source = value.get("source")?.as_str()?.trim();
    if source.is_empty() {
        None
    } else {
        Some(source.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_rel_is_the_card_name() {
        assert_eq!(
            card_name("/data/frames/ab/clip-00-2.jpg", "clip.mp4 @ 0.2s"),
            "clip.mp4 @ 0.2s"
        );
        assert_eq!(card_name("/photos/vacation/dog.jpg", "vacation/dog.jpg"), "dog.jpg");
        assert!(!frame_rel("photo @ 2s.jpg"));
        assert_eq!(group_number("group 2 · 0.991"), Some(2));
        assert_eq!(
            frame_source(r#"{"source":"/videos/clip.mp4","time":"0.2s"}"#).as_deref(),
            Some("/videos/clip.mp4")
        );
        assert!(frame_source("{}").is_none());
    }
}
