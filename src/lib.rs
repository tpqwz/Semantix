//! Local media library. Indexing, OCR, CLIP embeddings, and search run in this
//! crate. The window is a shell over the same engine.

pub mod clip;
mod hot;
pub mod config;
pub mod db;
pub mod embed;
pub mod engine;
pub mod exif;
pub mod geom;
pub mod gui;
pub mod imageutil;
pub mod models;
pub mod ocr;
mod pcloud;
pub mod runtime;
pub mod search;
pub mod video;

pub use config::Config;
pub use hot::host as hot_host;
pub use engine::{Engine, Progress};
pub use search::{Hit, Mode};
