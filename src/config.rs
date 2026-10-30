//! Where the library database, thumbnails, embeddings, and models live.
//!
//! A folder with a `portable` marker beside the executable keeps its data
//! next to the binary. Otherwise the data goes under the per-user data
//! directory. `SEMANTIX_DATA` and `SEMANTIX_MODELS` override both.

use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Config {
    pub data_dir: PathBuf,
    pub model_dir: PathBuf,
    pub cuda: bool,
    pub det_limit: u32,
}

impl Config {
    pub fn resolve(data: Option<PathBuf>, models: Option<PathBuf>, cuda: bool) -> Self {
        let data_dir = data.unwrap_or_else(default_data_dir);
        let model_dir = models.unwrap_or_else(|| default_model_dir(&data_dir));
        Self {
            data_dir,
            model_dir,
            cuda,
            det_limit: 960,
        }
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("library.db")
    }

    pub fn thumb_dir(&self) -> PathBuf {
        self.data_dir.join("thumbs")
    }

    pub fn embed_path(&self) -> PathBuf {
        self.data_dir.join("embeddings.bin")
    }

    pub fn frames_dir(&self) -> PathBuf {
        self.data_dir.join("frames")
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(self.thumb_dir())?;
        std::fs::create_dir_all(self.frames_dir())?;
        std::fs::create_dir_all(&self.model_dir)?;
        Ok(())
    }
}

pub fn default_data_dir() -> PathBuf {
    if let Some(dir) = env::var_os("SEMANTIX_DATA") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    if let Some(dir) = exe_dir() {
        if dir.join("portable").is_file() {
            return dir.join("data");
        }
    }
    user_data_dir()
}

pub fn default_model_dir(data_dir: &Path) -> PathBuf {
    if let Some(dir) = env::var_os("SEMANTIX_MODELS") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    if let Some(dir) = exe_dir() {
        let bundled = dir.join("models");
        if bundled.join("det.onnx").is_file() || bundled.join("vision.onnx").is_file() {
            return bundled;
        }
    }
    data_dir.join("models")
}

fn exe_dir() -> Option<PathBuf> {
    env::current_exe().ok().and_then(|path| path.parent().map(Path::to_path_buf))
}

fn user_data_dir() -> PathBuf {
    #[cfg(windows)]
    {
        env::var_os("LOCALAPPDATA")
            .or_else(|| env::var_os("APPDATA"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join("semantix")
    }
    #[cfg(target_os = "macos")]
    {
        home().join("Library/Application Support/semantix")
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(dir) = env::var_os("XDG_DATA_HOME") {
            if !dir.is_empty() {
                return PathBuf::from(dir).join("semantix");
            }
        }
        home().join(".local/share/semantix")
    }
}

fn home() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn reveal(path: &Path) {
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg("-R").arg(path).spawn();
    #[cfg(windows)]
    let _ = std::process::Command::new("explorer")
        .arg(format!("/select,{}", path.display()))
        .spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let dir = path.parent().unwrap_or(path);
        let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
    }
}

#[cfg(unix)]
pub fn lower_priority() {
    unsafe {
        // PRIO_PROCESS = 0, who = 0 means this process. 10 is a mild background niceness.
        setpriority(0, 0, 10);
    }
}

#[cfg(unix)]
unsafe extern "C" {
    fn setpriority(which: i32, who: u32, prio: i32) -> i32;
}

#[cfg(not(unix))]
pub fn lower_priority() {}
