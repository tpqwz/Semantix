//! Dev-time reload, in the style of CToy.
//!
//! The process stays up. Saving a source file rebuilds the library, the
//! window closes and opens again in that same process, and the database on
//! disk is left as it was. A failed build keeps the current window and shows
//! the compiler output. The published executable does not do this.
//!
//! ONNX Runtime allows one environment per process, so the first load leaks
//! its environment and later loads clone that handle. The standard library
//! has to be the dynamic one, or those clones would use two allocators.

use anyhow::{anyhow, Result};
use libloading::Library;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[repr(C)]
pub struct HotControl {
    pub env: *mut usize,
    pub x: *mut i32,
    pub y: *mut i32,
    pub w: *mut i32,
    pub h: *mut i32,
    pub generation: i32,
    pub cuda: i32,
    pub data: *const u8,
    pub models: *const u8,
}

struct Live {
    reload: Arc<AtomicBool>,
    compiling: Arc<AtomicBool>,
    error: Arc<Mutex<String>>,
    env: *mut usize,
    x: *mut i32,
    y: *mut i32,
    w: *mut i32,
    h: *mut i32,
    generation: i32,
    saved: [i32; 4],
}

// The pointers stay on the host stack for the whole load. The watcher only
// touches the atomics and the error string, which are behind Arc.
unsafe impl Send for Live {}

static LIVE: Mutex<Option<Live>> = Mutex::new(None);

pub fn host(cfg: crate::Config) -> Result<()> {
    if std::env::var_os("SEMANTIX_HOT_CHILD").is_none() {
        relaunch()?;
    }
    if cfg!(target_os = "linux") && !std_is_dynamic() {
        return Err(anyhow!(
            "--hot needs a dynamically linked standard library, and this build does not have one"
        ));
    }
    let manifest = manifest_dir();
    if !manifest.join("Cargo.toml").is_file() {
        return Err(anyhow!("--hot runs from a Semantix source checkout"));
    }
    let mut env_slot = 0usize;
    let mut x = 0i32;
    let mut y = 0i32;
    let mut w = 0i32;
    let mut h = 0i32;
    let mut generation = 0i32;
    let data = CString::new(cfg.data_dir.to_string_lossy().as_bytes()).map_err(|_| anyhow!("data path"))?;
    let models = CString::new(cfg.model_dir.to_string_lossy().as_bytes()).map_err(|_| anyhow!("model path"))?;
    loop {
        if generation == 0 {
            let built = cargo_build(true)?;
            if !built.status.success() {
                let err = String::from_utf8_lossy(&built.stderr);
                return Err(anyhow!("{}", tail(&err)));
            }
        }
        let copy = stage_library(&manifest, generation)?;
        let lib = unsafe { Library::new(&copy) }.map_err(|err| anyhow!("loading {copy:?}: {err}"))?;
        let run: libloading::Symbol<unsafe extern "C" fn(*mut HotControl) -> i32> =
            unsafe { lib.get(b"semantix_hot_run\0") }.map_err(|err| anyhow!("semantix_hot_run: {err}"))?;
        let mut control = HotControl {
            env: &mut env_slot,
            x: &mut x,
            y: &mut y,
            w: &mut w,
            h: &mut h,
            generation,
            cuda: i32::from(cfg.cuda),
            data: data.as_ptr().cast(),
            models: models.as_ptr().cast(),
        };
        let code = unsafe { run(&mut control) };
        // The old code stays mapped. Dropping it would unmap the leaked
        // ONNX environment and any thread still leaving run().
        std::mem::forget(lib);
        if code == 0 {
            return Ok(());
        }
        generation += 1;
        eprintln!("semantix: reloaded");
    }
}

fn relaunch() -> Result<()> {
    let manifest = manifest_dir();
    if !manifest.join("src").is_dir() {
        return Err(anyhow!("--hot runs from a Semantix source checkout"));
    }
    let flags = with_prefer_dynamic(std::env::var("RUSTFLAGS").ok());
    let libs = prepend_rustlib(std::env::var("LD_LIBRARY_PATH").ok());
    let exe = manifest.join("target/debug").join(if cfg!(windows) { "semantix.exe" } else { "semantix" });
    if cargo_is_parent() {
        // cargo run holds the build lock until we exit. The helper waits,
        // then compiles and takes our place.
        let pid = std::process::id();
        let mut args = String::new();
        for arg in std::env::args().skip(1) {
            args.push(' ');
            args.push_str(&shell_quote(&arg));
        }
        let script = format!(
            "while kill -0 {pid} 2>/dev/null; do sleep 0.05; done\n\
             cd {}\n\
             export RUSTFLAGS={}\n\
             export LD_LIBRARY_PATH={}\n\
             export SEMANTIX_HOT_CHILD=1\n\
             cargo build || exit $?\n\
             exec {}{}\n",
            shell_quote(&manifest.to_string_lossy()),
            shell_quote(&flags),
            shell_quote(&libs),
            shell_quote(&exe.to_string_lossy()),
            args,
        );
        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg(script);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        cmd.spawn().map_err(|err| anyhow!("starting the hot host: {err}"))?;
        std::process::exit(0);
    }
    let built = Command::new("cargo")
        .current_dir(&manifest)
        .arg("build")
        .env("RUSTFLAGS", &flags)
        .status()
        .map_err(|err| anyhow!("cargo: {err}"))?;
    if !built.success() {
        std::process::exit(built.code().unwrap_or(1));
    }
    let mut cmd = Command::new(&exe);
    cmd.args(std::env::args().skip(1));
    cmd.env("SEMANTIX_HOT_CHILD", "1");
    cmd.env("RUSTFLAGS", &flags);
    cmd.env("LD_LIBRARY_PATH", &libs);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        return Err(anyhow!("exec {}: {err}", exe.display()));
    }
    #[cfg(not(unix))]
    {
        let status = cmd.status()?;
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn cargo_is_parent() -> bool {
    #[cfg(target_os = "linux")]
    {
        let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
            return false;
        };
        let Some(ppid) = status.lines().find_map(|line| line.strip_prefix("PPid:")).map(str::trim) else {
            return false;
        };
        let Ok(cmd) = std::fs::read(format!("/proc/{ppid}/cmdline")) else {
            return false;
        };
        return String::from_utf8_lossy(&cmd).contains("cargo");
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

fn rustlib_dir() -> String {
    Command::new("rustc")
        .args(["--print", "target-libdir"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|dir| dir.trim().to_string())
        .filter(|dir| !dir.is_empty())
        .unwrap_or_default()
}

fn prepend_rustlib(existing: Option<String>) -> String {
    let dir = rustlib_dir();
    match existing {
        Some(old) if !old.is_empty() && !old.split(':').any(|part| part == dir) => format!("{dir}:{old}"),
        Some(old) if !old.is_empty() => old,
        _ => dir,
    }
}

fn shell_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', "'\\''"))
}

#[no_mangle]
pub unsafe extern "C" fn semantix_hot_run(control: *mut HotControl) -> i32 {
    if control.is_null() {
        return 0;
    }
    let control = unsafe { &*control };
    let reload = Arc::new(AtomicBool::new(false));
    let compiling = Arc::new(AtomicBool::new(false));
    let error = Arc::new(Mutex::new(String::new()));
    {
        let mut live = LIVE.lock().unwrap_or_else(|err| err.into_inner());
        *live = Some(Live {
            reload: reload.clone(),
            compiling: compiling.clone(),
            error: error.clone(),
            env: control.env,
            x: control.x,
            y: control.y,
            w: control.w,
            h: control.h,
            generation: control.generation,
            saved: [
                unsafe { control.x.read() },
                unsafe { control.y.read() },
                unsafe { control.w.read() },
                unsafe { control.h.read() },
            ],
        });
    }
    let stop = Arc::new(AtomicBool::new(false));
    let watch = spawn_watcher(stop.clone(), reload.clone(), compiling, error);
    let data = cstr_path(control.data);
    let models = cstr_path(control.models);
    let cfg = crate::Config::resolve(data, models, control.cuda != 0);
    let run = crate::gui::launch(cfg);
    stop.store(true, Ordering::Relaxed);
    if let Some(watch) = watch {
        let _ = watch.join();
    }
    let reload = {
        let live = LIVE.lock().unwrap_or_else(|err| err.into_inner());
        live.as_ref().is_some_and(|live| live.reload.load(Ordering::Relaxed))
    };
    let mut slot = LIVE.lock().unwrap_or_else(|err| err.into_inner());
    *slot = None;
    drop(slot);
    match run {
        Ok(()) if reload => 1,
        Ok(()) => 0,
        Err(err) => {
            eprintln!("semantix: {err:#}");
            0
        }
    }
}

fn spawn_watcher(
    stop: Arc<AtomicBool>,
    reload: Arc<AtomicBool>,
    compiling: Arc<AtomicBool>,
    error: Arc<Mutex<String>>,
) -> Option<std::thread::JoinHandle<()>> {
    let manifest = manifest_dir();
    let mut seen = source_stamp(&manifest);
    Some(std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(200));
            let now = source_stamp(&manifest);
            if !stamp_changed(&seen, &now) {
                continue;
            }
            std::thread::sleep(Duration::from_millis(150));
            if stop.load(Ordering::Relaxed) {
                break;
            }
            seen = source_stamp(&manifest);
            compiling.store(true, Ordering::Relaxed);
            match cargo_build(true) {
                Ok(output) if output.status.success() => {
                    if let Ok(mut slot) = error.lock() {
                        slot.clear();
                    }
                    compiling.store(false, Ordering::Relaxed);
                    reload.store(true, Ordering::Relaxed);
                    break;
                }
                Ok(output) => {
                    let text = format!(
                        "{}\n{}",
                        String::from_utf8_lossy(&output.stderr),
                        String::from_utf8_lossy(&output.stdout)
                    );
                    if let Ok(mut slot) = error.lock() {
                        *slot = tail(&text);
                    }
                    compiling.store(false, Ordering::Relaxed);
                    seen = source_stamp(&manifest);
                }
                Err(err) => {
                    if let Ok(mut slot) = error.lock() {
                        *slot = err.to_string();
                    }
                    compiling.store(false, Ordering::Relaxed);
                }
            }
        }
    }))
}

pub fn active() -> bool {
    LIVE.lock().ok().is_some_and(|live| live.is_some())
}

pub fn reload_requested() -> bool {
    LIVE.lock().ok().is_some_and(|live| live.as_ref().is_some_and(|live| live.reload.load(Ordering::Relaxed)))
}

pub fn compiling() -> bool {
    LIVE.lock().ok().is_some_and(|live| live.as_ref().is_some_and(|live| live.compiling.load(Ordering::Relaxed)))
}

pub fn generation() -> i32 {
    LIVE.lock().ok().and_then(|live| live.as_ref().map(|live| live.generation)).unwrap_or(0)
}

pub fn build_error() -> Option<String> {
    let live = LIVE.lock().ok()?;
    let live = live.as_ref()?;
    let message = live.error.lock().ok()?;
    if message.is_empty() {
        None
    } else {
        Some(message.clone())
    }
}

pub fn saved_bounds() -> Option<[i32; 4]> {
    let live = LIVE.lock().ok()?;
    let saved = live.as_ref()?.saved;
    (saved[2] > 200 && saved[3] > 200).then_some(saved)
}

pub fn remember_bounds(x: i32, y: i32, w: i32, h: i32) {
    let Ok(live) = LIVE.lock() else { return };
    let Some(live) = live.as_ref() else { return };
    unsafe {
        if !live.x.is_null() {
            *live.x = x;
        }
        if !live.y.is_null() {
            *live.y = y;
        }
        if !live.w.is_null() {
            *live.w = w;
        }
        if !live.h.is_null() {
            *live.h = h;
        }
    }
}

pub fn adopted_environment() -> Option<ort::environment::Environment> {
    let live = LIVE.lock().ok()?;
    let slot = live.as_ref()?.env;
    if slot.is_null() {
        return None;
    }
    let addr = unsafe { *slot };
    if addr == 0 {
        return None;
    }
    // The first load leaked this environment. Same rustc, same ort, dynamic libstd.
    let env = unsafe { &*(addr as *const ort::environment::Environment) };
    Some(env.clone())
}

pub fn publish_environment(env: &ort::environment::Environment) {
    let Ok(live) = LIVE.lock() else { return };
    let Some(live) = live.as_ref() else { return };
    if live.env.is_null() || unsafe { *live.env } != 0 {
        return;
    }
    let leaked = Box::leak(Box::new(env.clone()));
    unsafe {
        *live.env = leaked as *mut ort::environment::Environment as usize;
    }
}

fn cstr_path(ptr: *const u8) -> Option<PathBuf> {
    if ptr.is_null() {
        return None;
    }
    let mut len = 0usize;
    while len < 4096 && unsafe { *ptr.add(len) } != 0 {
        len += 1;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    let text = String::from_utf8_lossy(bytes);
    if text.is_empty() {
        None
    } else {
        Some(PathBuf::from(text.as_ref()))
    }
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn cargo_build(lib_only: bool) -> Result<std::process::Output> {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(manifest_dir()).arg("build");
    if lib_only {
        cmd.arg("--lib");
    }
    cmd.env("RUSTFLAGS", with_prefer_dynamic(std::env::var("RUSTFLAGS").ok()));
    cmd.output().map_err(|err| anyhow!("cargo: {err}"))
}

fn stage_library(manifest: &Path, generation: i32) -> Result<PathBuf> {
    let name = if cfg!(windows) {
        "semantix.dll"
    } else if cfg!(target_os = "macos") {
        "libsemantix.dylib"
    } else {
        "libsemantix.so"
    };
    let source = manifest.join("target/debug").join(name);
    if !source.is_file() {
        return Err(anyhow!("missing {}", source.display()));
    }
    let dir = manifest.join("target/hot");
    std::fs::create_dir_all(&dir)?;
    let dest = dir.join(format!("semantix-hot-{generation}-{name}"));
    std::fs::copy(&source, &dest)?;
    Ok(dest)
}

fn source_stamp(manifest: &Path) -> Vec<(PathBuf, u128)> {
    let mut out = Vec::new();
    for rel in ["Cargo.toml", "Cargo.lock"] {
        stamp_file(&mut out, &manifest.join(rel));
    }
    stamp_tree(&mut out, &manifest.join("src"));
    stamp_tree(&mut out, &manifest.join("assets"));
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn stamp_tree(out: &mut Vec<(PathBuf, u128)>, dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            stamp_tree(out, &path);
        } else {
            stamp_file(out, &path);
        }
    }
}

fn stamp_file(out: &mut Vec<(PathBuf, u128)>, path: &Path) {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else { return };
    if name.ends_with('~') || name.starts_with(".#") || name.ends_with(".swp") {
        return;
    }
    let Ok(meta) = std::fs::metadata(path) else { return };
    let modified = meta.modified().ok().and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok());
    let nanos = modified.map(|time| time.as_nanos()).unwrap_or(0);
    out.push((path.to_path_buf(), nanos));
}

fn stamp_changed(old: &[(PathBuf, u128)], new: &[(PathBuf, u128)]) -> bool {
    old != new
}

fn with_prefer_dynamic(flags: Option<String>) -> String {
    let flags = flags.unwrap_or_default();
    if flags.contains("prefer-dynamic") {
        flags
    } else if flags.trim().is_empty() {
        "-C prefer-dynamic".to_string()
    } else {
        format!("{flags} -C prefer-dynamic")
    }
}

fn tail(text: &str) -> String {
    let text = text.trim();
    if text.len() <= 1800 {
        return text.to_string();
    }
    text[text.len() - 1800..].to_string()
}

#[cfg(target_os = "linux")]
fn std_is_dynamic() -> bool {
    std::fs::read_to_string("/proc/self/maps").map(|maps| maps.contains("libstd-")).unwrap_or(false)
}

#[cfg(not(target_os = "linux"))]
fn std_is_dynamic() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_notices_a_new_mtime_and_a_new_file() {
        let src = PathBuf::from("src/gui.rs");
        let old = vec![(src.clone(), 1u128)];
        assert!(!stamp_changed(&old, &old));
        assert!(stamp_changed(&old, &[(src.clone(), 2)]));
        assert!(stamp_changed(&old, &[
            (src, 1),
            (PathBuf::from("src/hot.rs"), 1),
        ]));
    }

    #[test]
    fn shell_quote_wraps_spaces_and_quotes() {
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
    }

    #[test]
    fn prefer_dynamic_is_added_once() {
        assert_eq!(with_prefer_dynamic(None), "-C prefer-dynamic");
        assert_eq!(with_prefer_dynamic(Some("-C prefer-dynamic".into())), "-C prefer-dynamic");
        assert_eq!(
            with_prefer_dynamic(Some("-C debuginfo=0".into())),
            "-C debuginfo=0 -C prefer-dynamic"
        );
    }
}
