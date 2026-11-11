//! Locate ONNX Runtime and re-exec so the dynamic linker can see it.
//!
//! The search path is read at process start, so it has to be set before the
//! library is opened. CPU inference uses `libonnxruntime` only. When no copy
//! is already on disk, the app downloads the official CPU archive and checks
//! its SHA-256. CUDA provider libraries are loaded later, and only when that
//! device is requested. They are not downloaded.

use anyhow::{anyhow, Result};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::TensorElementType;
use ort::AsPointer;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

const READY: &str = "SEMANTIX_LIBS_READY";

pub fn prepare() {
    if env::var_os(READY).is_some() {
        return;
    }
    let Some(dir) = runtime_lib_dir() else {
        return;
    };
    let ort = dir.join(ort_library_file());
    if !ort.is_file() {
        return;
    }
    let exe = env::current_exe().unwrap_or_else(|_| PathBuf::from("semantix"));
    let joined = prepend_search_path(&dir, env::var(library_search_var()).ok().as_deref());
    let status = Command::new(exe)
        .args(env::args().skip(1))
        .env(READY, "1")
        .env(library_search_var(), &joined)
        .env("ORT_DYLIB_PATH", &ort)
        .status();
    match status {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(err) => eprintln!("semantix: failed to re-exec with ONNX Runtime: {err}"),
    }
}

pub fn environment(cuda: bool) -> Result<Environment> {
    // ONNX Runtime allows one environment per process. Sessions share that clone.
    static SLOT: OnceLock<Mutex<Option<Environment>>> = OnceLock::new();
    let slot = SLOT.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock().unwrap_or_else(|err| err.into_inner());
    if let Some(env) = guard.as_ref() {
        return Ok(env.clone());
    }
    if let Some(env) = crate::hot::adopted_environment() {
        *guard = Some(env.clone());
        return Ok(env);
    }
    let dir = match runtime_lib_dir() {
        Some(dir) => dir,
        None => download_runtime()?,
    };
    if cuda {
        preload_cuda();
    }
    let builder = ort::init_from(dir.join(ort_library_file()))
        .map_err(|err| anyhow!("loading ONNX Runtime: {err}"))?;
    let env = builder
        .with_name("semantix")
        .with_telemetry(false)
        .build()
        .map_err(|err| anyhow!("onnx runtime: {err}"))?;
    crate::hot::publish_environment(&env);
    *guard = Some(env.clone());
    Ok(env)
}

pub fn preload_cuda() {
    let Some(dir) = runtime_lib_dir() else {
        return;
    };
    if dir.join(cuda_provider_file()).is_file() {
        let _ = ort::ep::cuda::preload_dylibs(None, Some(dir.as_path()));
        let _ = ort::util::preload_dylib(dir.join(shared_provider_file()));
        let _ = ort::util::preload_dylib(dir.join(cuda_provider_file()));
    }
}

pub fn open_session(env: &Environment, path: &Path, cuda: bool) -> Result<Session> {
    let mut builder = Session::builder(env).map_err(|err| anyhow!("session: {err}"))?;
    if cuda {
        builder = builder
            .with_execution_providers([ort::ep::CUDA::default().with_device_id(0).build()])
            .map_err(|err| anyhow!("cuda provider: {err}"))?;
    } else {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(1, 8);
        builder = builder
            .with_intra_threads(threads)
            .map_err(|err| anyhow!("cpu threads: {err}"))?;
        // ort 2.0.0-rc.15 turns on MaxEfficiency device selection for every
        // session. ONNX Runtime 1.22 discovers no hardware devices on Linux,
        // and that policy calls front() on the empty list and aborts.
        // Registering the classic CPU provider skips automatic selection.
        append_cpu_provider(&mut builder)?;
    }
    builder
        .commit_from_file(path)
        .map_err(|err| anyhow!("loading {}: {err}", path.display()))
}

fn append_cpu_provider(builder: &mut ort::session::builder::SessionBuilder) -> Result<()> {
    let lib_path = runtime_lib_dir()
        .map(|dir| dir.join(ort_library_file()))
        .ok_or_else(|| anyhow!("ONNX Runtime library was not found"))?;
    let library = unsafe { libloading::Library::new(&lib_path) }
        .map_err(|err| anyhow!("opening ONNX Runtime to register CPU: {err}"))?;
    let register: libloading::Symbol<
        unsafe extern "C" fn(*mut ort::sys::OrtSessionOptions, i32) -> *mut ort::sys::OrtStatus,
    > = unsafe { library.get(b"OrtSessionOptionsAppendExecutionProvider_CPU\0") }
        .map_err(|err| anyhow!("this ONNX Runtime has no CPU execution provider: {err}"))?;
    let status = unsafe { register(builder.ptr_mut(), 1) };
    if status.is_null() {
        return Ok(());
    }
    let api = ort::api();
    let message = unsafe { std::ffi::CStr::from_ptr((api.GetErrorMessage)(status)) }
        .to_string_lossy()
        .into_owned();
    unsafe { (api.ReleaseStatus)(status) };
    Err(anyhow!("CPU execution provider: {message}"))
}

pub fn tensor_shape(dtype: &ort::value::ValueType) -> Vec<i64> {
    match dtype {
        ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
        _ => Vec::new(),
    }
}

pub fn positive_dim(shape: &[i64], index: usize) -> Option<usize> {
    shape.get(index).copied().filter(|dim| *dim > 0).map(|dim| dim as usize)
}

pub fn extract_f32(value: &ort::value::DynValue) -> Result<(Vec<usize>, Vec<f32>)> {
    match value.try_extract_tensor::<f32>() {
        Ok((shape, data)) => {
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            Ok((shape, data.to_vec()))
        }
        Err(_) => {
            let (shape, data) = value
                .try_extract_tensor::<half::f16>()
                .map_err(|err| anyhow!("output is neither f32 nor f16: {err}"))?;
            let shape = shape.iter().map(|dim| *dim as usize).collect();
            let data = data.iter().copied().map(half::f16::to_f32).collect();
            Ok((shape, data))
        }
    }
}

pub fn is_int64(kind: Option<TensorElementType>) -> bool {
    !matches!(kind, Some(TensorElementType::Int32))
}

pub fn runtime_label() -> String {
    match runtime_lib_dir() {
        Some(dir) => dir.display().to_string(),
        None => "system ONNX Runtime".to_string(),
    }
}

pub fn ort_library_file() -> String {
    dylib("onnxruntime")
}

pub fn cuda_provider_file() -> String {
    dylib("onnxruntime_providers_cuda")
}

pub fn shared_provider_file() -> String {
    dylib("onnxruntime_providers_shared")
}

pub fn library_search_var() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

pub fn prepend_search_path(dir: &Path, old: Option<&str>) -> String {
    let dir = dir.display().to_string();
    match old {
        Some(old) if !old.is_empty() => format!("{dir}{}{old}", search_path_sep()),
        _ => dir,
    }
}

fn search_path_sep() -> char {
    if cfg!(windows) { ';' } else { ':' }
}

fn dylib(name: &str) -> String {
    format!("{}{name}{}", env::consts::DLL_PREFIX, env::consts::DLL_SUFFIX)
}

pub fn runtime_lib_dir() -> Option<PathBuf> {
    let library = ort_library_file();
    let mut candidates = Vec::new();
    if let Some(dir) = env::var_os("SEMANTIX_RUNTIME") {
        candidates.push(PathBuf::from(dir));
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("lib"));
            candidates.push(parent.join("runtime").join("lib"));
            candidates.push(parent.join("..").join("..").join("runtime").join("lib"));
        }
    }
    candidates.push(crate::config::default_data_dir().join("lib"));
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime/lib"));
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../st/starlign/runtime/lib"));
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../starlign/runtime/lib"));
    candidates.into_iter().find(|dir| dir.join(&library).is_file())
}

const ORT_VERSION: &str = "1.22.0";

struct OrtAsset {
    os: &'static str,
    arch: &'static str,
    archive: &'static str,
    sha256: &'static str,
}

// Official CPU archives. The hash is the archive, so a replaced upload is rejected.
const ORT_ASSETS: &[OrtAsset] = &[
    OrtAsset {
        os: "linux",
        arch: "x86_64",
        archive: "onnxruntime-linux-x64-1.22.0.tgz",
        sha256: "8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3",
    },
    OrtAsset {
        os: "linux",
        arch: "aarch64",
        archive: "onnxruntime-linux-aarch64-1.22.0.tgz",
        sha256: "bb76395092d150b52c7092dc6b8f2fe4d80f0f3bf0416d2f269193e347e24702",
    },
    OrtAsset {
        os: "windows",
        arch: "x86_64",
        archive: "onnxruntime-win-x64-1.22.0.zip",
        sha256: "174c616efc0271194488642a72f1a514e01487da4dfe84c49296d66e40ebe0da",
    },
    OrtAsset {
        os: "macos",
        arch: "aarch64",
        archive: "onnxruntime-osx-arm64-1.22.0.tgz",
        sha256: "cab6dcbd77e7ec775390e7b73a8939d45fec3379b017c7cb74f5b204c1a1cc07",
    },
    OrtAsset {
        os: "macos",
        arch: "x86_64",
        archive: "onnxruntime-osx-x86_64-1.22.0.tgz",
        sha256: "e4ec94a7696de74fb1b12846569aa94e499958af6ffa186022cfde16c9d617f0",
    },
];

fn current_asset() -> Result<&'static OrtAsset> {
    ORT_ASSETS
        .iter()
        .find(|asset| asset.os == env::consts::OS && asset.arch == env::consts::ARCH)
        .ok_or_else(|| anyhow!("no CPU ONNX Runtime build is published for {}-{}", env::consts::OS, env::consts::ARCH))
}

fn download_runtime() -> Result<PathBuf> {
    let exe_dir = env::current_exe().ok().and_then(|path| path.parent().map(Path::to_path_buf));
    let data_dir = crate::config::default_data_dir();
    let dir = install_dir(exe_dir.as_deref(), &data_dir);
    download_runtime_to(&dir)?;
    let _ = ACQUIRED.set(dir.clone());
    Ok(dir)
}

static ACQUIRED: OnceLock<PathBuf> = OnceLock::new();

pub fn acquired_note() -> Option<String> {
    ACQUIRED.get().map(|dir| format!("Downloaded ONNX Runtime into {}.", dir.display()))
}

fn install_dir(exe_dir: Option<&Path>, data_dir: &Path) -> PathBuf {
    if let Some(dir) = exe_dir {
        let lib = dir.join("lib");
        if can_write(&lib) {
            return lib;
        }
    }
    data_dir.join("lib")
}

fn can_write(dir: &Path) -> bool {
    if std::fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(".semantix-write-test");
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

fn download_runtime_to(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let asset = current_asset()?;
    let scratch = std::env::temp_dir().join(format!("semantix-ort-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch)?;
    let archive = scratch.join(asset.archive);
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{}",
        asset.archive
    );
    eprintln!("downloading {url}");
    let saved = download_verified(&url, &archive, asset.sha256);
    if let Err(err) = saved {
        let _ = std::fs::remove_dir_all(&scratch);
        return Err(err);
    }
    let extract = scratch.join("extract");
    std::fs::create_dir_all(&extract)?;
    let unpacked = Command::new("tar")
        .arg("-xf")
        .arg(&archive)
        .arg("-C")
        .arg(&extract)
        .status()
        .map_err(|err| anyhow!("tar is required to unpack ONNX Runtime: {err}"));
    match unpacked {
        Ok(status) if status.success() => {}
        Ok(status) => {
            let _ = std::fs::remove_dir_all(&scratch);
            return Err(anyhow!("tar exited {status} while unpacking ONNX Runtime"));
        }
        Err(err) => {
            let _ = std::fs::remove_dir_all(&scratch);
            return Err(err);
        }
    }
    let source = match find_runtime_blob(&extract) {
        Ok(path) => path,
        Err(err) => {
            let _ = std::fs::remove_dir_all(&scratch);
            return Err(err);
        }
    };
    let dest = dir.join(ort_library_file());
    let partial = dir.join(format!("{}.partial", ort_library_file()));
    std::fs::copy(&source, &partial)?;
    std::fs::rename(&partial, &dest)?;
    let _ = std::fs::remove_dir_all(&scratch);
    eprintln!("ONNX Runtime is in {}", dest.display());
    Ok(())
}

fn download_verified(url: &str, dest: &Path, sha256: &str) -> Result<()> {
    let partial = dest.with_extension("partial");
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(120))
        .build();
    let response = agent.get(url).call().map_err(|err| anyhow!("GET {url}: {err}"))?;
    let mut reader = response.into_reader();
    let mut out = std::fs::File::create(&partial)?;
    std::io::copy(&mut reader, &mut out)?;
    out.sync_all()?;
    drop(out);
    let got = crate::models::hash_of(&partial)?;
    if got != sha256 {
        let _ = std::fs::remove_file(&partial);
        return Err(anyhow!("ONNX Runtime hash mismatch: got {got}, expected {sha256}"));
    }
    if dest.exists() {
        let _ = std::fs::remove_file(dest);
    }
    std::fs::rename(&partial, dest)?;
    Ok(())
}

fn find_runtime_blob(root: &Path) -> Result<PathBuf> {
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in walkdir::WalkDir::new(root).follow_links(false) {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_file() {
            continue;
        }
        if !is_runtime_blob(entry.path()) {
            continue;
        }
        let len = entry.metadata().map(|meta| meta.len()).unwrap_or(0);
        if best.as_ref().is_none_or(|found| len > found.0) {
            best = Some((len, entry.path().to_path_buf()));
        }
    }
    best.map(|(_, path)| path).ok_or_else(|| anyhow!("ONNX Runtime archive has no CPU library"))
}

fn is_runtime_blob(path: &Path) -> bool {
    if path.parent().and_then(|parent| parent.file_name()).and_then(|name| name.to_str()) != Some("lib") {
        return false;
    }
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if name.contains("provider") {
        return false;
    }
    if name.eq_ignore_ascii_case("onnxruntime.dll") {
        return true;
    }
    name.starts_with("libonnxruntime.so") || (name.starts_with("libonnxruntime") && name.ends_with(".dylib"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(unix, not(target_os = "macos")))]
    #[test]
    fn unix_runtime_names() {
        assert_eq!(ort_library_file(), "libonnxruntime.so");
        assert_eq!(library_search_var(), "LD_LIBRARY_PATH");
    }

    #[test]
    fn prepends_library_dir() {
        let dir = Path::new("/opt/semantix/lib");
        assert_eq!(prepend_search_path(dir, None), "/opt/semantix/lib");
        let sep = search_path_sep();
        assert_eq!(
            prepend_search_path(dir, Some("/usr/lib")),
            format!("/opt/semantix/lib{sep}/usr/lib")
        );
    }

    #[test]
    fn this_platform_has_a_pinned_runtime() {
        let asset = current_asset().unwrap();
        assert!(asset.archive.contains(ORT_VERSION), "{}", asset.archive);
        assert_eq!(asset.sha256.len(), 64);
    }

    #[test]
    fn install_dir_uses_the_exe_folder_when_it_is_writable() {
        let root = crate::db::scratch_dir("ort-dir");
        let exe = root.join("app");
        std::fs::create_dir_all(&exe).unwrap();
        let data = root.join("data");
        assert_eq!(install_dir(Some(&exe), &data), exe.join("lib"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn install_dir_falls_back_when_the_exe_folder_is_read_only() {
        use std::os::unix::fs::PermissionsExt;
        let root = crate::db::scratch_dir("ort-ro");
        let exe = root.join("app");
        std::fs::create_dir_all(&exe).unwrap();
        let mut perms = std::fs::metadata(&exe).unwrap().permissions();
        perms.set_mode(0o555);
        std::fs::set_permissions(&exe, perms.clone()).unwrap();
        let data = root.join("data");
        assert_eq!(install_dir(Some(&exe), &data), data.join("lib"));
        perms.set_mode(0o755);
        std::fs::set_permissions(&exe, perms).unwrap();
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn runtime_blob_is_the_real_library() {
        let root = crate::db::scratch_dir("ort-blob");
        let lib = root.join("pkg").join("lib");
        std::fs::create_dir_all(&lib).unwrap();
        let real = lib.join("libonnxruntime.so.1.22.0");
        std::fs::write(&real, b"runtime-bytes").unwrap();
        std::os::unix::fs::symlink("libonnxruntime.so.1.22.0", lib.join("libonnxruntime.so")).unwrap();
        std::fs::write(lib.join("libonnxruntime_providers_shared.so"), b"shared").unwrap();
        let found = find_runtime_blob(&root).unwrap();
        assert_eq!(found, real);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    #[ignore = "downloads the official ONNX Runtime archive"]
    fn fetches_pinned_cpu_runtime() {
        let root = crate::db::scratch_dir("ort-fetch");
        download_runtime_to(&root).unwrap();
        let lib = root.join(ort_library_file());
        let len = std::fs::metadata(&lib).unwrap().len();
        assert!(len > 1_000_000, "{len}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
