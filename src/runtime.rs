//! Locate ONNX Runtime and re-exec so the dynamic linker can see it.
//!
//! The search path is read at process start, so it has to be set before the
//! library is opened. CPU inference uses `libonnxruntime` only. CUDA provider
//! libraries are loaded later, and only when that device is requested.

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
    if cuda {
        preload_cuda();
    }
    let builder = if let Some(dir) = runtime_lib_dir() {
        ort::init_from(dir.join(ort_library_file())).map_err(|err| anyhow!("loading ONNX Runtime: {err}"))?
    } else {
        ort::init()
    };
    let env = builder
        .with_name("semantix")
        .with_telemetry(false)
        .build()
        .map_err(|err| anyhow!("onnx runtime: {err}"))?;
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
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runtime/lib"));
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../st/starlign/runtime/lib"));
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../starlign/runtime/lib"));
    candidates.into_iter().find(|dir| dir.join(&library).is_file())
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
}
