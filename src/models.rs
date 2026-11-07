//! Verified download of the OCR and CLIP graphs.
//!
//! Files land in the model directory. A file whose SHA-256 already matches is
//! left alone, so a second launch does not touch the network.

use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

pub struct ModelFile {
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
}

pub const FILES: &[ModelFile] = &[
    ModelFile {
        name: "det.onnx",
        url: "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx",
        sha256: "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae",
    },
    ModelFile {
        name: "rec.onnx",
        url: "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec/en_PP-OCRv5_rec_mobile.onnx",
        sha256: "c3461add59bb4323ecba96a492ab75e06dda42467c9e3d0c18db5d1d21924be8",
    },
    ModelFile {
        name: "cls.onnx",
        url: "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        sha256: "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    },
    ModelFile {
        name: "dict.txt",
        url: "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv5_en_dict.txt",
        sha256: "e025a66d31f327ba0c232e03f407ae8d105e1e709e7ccb3f408aa778c24e70d6",
    },
    ModelFile {
        name: "vocab.json",
        url: "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/vocab.json",
        sha256: "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
    },
    ModelFile {
        name: "merges.txt",
        url: "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/merges.txt",
        sha256: "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    },
    ModelFile {
        name: "vision.onnx",
        url: "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx",
        sha256: "583fd1110a514667812fee7d684952aaf82a99b959760c8d7dca7e0ab9839299",
    },
    ModelFile {
        name: "text.onnx",
        url: "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model_quantized.onnx",
        sha256: "73baab855d406190da9faa498cfedf65f15cf309f4cc7385b7b032e6d08e5c3a",
    },
];

#[derive(Clone, Debug)]
pub struct ModelStatus {
    pub ocr: bool,
    pub clip: bool,
    pub cls: bool,
    pub missing: Vec<String>,
}

pub fn status(dir: &Path) -> ModelStatus {
    let has = |name: &str| dir.join(name).is_file();
    let ocr = has("det.onnx") && has("rec.onnx");
    let clip = has("vision.onnx") && has("text.onnx") && has("vocab.json") && has("merges.txt");
    let cls = has("cls.onnx");
    let mut missing = Vec::new();
    for file in FILES {
        if file.name == "dict.txt" || file.name == "cls.onnx" {
            continue;
        }
        if !has(file.name) {
            missing.push(file.name.to_string());
        }
    }
    ModelStatus { ocr, clip, cls, missing }
}

pub fn fetch(dir: &Path, mut note: impl FnMut(&str)) -> Result<ModelStatus> {
    std::fs::create_dir_all(dir)?;
    for file in FILES {
        let dest = dir.join(file.name);
        if dest.is_file() && hash_of(&dest).ok().as_deref() == Some(file.sha256) {
            note(&format!("{} already matches", file.name));
            continue;
        }
        note(&format!("downloading {}", file.name));
        download(&dest, file)?;
        note(&format!("verified {}", file.name));
    }
    Ok(status(dir))
}

fn download(dest: &Path, file: &ModelFile) -> Result<()> {
    let partial = partial_path(dest);
    // A quiet socket fails. Bytes that keep arriving do not, so a large
    // graph can finish on a slow link.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(120))
        .build();
    let response = agent
        .get(file.url)
        .call()
        .map_err(|err| anyhow!("GET {}: {err}", file.url))?;
    let mut reader = response.into_reader();
    let mut out = File::create(&partial)?;
    std::io::copy(&mut reader, &mut out)?;
    out.sync_all()?;
    drop(out);
    let got = hash_of(&partial)?;
    if got != file.sha256 {
        let _ = std::fs::remove_file(&partial);
        return Err(anyhow!(
            "{} hash mismatch: got {got}, expected {}",
            file.name,
            file.sha256
        ));
    }
    std::fs::rename(&partial, dest)?;
    Ok(())
}

fn partial_path(dest: &Path) -> PathBuf {
    let name = dest.file_name().unwrap_or_default().to_string_lossy();
    dest.with_file_name(format!("{name}.partial"))
}

pub fn hash_of(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn file_sig(path: &Path) -> String {
    match std::fs::metadata(path) {
        Ok(meta) => format!("{}:{}", meta.len(), crate::imageutil::mtime_ns(&meta)),
        Err(_) => "missing".to_string(),
    }
}
