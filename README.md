# Semantix

Version 0.1.0-alpha.2.

Offline desktop gallery. Point it at a folder of photos, screenshots, and scans. It indexes them on this machine: thumbnails, EXIF, PaddleOCR text, and CLIP embeddings. Search a receipt number, describe a picture, or drop an image to find similar ones.

Nothing leaves the computer after the models are present. There is no account and no telemetry.

## Download

[v0.1.0-alpha.2](https://github.com/tpqwz/semantix/releases/tag/v0.1.0-alpha.2) is one executable per platform. The file is the program.

| Platform | File |
| --- | --- |
| Linux x64 | [semantix-linux-x64](https://github.com/tpqwz/semantix/releases/download/v0.1.0-alpha.2/semantix-linux-x64) |
| macOS arm64 | [semantix-osx-arm64](https://github.com/tpqwz/semantix/releases/download/v0.1.0-alpha.2/semantix-osx-arm64) |
| Windows x64 | [semantix-win-x64.exe](https://github.com/tpqwz/semantix/releases/download/v0.1.0-alpha.2/semantix-win-x64.exe) |

On Linux and macOS, `chmod +x` the download before the first launch. The first OCR or CLIP run downloads ONNX Runtime. Download models, or `semantix fetch-models`, downloads the model files. Both are checked against a published SHA-256.

The window is [egui](https://github.com/emilk/egui). Indexing, ONNX Runtime, SQLite, and embedding search live in this Rust crate. The portable program is the binary. A `portable` file beside it keeps the library data next to the executable. The binary downloads ONNX Runtime and the models when they are not already on disk.

## Build

```sh
cd semantix
cargo build --release
```

ONNX Runtime is loaded from the first of these that already contains the library:

- `SEMANTIX_RUNTIME`
- `lib/` or `runtime/lib/` next to the executable
- `lib/` inside the data directory
- `runtime/lib/` inside this crate
- `../st/starlign/runtime/lib` when this crate sits in `~/code` next to the Starlign checkout

If none of those exist, the first OCR or CLIP load downloads the official CPU ONNX Runtime 1.22 archive, checks its SHA-256, and saves the library. A writable folder next to the executable gets `lib/`. Otherwise the file goes under the data directory. CUDA provider libraries are not downloaded. `--cuda` uses them only when they are already in that same directory. If CUDA session setup fails, restart without `--cuda`. This process will not open a replacement CPU session after that failure.

## Models

The default graphs are PP-OCRv5 mobile detection, the English PP-OCRv5 recognizer, the PP-OCRv4 angle classifier, and OpenAI CLIP ViT-B/32 (quantized ONNX). CLIP is the MIT-licensed stand-in for a MobileCLIP / OpenCLIP variant: MobileCLIP’s Apple license is not a standard permissive grant.

```sh
semantix fetch-models
```

That writes `det.onnx`, `rec.onnx`, `cls.onnx`, `dict.txt`, `vision.onnx`, `text.onnx`, `vocab.json`, and `merges.txt` under the model directory and checks the published SHA-256 of each file. A file that already matches is skipped. The window has the same Download button.

Model directory, first match:

- `--models` or `SEMANTIX_MODELS`
- `models/` next to the executable when `det.onnx` or `vision.onnx` is there
- `<data>/models`

The English character list is also compiled into the binary (`assets/ppocrv5_en_dict.txt`). A `dict.txt` beside the models replaces it.

## Use

```sh
semantix                         # window
semantix index ~/Pictures
semantix search 1042
semantix search --semantic "dog on beach"
semantix search --hybrid "alice receipt"
semantix similar photo.jpg
semantix duplicates --threshold 0.95
semantix save-search receipts --hybrid alice 1042
semantix search --saved receipts
semantix collection keepers --add photo.jpg
semantix collection keepers --remove photo.jpg
semantix collection keepers
semantix --data ./library --cuda index ~/Scans
```

Search prints `score`, path, and snippet, tab-separated. No hits is still a successful exit. `duplicates` prints each member of a near-duplicate group. `collection` with no `--add` prints the paths already in that collection.

In the window: Index walks the folder, Pause and Cancel apply to the background worker, and the grid is virtualized. Video samples stay out of the photo grid until Frames is on. Click a folder in the side panel to show only that folder; click it again for the whole library. Search still looks through frames. The grid orders a picture by its stored EXIF capture time when it has one, and by file modification time otherwise, so a copied camera roll stays in the order the photos were taken. Text search hits SQLite FTS5 immediately. Semantic and Hybrid wait a quarter second, or Enter. The Duplicates tab groups pictures whose CLIP cosine is at least 0.95. The side panel saves the current search, or collects the selected image (otherwise the cards on screen). Drop a folder to index it, or drop one image to search for similar pictures. Arrow keys move the selection when the search box is unfocused. Enter opens the viewer. Esc closes it.

A file is indexed again when its size or modification time changes. A rename that keeps the same bytes reuses the OCR text and the embedding. Hidden files and directories are skipped. Supported pictures are JPEG, PNG, WebP, GIF, BMP, and TIFF. A JPEG's stored capture time prefers DateTimeOriginal, then DateTimeDigitized, then DateTime. Video files (`mp4`, `mkv`, `webm`, `mov`, `m4v`, `avi`) are sampled with the system `ffmpeg` when it is on PATH: up to eight frames, spread through the clip, stored under `frames/` and indexed like stills. The app does not bundle ffmpeg. If `ffmpeg` is missing or fails to start, videos are skipped and pictures are still indexed. Frames are reused until the video's size or modification time changes. They keep that video's modification time, so sampling does not pin them to the top of the gallery. A frame is labeled `clip.mp4 @ 1.2s` instead of the cache file name. The side panel lists the other samples from that clip. Show clip fills the grid in time order, and in the viewer the left and right keys step through them. Each near-duplicate thumbnail carries its group number and a colored bar. The Duplicates tab has a cosine slider. A collection can drop the selected picture without deleting the collection.

## Data

| File | Contents |
| --- | --- |
| `library.db` | Paths, EXIF JSON, status, FTS5 OCR text, saved searches, collections |
| `thumbs/` | JPEG thumbnails, named by content hash |
| `embeddings.bin` | int8 CLIP vectors, one slot per file id |
| `frames/` | JPEG samples from videos, plus a manifest and a source sidecar |

Default data directory:

- `data/` next to the executable when a `portable` file is present
- otherwise `$XDG_DATA_HOME/semantix`, `~/.local/share/semantix`, `~/Library/Application Support/semantix`, or `%LOCALAPPDATA%\semantix`

`--data` and `SEMANTIX_DATA` override that.

## Portable executable

The release build is the `semantix` binary. It does not need a pack step. Put an empty `portable` file beside it when the database and thumbnails should stay in `data/` next to that binary. The first time OCR or CLIP runs, the binary downloads ONNX Runtime and, if you use Download models or `semantix fetch-models`, the model files. Both are hash-checked. A later launch uses the copies on disk and does not touch the network.

GitHub Actions builds that binary for Linux x64, Windows x64, and macOS arm64 on every push to `main`. A `v*` tag publishes the executable itself: `semantix-linux-x64`, `semantix-osx-arm64`, and `semantix-win-x64.exe`. Weights and the runtime are not in the download. On Linux and macOS, `chmod +x` the file before the first launch.

## License

The application is MIT. Model and runtime attributions are in `NOTICE`.
