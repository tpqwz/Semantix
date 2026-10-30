#!/bin/sh
# Assemble a portable folder: release binary, CPU ONNX Runtime, optional models.
# CUDA provider libraries are not copied. Models are copied only when already
# present; the executable downloads them itself with `fetch-models`.
set -eu
cd "$(dirname "$0")"
root=$(pwd)
# Must match the ort crate's api-22 feature.
ort_version=1.22.0

os=$(uname -s)
arch=$(uname -m)
case "$os" in
  Linux)
    platform=linux-x64
    archive=tgz
    lib_file=libonnxruntime.so
    lib_glob='libonnxruntime.so*'
    bin_name=semantix
    ;;
  Darwin)
    case "$arch" in
      arm64) platform=osx-arm64 ;;
      x86_64) platform=osx-x86_64 ;;
      *) echo "unsupported arch $arch" >&2; exit 1 ;;
    esac
    archive=tgz
    lib_file=libonnxruntime.dylib
    lib_glob='libonnxruntime*.dylib'
    bin_name=semantix
    ;;
  MINGW*|MSYS*|CYGWIN*)
    platform=win-x64
    archive=zip
    lib_file=onnxruntime.dll
    lib_glob='onnxruntime.dll'
    bin_name=semantix.exe
    ;;
  *)
    echo "unsupported OS $os" >&2
    exit 1
    ;;
esac

cargo build --release
dest="$root/dist/semantix-$platform"
rm -rf "$dest"
mkdir -p "$dest/lib" "$dest/models"
cp "$root/target/release/$bin_name" "$dest/$bin_name"
if [ "$bin_name" = semantix ]; then
  chmod +x "$dest/$bin_name"
fi

runtime="${SEMANTIX_RUNTIME:-}"
if [ -z "$runtime" ] && [ -f "$root/runtime/lib/$lib_file" ]; then
  runtime="$root/runtime/lib"
fi
if [ -z "$runtime" ] && [ -f "$root/../st/starlign/runtime/lib/$lib_file" ]; then
  runtime="$root/../st/starlign/runtime/lib"
fi
if [ -z "$runtime" ] && [ -f "$root/../starlign/runtime/lib/$lib_file" ]; then
  runtime="$root/../starlign/runtime/lib"
fi
if [ -z "$runtime" ]; then
  url="https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/onnxruntime-${platform}-${ort_version}.${archive}"
  tmp=$(mktemp -d)
  echo "downloading $url"
  curl -fsSL -o "$tmp/ort.$archive" "$url"
  tar -xf "$tmp/ort.$archive" -C "$tmp"
  found=$(find "$tmp" -name "$lib_file" | head -n 1)
  if [ -z "$found" ]; then
    echo "ONNX Runtime archive has no $lib_file" >&2
    exit 1
  fi
  mkdir -p "$root/runtime/lib"
  # -L copies the real library under each soname, not a dangling symlink.
  cp -aL "$(dirname "$found")"/$lib_glob "$root/runtime/lib/"
  runtime="$root/runtime/lib"
  rm -rf "$tmp"
fi
cp -aL "$runtime"/$lib_glob "$dest/lib/"

models="${SEMANTIX_MODELS:-}"
if [ -z "$models" ] && [ -f "$root/models/det.onnx" ]; then
  models="$root/models"
fi
if [ -n "$models" ]; then
  for name in det.onnx rec.onnx cls.onnx dict.txt vision.onnx text.onnx vocab.json merges.txt; do
    if [ -f "$models/$name" ]; then
      cp "$models/$name" "$dest/models/$name"
    fi
  done
fi
: > "$dest/portable"
echo "packed $dest"
