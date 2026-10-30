#!/bin/sh
# Assemble a portable Linux folder: binary, CPU ONNX Runtime, optional models.
# CUDA provider libraries are not copied.
set -eu
cd "$(dirname "$0")"
cargo build --release
root=$(pwd)
dest="$root/dist/semantix-linux-x64"
rm -rf "$dest"
mkdir -p "$dest/lib" "$dest/models"
cp "$root/target/release/semantix" "$dest/semantix"
chmod +x "$dest/semantix"

runtime="${SEMANTIX_RUNTIME:-}"
if [ -z "$runtime" ] && [ -f "$root/runtime/lib/libonnxruntime.so" ]; then
  runtime="$root/runtime/lib"
fi
if [ -z "$runtime" ] && [ -f "$root/../st/starlign/runtime/lib/libonnxruntime.so" ]; then
  runtime="$root/../st/starlign/runtime/lib"
fi
if [ -z "$runtime" ] && [ -f "$root/../starlign/runtime/lib/libonnxruntime.so" ]; then
  runtime="$root/../starlign/runtime/lib"
fi
if [ -n "$runtime" ]; then
  # -L copies the real library under each soname, not a dangling symlink.
  cp -aL "$runtime"/libonnxruntime.so* "$dest/lib/"
fi

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
