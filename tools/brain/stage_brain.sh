#!/usr/bin/env bash
# Stage the build inputs that are not in git, from the repo root:
#   third_party/.../g1_body29_hand14.urdf  G1 URDF from unitreerobotics/xr_teleoperate @ 817fb00 (same rev as setup_instance.sh)
#   tools/brain/llama.cpp/   llama.cpp release b12110 (the source the final image was built from), source only
#   tools/brain/models/      Qwen3-VL-8B-Instruct Q8_0 + mmproj (Qwen/Qwen3-VL-8B-Instruct-GGUF), sha256-checked
# The docker build itself needs no internet after this. Usage: bash tools/brain/stage_brain.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
LLAMA_TAG=b12110
HF=https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/resolve/main
M=tools/brain/models
mkdir -p "$M"

U=third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf
if [ ! -f "$U" ]; then
  mkdir -p "$(dirname "$U")"
  curl -fL --retry 3 -o "$U" https://raw.githubusercontent.com/unitreerobotics/xr_teleoperate/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b/assets/g1/g1_body29_hand14.urdf
fi
echo "8bbf006633fc50b616f665c7a970780cc296577a0adfd7d28b049e751c238735  $U" | sha256sum -c -

if [ ! -f tools/brain/llama.cpp/CMakeLists.txt ]; then
  rm -rf tools/brain/llama.cpp
  git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp tools/brain/llama.cpp
  rm -rf tools/brain/llama.cpp/.git
fi

fetch() {  # file sha256
  if [ ! -f "$M/$1" ] || ! echo "$2  $M/$1" | sha256sum -c --quiet - 2>/dev/null; then
    curl -fL --retry 3 -o "$M/$1" "$HF/$1"
  fi
  echo "$2  $M/$1" | sha256sum -c -
}
fetch Qwen3VL-8B-Instruct-Q8_0.gguf        0d264b3941185d00a74f75c4245521dae088ff1efc90ab8d1754e83f5844adb0
fetch mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf c6ba85508d82f42590e6eb77d5340369ab6fecf107a7561d809523d8aa5f3bfd
echo "staged: $U, tools/brain/llama.cpp ($LLAMA_TAG), $M/*.gguf"
