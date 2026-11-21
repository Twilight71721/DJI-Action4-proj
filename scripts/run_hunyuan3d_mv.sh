#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$project_dir/.venv-hunyuan3d/bin/python"
repo_dir="$project_dir/third_party/Hunyuan3D-2"
output_dir="$project_dir/outputs/hunyuan3d/gradio_cache"
model_cache_dir="$project_dir/.cache/huggingface"
hy3dgen_model_dir="$project_dir/.cache/hy3dgen"
port="${HUNYUAN3D_PORT:-8080}"

if [[ ! -x "$python_bin" ]]; then
    echo "Hunyuan3D environment not found: $python_bin" >&2
    exit 1
fi

mkdir -p "$output_dir" "$model_cache_dir" "$hy3dgen_model_dir"
export HF_HOME="$model_cache_dir"
export HY3DGEN_MODELS="$hy3dgen_model_dir"
export GRADIO_ANALYTICS_ENABLED=False

cd "$repo_dir"
exec "$python_bin" gradio_app.py \
    --model_path tencent/Hunyuan3D-2mv \
    --subfolder hunyuan3d-dit-v2-mv-turbo \
    --disable_tex \
    --low_vram_mode \
    --enable_flashvdm \
    --host 127.0.0.1 \
    --port "$port" \
    --cache-path "$output_dir"
