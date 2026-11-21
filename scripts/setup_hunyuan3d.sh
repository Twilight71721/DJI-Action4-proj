#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="$project_dir/.venv-hunyuan3d"
repo_dir="$project_dir/third_party/Hunyuan3D-2"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found in PATH." >&2
    exit 1
fi

mkdir -p "$project_dir/third_party"
if [[ ! -d "$repo_dir/.git" ]]; then
    git clone --depth 1 \
        https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git \
        "$repo_dir"
fi

uv venv --python 3.10 "$venv_dir"
uv pip install --python "$venv_dir/bin/python" \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$venv_dir/bin/python" \
    -r "$repo_dir/requirements.txt" \
    gradio==4.44.1 \
    huggingface-hub==0.36.0 \
    transformers==4.49.0 \
    diffusers==0.32.2 \
    pydantic==2.10.6 \
    fastapi==0.115.6 \
    starlette==0.41.3 \
    uvicorn==0.34.0 \
    'httpx[socks]'
uv pip install --python "$venv_dir/bin/python" --no-deps -e "$repo_dir"
uv pip check --python "$venv_dir/bin/python"

echo "Environment ready. Start the app with: ./scripts/run_hunyuan3d_mv.sh"
