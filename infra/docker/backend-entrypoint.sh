#!/bin/sh
set -e

MODEL_PATH="/app/models/classification/convnextv2_best.pt"
MODEL_URL="https://huggingface.co/vng0824/convnextv2_best/resolve/main/convnextv2_best.pt"

# Tự tải classifier nếu checkpoint chưa tồn tại
if [ ! -f "$MODEL_PATH" ]; then
    echo "Classification model not found. Downloading..."
    mkdir -p "$(dirname "$MODEL_PATH")"
    curl -L --fail --retry 3 "$MODEL_URL" -o "$MODEL_PATH"
fi

echo "Starting SnakeGuard API..."
exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
