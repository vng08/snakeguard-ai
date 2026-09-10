#!/bin/sh
set -e

YOLO_CROP_DIR="/app/dataset/yolo-crop"
TRAIN_DIR="$YOLO_CROP_DIR/train"
TRAIN_ZIP="/tmp/train.zip"
TRAIN_URL="https://huggingface.co/datasets/vng0824/snake-image/resolve/main/train.zip"

echo "=== SnakeGuard database initialization ==="

# 1. Chạy migration để tạo/cập nhật database schema
echo "[1/5] Running Alembic migrations..."
alembic upgrade head

# 2. Seed thông tin species
echo "[2/5] Seeding snake species..."
python -m scripts.species.seed_species

# 3. Tải bộ ảnh reference nếu Docker volume chưa có
if [ ! -d "$TRAIN_DIR" ] || [ -z "$(ls -A "$TRAIN_DIR" 2>/dev/null)" ]; then
    echo "[3/5] Downloading snake reference images..."

    mkdir -p "$YOLO_CROP_DIR"

    curl -L --fail --retry 3 \
        "$TRAIN_URL" \
        -o "$TRAIN_ZIP"

    echo "Extracting reference images..."
    unzip -q "$TRAIN_ZIP" -d "$YOLO_CROP_DIR"
    rm -f "$TRAIN_ZIP"
else
    echo "[3/5] Reference images already exist. Skipping download."
fi

# Kiểm tra sau khi giải nén
if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: $TRAIN_DIR not found after extraction."
    exit 1
fi

# 4. Seed ảnh và tạo SigLIP embeddings
echo "[4/5] Seeding and embedding snake images..."
python -m scripts.image_retrieval.seed_snake_images
python -m scripts.image_retrieval.embed_snake_images

# 5. Ingest knowledge chunks + embeddings
echo "[5/5] Ingesting knowledge documents..."
python -m scripts.knowledge.ingest_knowledge

echo "=== SnakeGuard database initialization complete ==="
