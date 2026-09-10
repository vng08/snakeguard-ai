FROM python:3.12-slim

WORKDIR /app

# Thư viện hệ thống cần cho OpenCV/Ultralytics và tải model
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r backend-requirements.txt

COPY . .

RUN chmod +x \
    /app/infra/docker/backend-entrypoint.sh \
    /app/infra/docker/db-init.sh

EXPOSE 8000

ENTRYPOINT ["/app/infra/docker/backend-entrypoint.sh"]
