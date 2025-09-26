FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# LightGBM runtime deps + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libstdc++6 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload NLTK data used by your app
RUN python - <<'PY'
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
PY

# App code
COPY . .

EXPOSE 5000

# Optional: container healthcheck hits your /health endpoint
HEALTHCHECK --interval=30s --timeout=3s --retries=5 \
  CMD curl -fsS http://localhost:5000/health || exit 1

CMD ["python", "-u", "app.py"]
