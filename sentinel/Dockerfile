FROM python:3.11-slim

# Build-time deps for native extensions (numpy, tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer caching: dependencies change less than code)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (including pyproject.toml and sentinel code)
COPY . .

# Install the package in editable mode so absolute imports like 'from lydian.x' work
RUN pip install --no-cache-dir -e .

# Ensure the app is in the python path
ENV PYTHONPATH=/app

# Create directories that must exist at runtime
RUN mkdir -p /app/data/lancedb /app/data/models

# Non-root user
RUN useradd -m -u 1000 sentinel
RUN chown -R sentinel:sentinel /app
USER sentinel

# Expose FastAPI port
EXPOSE 8000

# Seed DB then start the server.
# Using 'python -m sentinel.storage.seed' works because the package is installed -e.
CMD ["sh", "-c", "python -m sentinel.storage.seed && uvicorn sentinel.main:app --host 0.0.0.0 --port 8000 --workers 1"]
