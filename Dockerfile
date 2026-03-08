FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the package into /app/overflow_env
COPY . /app/overflow_env

# Install runtime dependencies (no openenv-core to avoid websockets conflict)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "fastapi>=0.115.0" \
    "pydantic>=2.0.0" \
    "uvicorn[standard]>=0.24.0" \
    "requests>=2.31.0" \
    "torch==2.5.1+cpu" \
    "numpy>=1.24.0" \
    "gymnasium>=0.29.0" \
    "matplotlib>=3.8.0" \
    "pillow==10.4.0"

# Make overflow_env importable as a top-level package
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "overflow_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
