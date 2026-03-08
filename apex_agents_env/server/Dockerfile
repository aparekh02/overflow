FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy environment code into a proper package directory
COPY . /app/apex_agents_env

# Install dependencies via pip using requirements.txt
RUN pip install --no-cache-dir -r /app/apex_agents_env/server/requirements.txt

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "apex_agents_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
