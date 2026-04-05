FROM python:3.11-slim

# Keep Python output unbuffered in container logs.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# curl is required by the healthcheck command.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Runtime source code.
COPY src ./src
COPY inference.py ./inference.py
COPY README.md ./README.md
COPY openenv.yaml ./openenv.yaml

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]