FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-live.txt ./

# Build with --build-arg INSTALL_LIVE_DEPS=true once you're ready to move
# past paper trading (adds solana/ccxt/web3 for real swap execution).
ARG INSTALL_LIVE_DEPS=false
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_LIVE_DEPS" = "true" ]; then pip install -r requirements-live.txt; fi

COPY app ./app
COPY scripts ./scripts

RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
