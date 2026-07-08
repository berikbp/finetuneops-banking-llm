FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

COPY pyproject.toml uv.lock README.md ./

RUN pip install uv

RUN uv python install 3.12 && uv sync --frozen --python 3.12

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
