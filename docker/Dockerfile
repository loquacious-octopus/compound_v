FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ARG VLLM_VERSION=0.20.0
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      fonts-liberation \
      fonts-noto-color-emoji \
      libasound2 \
      libatk-bridge2.0-0 \
      libgbm1 \
      libgtk-3-0 \
      libnss3 \
      libx11-xcb1 \
      libxcomposite1 \
      libxdamage1 \
      libxrandr2 \
      xz-utils \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==2.2.1 \
    && poetry config virtualenvs.create false

ARG VLLM_WHEEL_URL=https://github.com/vllm-project/vllm/releases/download/v0.20.0/vllm-0.20.0%2Bcu129-cp38-abi3-manylinux_2_31_x86_64.whl

RUN pip install --no-cache-dir --force-reinstall \
      --index-url "${TORCH_CUDA_INDEX_URL}" \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" \
    && pip install --no-cache-dir \
      --extra-index-url "${TORCH_CUDA_INDEX_URL}" \
      "${VLLM_WHEEL_URL}" \
    && python -c "import torch, vllm; assert torch.__version__.startswith('${TORCH_VERSION}+cu128'), torch.__version__; assert torch.version.cuda == '12.8', torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__)"

COPY miner-reference/pyproject.toml miner-reference/poetry.lock* /app/miner-reference/
WORKDIR /app/miner-reference
RUN poetry install --without dev --no-root --no-interaction

COPY miner-reference/validator/package*.json /app/miner-reference/validator/
WORKDIR /app/miner-reference/validator
RUN npm ci

COPY miner-reference /app/miner-reference
COPY docker/start_miner.sh /app/start_miner.sh
RUN chmod +x /app/start_miner.sh

ENV PYTHONUNBUFFERED=1 \
    MINER_PROJECT_ROOT=/app/miner-reference \
    MINER_GENERATOR_MODE=ensemble \
    ENSEMBLE_RENDER=0 \
    ENSEMBLE_CRITIQUE=0 \
    ENSEMBLE_TRACKS=baseline_qwen25 \
    ENSEMBLE_CANDIDATE_COUNT=1 \
    ENSEMBLE_INCLUDE_OBJECT_IDENTITY=0 \
    ENSEMBLE_VALIDATOR_REPAIR=1 \
    ENSEMBLE_ALLOW_PLACEHOLDER_FALLBACK=0 \
    ENSEMBLE_INCLUDE_STYLE_EXAMPLE=0 \
    ENSEMBLE_ANALYSIS_MAX_TOKENS=1024 \
    ENSEMBLE_CODE_MAX_TOKENS=4096 \
    ENSEMBLE_REPAIR_MAX_TOKENS=4096 \
    MINER_INFERENCE_BACKEND=local_vllm \
    LOCAL_LLM_START_MODE=managed \
    LOCAL_LLM_PRELOAD_MODEL=vision \
    LOCAL_MANAGED_CUDA_DEVICES=0,1,2,3 \
    LOCAL_MANAGED_TENSOR_PARALLEL_SIZE=4 \
    PRODUCTION_VISION_MODEL=Qwen/Qwen3-VL-30B-A3B-Thinking \
    PRODUCTION_CODE_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0 \
    VLLM_DEEP_GEMM_WARMUP=skip \
    CHUTES_API_KEY=local-vllm

WORKDIR /app/miner-reference
EXPOSE 10006

CMD ["/app/start_miner.sh"]
