FROM python:3.12-slim-bookworm

ARG BASAL_REVISION=14ec0727ea757d5c6a4e9322858fc735cc06ca78
ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor

RUN pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
    && pip install "basal @ https://github.com/rkinas/basal/archive/${BASAL_REVISION}.tar.gz" \
    && pip check

LABEL org.opencontainers.image.source="https://github.com/rkinas/basal" \
      org.opencontainers.image.revision="${BASAL_REVISION}"

EXPOSE 8000
ENTRYPOINT ["basal-serve"]
CMD ["--model", "Remek/basal-2.0-mini", "--mode", "eager", "--device", "cuda", "--host", "0.0.0.0", "--port", "8000", "--context-overflow", "refuse"]
