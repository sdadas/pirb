FROM python:3.12-slim-bookworm

ARG GLIDE_CODE_REPO=fastino/GLiDE-27B
ARG GLIDE_CODE_REVISION=4d32743ae96b74f0116f050d37742bbd43c04e7d

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRITON_CACHE_DIR=/cache/triton \
    GLIDE_MODEL=fastino/GLiDE-27B

# Triton compiles GPU kernels at runtime. PyTorch's pip dependencies supply CUDA.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/glide
# Download only the official runtime, dependencies and license files, not weights.
RUN pip install "huggingface_hub==1.33.0" \
    && hf download "${GLIDE_CODE_REPO}" --revision "${GLIDE_CODE_REVISION}" \
        --local-dir /opt/glide --include "glide/*.py" "requirements.txt" "LICENSE" "NOTICE" "licenses/*" \
    && pip install -r requirements.txt \
    && pip check

LABEL org.opencontainers.image.source="https://huggingface.co/${GLIDE_CODE_REPO}" \
      org.opencontainers.image.revision="${GLIDE_CODE_REVISION}"

EXPOSE 8000
ENTRYPOINT ["python", "-m", "glide.server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
