FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ARG BASELABS_RUNTIME_REPO=baselabs/mammoth-1
ARG BASELABS_RUNTIME_REVISION=61cec44c281c693992e3d6ebeb44dd3402dfa186
ARG MAX_JOBS=4

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRITON_CACHE_DIR=/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/baselabs
# Download only installation files. The pinned upstream setup applies serve.patch
# to its pinned OneJev commit and preserves the base image's CUDA PyTorch build.
# Model weights are fetched by qev only when the container starts.
RUN python -c "import urllib.request; base='https://huggingface.co/${BASELABS_RUNTIME_REPO}/resolve/${BASELABS_RUNTIME_REVISION}/'; [urllib.request.urlretrieve(base + name, name) for name in ('setup.sh', 'serve.patch', 'constraints.txt')]" \
    && bash setup.sh /opt/baselabs \
    && python -c "import qev.cli, qev.server, qev.readout, causal_conv1d, fla" \
    && qev serve --help

LABEL org.opencontainers.image.source="https://huggingface.co/${BASELABS_RUNTIME_REPO}" \
      org.opencontainers.image.revision="${BASELABS_RUNTIME_REVISION}"

EXPOSE 8000
ENTRYPOINT ["qev", "serve", "--host", "0.0.0.0", "--port", "8000", "--max-branch-tokens", "16384", "--max-request-tokens", "16384"]
CMD ["--model", "baselabs/mammoth-1"]
