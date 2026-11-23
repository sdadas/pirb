FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

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
# Install the pinned Qev runtime for prompt rendering and the System One API.
# Download only setup files here; the launcher fetches model weights at startup.
RUN python -c "from urllib.request import urlretrieve; base='https://huggingface.co/baselabs/mammoth-1/resolve/${BASELABS_RUNTIME_REVISION}/'; [urlretrieve(base + name, name) for name in ('setup.sh', 'serve.patch', 'constraints.txt')]" \
    && bash setup.sh /opt/baselabs

LABEL org.opencontainers.image.source="https://huggingface.co/baselabs/mammoth-1" \
      org.opencontainers.image.revision="${BASELABS_RUNTIME_REVISION}"

COPY serve_baselabs_gemma.py /opt/serve_baselabs_gemma.py
RUN python /opt/serve_baselabs_gemma.py --help \
    && python -c "from transformers import Gemma4ForConditionalGeneration; from qev.engine import BranchResult, DecisionEngine; from qev.readout import token_layout"

EXPOSE 8000
ENTRYPOINT ["python", "/opt/serve_baselabs_gemma.py"]
CMD ["--model", "baselabs/bison-1"]
