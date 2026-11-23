FROM python:3.12-slim

ARG KARPO_REVISION=c52d3507a254d3ad8423726c6bc6568c99113e7d

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    TRITON_CACHE_DIR=/cache/triton

# Triton needs a C compiler to build its launchers at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/karpo-decision
RUN python -c "from urllib.request import urlretrieve; urlretrieve('https://github.com/machinepulse-ai/karpo-decision/archive/${KARPO_REVISION}.tar.gz', '/tmp/karpo.tar.gz')" \
    && tar -xzf /tmp/karpo.tar.gz --strip-components=1 \
    && rm /tmp/karpo.tar.gz \
    && python -m pip install -r requirements-serve.txt \
    && python -m pip install --no-deps . \
    && python -m pip check \
    && python -m karpo_decision.serve.server --help

COPY serve_karpo.py /opt/serve_karpo.py
RUN python /opt/serve_karpo.py --help

LABEL org.opencontainers.image.source="https://github.com/machinepulse-ai/karpo-decision" \
      org.opencontainers.image.revision="${KARPO_REVISION}"

EXPOSE 18440
ENTRYPOINT ["python", "/opt/serve_karpo.py"]
