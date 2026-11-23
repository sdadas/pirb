FROM vllm/vllm-openai:v0.32.0

ARG BIELIK_ONE_REVISION=b92c1118d273a5fcdd3f556afbf077a7f5735892

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRITON_CACHE_DIR=/cache/triton \
    VLLM_CACHE_ROOT=/cache/vllm

WORKDIR /opt/bielik-one
COPY bielik-one-otel.txt /opt/bielik-one-otel.txt
# Include the heads plugin for adapter packages as well as merged checkpoints.
RUN python3 -c "import urllib.request; urllib.request.urlretrieve('https://github.com/speakleash/bielik-one/archive/${BIELIK_ONE_REVISION}.tar.gz', '/tmp/bielik-one.tar.gz')" \
    && tar -xzf /tmp/bielik-one.tar.gz --strip-components=1 -C /opt/bielik-one \
    && printf 'vllm==0.32.0\n' > /tmp/bielik-constraints.txt \
    && cat /opt/bielik-one-otel.txt >> /tmp/bielik-constraints.txt \
    && python3 -m pip uninstall -y opentelemetry-exporter-otlp-common opentelemetry-exporter-http-transport \
    && python3 -m pip install -c /tmp/bielik-constraints.txt '.[vllm]' ./vllm-heads-plugin \
        -r /opt/bielik-one-otel.txt 'lmcache==0.5.6' \
    && python3 -m pip check \
    && bielik-one --help \
    && python3 -c "import bielik_one.launch, systemone_vllm_heads" \
    && rm /tmp/bielik-one.tar.gz

COPY serve_bielik_one.py /opt/serve_bielik_one.py
RUN python3 /opt/serve_bielik_one.py --help

LABEL org.opencontainers.image.source="https://github.com/speakleash/bielik-one" \
      org.opencontainers.image.revision="${BIELIK_ONE_REVISION}"

EXPOSE 8000
ENTRYPOINT ["python3", "/opt/serve_bielik_one.py"]
CMD ["--model", "speakleash/Bielik-One-4B-r1"]
