# Consume the pre-built cuda12 server image as our foundation
ARG BASE_SERVER_IMAGE
FROM ${BASE_SERVER_IMAGE}

# 1. Install huggingface-cli and the hf-xet parallel download acceleration framework
RUN apt-get update && \
    apt-get install -yq --no-install-recommends python3 python3-pip && \
    pip install --break-system-packages -U "huggingface_hub[cli]" hf-xet && \
    rm -rf /var/lib/apt/lists/*

# 2. Configure High-Performance hf-xet Environment Variables
ENV HF_XET_HIGH_PERFORMANCE=1

ENV HF_HOME=/models/.hf-cache
ENV HF_HUB_CACHE=/models/.hf-cache/hub

# 3. Handle the prebaked model allocation
WORKDIR /models
COPY ./models/model.gguf /models/model.gguf

# Set up runtime environmental overrides so llama-server defaults to the prebaked model
WORKDIR /app
ENV LLAMA_ARG_MODEL=/models/model.gguf
ENV LLAMA_ARG_HOST=::
ENV LLAMA_ARG_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
