ARG UBUNTU_VERSION=24.04

# Stage 1: Build
FROM docker.io/ubuntu:$UBUNTU_VERSION AS build

# Build arguments
ARG GGML_NATIVE=ON
ARG GGML_AVX2=ON
ARG USE_CCACHE=true

# Environment variables for portability and GitHub Actions
ENV LLAMA_CURL=1
ENV LC_ALL=C.utf8

# ccache configuration
ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=1G
ENV CCACHE_COMPRESS=1
ENV CCACHE_COMPRESSLEVEL=6
# This is CRITICAL for GitHub Actions: it ignores the absolute path of the runner
ENV CCACHE_BASEDIR=/app

RUN apt-get update && \
    apt-get install -yq --no-install-recommends ca-certificates build-essential libcurl4-openssl-dev curl libgomp1 cmake ccache git && \
    rm -rf /var/lib/apt/lists/*

# Copy source code (excluding hidden files/dirs via .dockerignore)
COPY . /app

WORKDIR /app

# Build using ccache and optional custom commit
RUN --mount=type=cache,target=/ccache \
    --mount=type=bind,source=.git,target=.git \
    if [ "${USE_CCACHE}" = "true" ]; then \
        export PATH="/usr/lib/ccache:$PATH"; \
        ccache -z; \
    fi && \
    cmake -B build \
        -DGGML_NATIVE=${GGML_NATIVE} \
        -DLLAMA_CURL=ON && \
    cmake --build build --config Release -j$(nproc) && \
    if [ "${USE_CCACHE}" = "true" ]; then \
        ccache -s; \
    fi

# Collect build artifacts
RUN mkdir -p /app/dist/lib /app/dist/full /app/dist/bin && \
    find build -name "*.so" -exec cp {} /app/dist/lib \; && \
    cp build/bin/* /app/dist/bin/ && \
    cp build/bin/* /app/dist/full/ && \
    cp *.py /app/dist/full/ && \
    cp -r gguf-py /app/dist/full/ && \
    cp -r requirements /app/dist/full/ && \
    cp requirements.txt /app/dist/full/ && \
    cp .devops/tools.sh /app/dist/full/

# Stage 2: Base (Shared Runtime)
FROM docker.io/ubuntu:$UBUNTU_VERSION AS base
RUN apt-get update && \
    apt-get install -yq --no-install-recommends libgomp1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV LD_LIBRARY_PATH=/app/lib
COPY --from=build /app/dist/lib /app/lib

# Stage 3: Full (Python/Dev Tools)
FROM base AS full
COPY --from=build /app/dist/full /app
RUN apt-get update && \
    apt-get install -yq --no-install-recommends git python3 python3-pip && \
    pip install --break-system-packages -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["/app/tools.sh"]

# Stage 4: Server
FROM base AS server
ENV LLAMA_ARG_HOST=0.0.0.0
COPY --from=build /app/dist/bin/llama-server /app/llama-server
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD [ "curl", "-f", "http://localhost:8080/health" ]
ENTRYPOINT [ "/app/llama-server" ]

# Stage 5: Swap
FROM server AS swap
ARG LS_REPO=mostlygeek/llama-swap
ARG LS_VER=199
RUN curl -sSL "https://github.com/${LS_REPO}/releases/download/v${LS_VER}/llama-swap_${LS_VER}_linux_amd64.tar.gz" \
    | tar -xz

COPY --from=build /app/docker/ik_llama-cpu-swap.config.yaml /app/config.yaml
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD [ "curl", "-f", "http://localhost:8080"]
ENTRYPOINT [ "/app/llama-swap", "-config", "/app/config.yaml" ]