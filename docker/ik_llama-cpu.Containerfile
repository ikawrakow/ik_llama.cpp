ARG UBUNTU_VERSION=24.04

# Stage 1: Build
FROM docker.io/ubuntu:$UBUNTU_VERSION AS build
ENV LLAMA_CURL=1
ENV LC_ALL=C.utf8
# Add the toggle for ccache
ARG USE_CCACHE=false
ENV CCACHE_DIR=/ccache
ENV CCACHE_UMASK=000
ENV CCACHE_MAXSIZE=1G

RUN apt-get update && \
    apt-get install -yq --no-install-recommends build-essential libcurl4-openssl-dev curl libgomp1 cmake ccache git && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY . /app
WORKDIR /app

# Use a cache mount for /ccache and .git to persist objects between builds
RUN --mount=type=cache,target=/ccache \
    --mount=type=bind,source=.git,target=.git \
    if [ "${USE_CCACHE}" = "true" ]; then \
        export PATH="/usr/lib/ccache:$PATH"; \
        echo "ccache enabled. Current stats:"; \
        ccache -s; \
    fi && \
    # Fetch full git history for accurate BUILD_NUMBER
    git fetch --unshallow 2>/dev/null || true && \
    cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=ON -DGGML_IQK_FA_ALL_QUANTS=ON && \
    cmake --build build --config Release -j$(nproc) && \
    if [ "${USE_CCACHE}" = "true" ]; then \
        echo "Build finished. Updated stats:"; \
        ccache -s; \
    fi

RUN mkdir -p /app/lib /app/build/src /app/full \
    && find build -name "*.so" -exec cp {} /app/lib \; \
    && find build -name "*.so" -exec cp {} /app/build/src \; \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

# Stage 2: Base
FROM docker.io/ubuntu:$UBUNTU_VERSION AS base
RUN apt-get update && \
    apt-get install -yq --no-install-recommends libgomp1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
COPY --from=build /app/lib/ /app

# Stage 3: Full
FROM base AS full
COPY --from=build /app/full /app
RUN mkdir -p /app/build/src
COPY --from=build /app/build/src /app/build/src
WORKDIR /app
RUN apt-get update && \
    apt-get install -yq --no-install-recommends git python3 python3-pip && \
    pip install --break-system-packages -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENTRYPOINT ["/app/full/tools.sh"]

# Stage 4: Server
FROM base AS server
ENV LLAMA_ARG_HOST=0.0.0.0
ENV LD_LIBRARY_PATH=/app/lib
COPY --from=build /app/lib /app/lib
COPY --from=build /app/full/llama-server /app/llama-server
WORKDIR /app
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD [ "curl", "-f", "http://localhost:8080/health" ]
ENTRYPOINT [ "/app/llama-server" ]

# Stage 5: Swap
FROM server AS swap
ARG LS_REPO=mostlygeek/llama-swap
ARG LS_VER=199
RUN curl -sSL "https://github.com/${LS_REPO}/releases/download/v${LS_VER}/llama-swap_${LS_VER}_linux_amd64.tar.gz" \
    -o "llama-swap_${LS_VER}_linux_amd64.tar.gz" \
    && tar -zxf "llama-swap_${LS_VER}_linux_amd64.tar.gz" \
    && rm "llama-swap_${LS_VER}_linux_amd64.tar.gz"
COPY --from=build /app/docker/ik_llama-cpu-swap.config.yaml /app/config.yaml
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD [ "curl", "-f", "http://localhost:8080"]
ENTRYPOINT [ "/app/llama-swap", "-config", "/app/config.yaml" ]