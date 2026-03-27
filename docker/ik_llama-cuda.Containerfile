ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.6.2
ARG BASE_CUDA_DEV_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Stage 1: Build
FROM ${BASE_CUDA_DEV_CONTAINER} AS build
ARG CUDA_DOCKER_ARCH=86 # CUDA architecture to build for
# Add the toggle for ccache
ARG USE_CCACHE=false
ENV CCACHE_DIR=/ccache
ENV CCACHE_UMASK=000
ENV CCACHE_MAXSIZE=1G

# Install build dependencies + ccache
RUN apt-get update && apt-get install -yq build-essential libcurl4-openssl-dev curl libgomp1 cmake ccache git libnccl-dev

COPY . /app
WORKDIR /app

# We use a cache mount for /ccache and .git to persist objects between builds
RUN --mount=type=cache,target=/ccache \
    --mount=type=bind,source=.git,target=.git \
    if [ "${USE_CCACHE}" = "true" ]; then \
        export PATH="/usr/lib/ccache:$PATH"; \
        echo "ccache enabled. Current stats:"; \
        ccache -s; \
    fi && \
    if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON -DLLAMA_CURL=ON ${CMAKE_ARGS} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build --config Release -j$(nproc) && \
    if [ "${USE_CCACHE}" = "true" ]; then \
        echo "Build finished. Updated stats:"; \
        ccache -s; \
    fi

RUN mkdir -p /app/lib && \
    find build -name "*.so" -exec cp {} /app/lib \;
RUN mkdir -p /app/build/src && \
    find build -name "*.so" -exec cp {} /app/build/src \;
RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

# Stage 2: base
FROM ${BASE_CUDA_RUN_CONTAINER} AS base
RUN apt-get update && apt-get install -yq libgomp1 curl || true

# Stage 3: full
FROM base AS full
COPY --from=build /app/full /app
RUN mkdir -p /app/build/src
COPY --from=build /app/build/src /app/build/src
WORKDIR /app
RUN apt-get update && apt-get install -yq git python3 python3-pip || true \
    && pip3 install --break-system-packages -r requirements.txt || true
ENTRYPOINT ["/app/tools.sh"]

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
COPY --from=build /app/docker/ik_llama-cuda-swap.config.yaml /app/config.yaml
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD [ "curl", "-f", "http://localhost:8080"]
ENTRYPOINT [ "/app/llama-swap", "-config", "/app/config.yaml" ]