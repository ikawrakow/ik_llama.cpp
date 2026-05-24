ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.6.2
ARG BASE_CUDA_DEV_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Stage 1: Build
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

ARG CUDA_DOCKER_ARCH="61-real"
ARG GGML_NATIVE=OFF
ARG USE_CCACHE=true

ENV CCACHE_DIR=/ccache
ENV CCACHE_UMASK=000
ENV CCACHE_MAXSIZE=5G
ENV CCACHE_COMPRESS=1
ENV CCACHE_BASEDIR=/app

RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    ca-certificates build-essential libcurl4-openssl-dev curl libgomp1 cmake ccache git && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

RUN --mount=type=cache,target=/ccache \
    --mount=type=bind,source=.git,target=.git \
    if [ "${USE_CCACHE}" = "true" ]; then \
        export PATH="/usr/lib/ccache:$PATH"; \
        ccache -z; \
    fi && \
    cmake -B build \
        -DGGML_NATIVE=${GGML_NATIVE} \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" \
        -DLLAMA_CURL=ON \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build --config Release -j$(nproc) && \
    if [ "${USE_CCACHE}" = "true" ]; then \
        ccache -s; \
    fi

RUN mkdir -p /app/dist/lib /app/dist/bin && \
    find build -name "*.so" -exec cp {} /app/dist/lib \; && \
    cp build/bin/* /app/dist/bin/

# Stage 2: Runtime (only server)
FROM ${BASE_CUDA_RUN_CONTAINER} AS server

RUN apt-get update && \
    apt-get install -yq --no-install-recommends libgomp1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV LLAMA_ARG_HOST=0.0.0.0
ENV LD_LIBRARY_PATH=/app/lib

COPY --from=build /app/dist/lib /app/lib
COPY --from=build /app/dist/bin/llama-server /app/llama-server

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
