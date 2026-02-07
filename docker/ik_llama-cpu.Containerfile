ARG UBUNTU_VERSION=22.04

# Stage 1: Build
FROM docker.io/ubuntu:$UBUNTU_VERSION AS build
ENV LLAMA_CURL=1
ENV LC_ALL=C.utf8
ARG CUSTOM_COMMIT

RUN apt-get update && apt-get install -yq build-essential git libcurl4-openssl-dev curl libgomp1 cmake
RUN git clone https://github.com/ikawrakow/ik_llama.cpp.git /app
WORKDIR /app
RUN if [ -n "$CUSTOM_COMMIT" ]; then git switch --detach "$CUSTOM_COMMIT"; fi
RUN cmake -B build -DGGML_NATIVE=OFF -DLLAMA_CURL=ON -DGGML_IQK_FA_ALL_QUANTS=ON && \
    cmake --build build --config Release -j$(nproc)
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

# Stage 2: Base
FROM docker.io/ubuntu:$UBUNTU_VERSION AS base
RUN apt-get update && apt-get install -yq libgomp1 curl \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete
COPY --from=build /app/lib/ /app

# Stage 3: Full
FROM base AS full
COPY --from=build /app/full /app
RUN mkdir -p /app/build/src
COPY --from=build /app/build/src /app/build/src
WORKDIR /app
RUN apt-get update && apt-get install -yq \
    git \
    python3 \
    python3-pip \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete
ENTRYPOINT ["/app/full/tools.sh"]

# Stage 4: Server
FROM base AS server
ENV LLAMA_ARG_HOST=0.0.0.0
COPY --from=build /app/full/llama-server /app/llama-server
WORKDIR /app
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]
ENTRYPOINT [ "/app/llama-server" ]

# Stage 5: Swap
FROM server AS swap
ARG LS_REPO=mostlygeek/llama-swap
ARG LS_VER=189
RUN curl -LO "https://github.com/${LS_REPO}/releases/download/v${LS_VER}/llama-swap_${LS_VER}_linux_amd64.tar.gz" \
    && tar -zxf "llama-swap_${LS_VER}_linux_amd64.tar.gz" \
    && rm "llama-swap_${LS_VER}_linux_amd64.tar.gz"
COPY ./ik_llama-cpu-swap.config.yaml /app/config.yaml
HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080"]
ENTRYPOINT [ "/app/llama-swap", "-config", "/app/config.yaml" ]
