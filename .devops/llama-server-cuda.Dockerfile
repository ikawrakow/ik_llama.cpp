ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=12.4.1
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# Target the CUDA runtime image
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# Set targeted arch here as needed, default: 86 (Ampere) and 90 (Hopper)
ARG CUDA_DOCKER_ARCH="86;90"

RUN apt-get update && \
    apt-get install -y build-essential git libcurl4-openssl-dev ninja-build python3-pip \
 && pip3 install --no-cache-dir cmake \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Set nvcc architecture
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}
# Enable CUDA
ENV GGML_CUDA=1
# Enable cURL
ENV LLAMA_CURL=1
# Must be set to 0.0.0.0 so it can listen to requests from host machine
ENV LLAMA_ARG_HOST=0.0.0.0

RUN cmake -S . -B build -G Ninja \
    -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_C_FLAGS="-fPIC -mcmodel=large" \
    -DCMAKE_CXX_FLAGS="-fPIC -mcmodel=large" \
 && cmake --build build --target llama-server

FROM ${BASE_CUDA_RUN_CONTAINER} AS runtime

RUN apt-get update && \
    apt-get install -y libcurl4-openssl-dev libgomp1 curl

COPY --from=build /app/build/bin/llama-server /llama-server

COPY --from=build /app/build/examples/mtmd/libmtmd.so /usr/local/lib/
COPY --from=build /app/build/ggml/src/libggml.so /usr/local/lib/
COPY --from=build /app/build/src/libllama.so /usr/local/lib/
RUN ldconfig

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/llama-server" ]
