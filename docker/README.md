# Build and use ik_llama.cpp with CPU or CPU+CUDA

Built on top of [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and [llama-swap](https://github.com/mostlygeek/llama-swap)

Commands are provided for Podman and Docker.

CPU or CUDA sections under [Prebuilt](#Prebuilt)/[Build](#Build) and [Run]($Run) are enough to get up and running.

## Overview

- [Prebuilt](#Prebuilt)
- [Build](#Build)
- [Run](#Run)
- [Troubleshooting](#Troubleshooting)
- [Extra Features](#Extra)
- [Credits](#Credits)

## Prebuilt Docker images

Pull one of the available images from `ghcr.io`. [View all tags](https://github.com/ikawrakow/ik_llama.cpp/pkgs/container/ik-llama-cpp/versions?filters%5Bversion_type%5D=tagged)

```bash
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cpu-swap
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cpu-server
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cpu-full

docker pull ghcr.io/ikawrakow/ik-llama-cpp:cu12-swap
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cu12-server
docker pull ghcr.io/ikawrakow/ik-llama-cpp:cu12-full
```

## Build

The project uses Docker Bake for building multiple targets efficiently.

Clone the repository: `git clone https://github.com/ikawrakow/ik_llama.cpp`

Use `docker-bake`.

```bash
docker buildx create --name ik-llama-builder --use
```

### CPU Variant

```bash
VARIANT=cpu docker buildx bake --builder ik-llama-builder --load full swap
```

Or with custom tags:

```bash
REPO_OWNER=yourname VARIANT=cpu docker buildx bake --builder ik-llama-builder --load \
  -f ./docker-bake.hcl \
  full swap
```

### CUDA Variant

First, set the CUDA version and GPU architecture in `ik_llama-cuda.Containerfile`:
- `CUDA_DOCKER_ARCH`: Your GPU's compute capability (e.g., `86` for RTX 30*, `89` for RTX 40*, `12.0` for RTX 50*)
- `CUDA_VERSION`: CUDA Toolkit version (e.g., `12.6.2`, `13.1.1`)

```bash
VARIANT=cu12 docker buildx bake --builder ik-llama-builder --load full swap
```

### Build Targets

Builds two image tags per variant:

- **`full`**: Includes `llama-server`, `llama-quantize`, and other utilities.
- **`swap`**: Includes only `llama-swap` and `llama-server`.

## Run

- Download `.gguf` model files to your favorite directory (e.g., `/my_local_files/gguf`).
- Map it to `/models` inside the container.
- Open browser `http://localhost:9292` and enjoy the features.
- API endpoints are available at `http://localhost:9292/v1` for use in other applications.

### CPU

```bash
podman run -it --name ik_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro localhost/ik_llama-cpu:swap
```

```bash
docker run -it --name ik_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro localhost/ik_llama-cpu:swap
```

### CUDA

- Install Nvidia Drivers and CUDA on the host.
- For Docker, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- For Podman, install [CDI Container Device Interface](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
- Identify your GPU:
  - [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus) (e.g., `8.6` for RTX30*, `8.9` for RTX40*, `12.0` for RTX50*)
  - [CUDA Toolkit supported version](https://developer.nvidia.com/cuda-toolkit-archive)

```bash
podman run -it --name ik_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro --device nvidia.com/gpu=all --security-opt=label=disable localhost/ik_llama-cuda:swap
```

```bash
docker run -it --name ik_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro --runtime nvidia localhost/ik_llama-cuda:swap
```

## Troubleshooting

- If CUDA is not available, use `ik_llama-cpu` instead.
- If models are not found, ensure you mount the correct directory: `-v /my_local_files/gguf:/models:ro`
- If you need to install `podman` or `docker` follow the [Podman Installation](https://podman.io/docs/installation) or [Install Docker Engine](https://docs.docker.com/engine/install) for your OS.

## Extra

- **Custom commit**: Build a specific `ik_llama.cpp` commit by modifying the Containerfile or using build args.

```bash
docker buildx bake --builder ik-llama-builder --set full.args.BUILD_COMMIT=1ec12b8 full
```

- **Using the tools in the `full` image**:

```bash
$ podman run -it --name ik_llama_full --rm -v /my_local_files/gguf:/models:ro --entrypoint bash localhost/ik_llama-cpu:full
# ./llama-quantize ...
# python3 gguf-py/scripts/gguf_dump.py ...
# ./llama-perplexity ...
# ./llama-sweep-bench ...
```

```bash
docker run -it --name ik_llama_full --rm -v /my_local_files/gguf:/models:ro --runtime nvidia --entrypoint bash localhost/ik_llama-cuda:full
# ./llama-quantize ...
# python3 gguf-py/scripts/gguf_dump.py ...
# ./llama-perplexity ...
# ./llama-sweep-bench ...
```

- **Customize `llama-swap` config**: Save the `./docker/ik_llama-cpu-swap.config.yaml` or `./docker/ik_llama-cuda-swap.config.yaml` locally (e.g., under `/my_local_files/`) then map it to `/app/config.yaml` inside the container appending `-v /my_local_files/ik_llama-cpu-swap.config.yaml:/app/config.yaml:ro` to your `podman run ...` or `docker run ...`.

- **Run in background**: Replace `-it` with `-d`: `podman run -d ...` or `docker run -d ...`. To stop it: `podman stop ik_llama` or `docker stop ik_llama`.

- **GGML_NATIVE**: If you build the image on a different machine, change `-DGGML_NATIVE=ON` to `-DGGML_NATIVE=OFF` in the `.Containerfile`.

- **KV quantization types**: To use more KV quantization types, build with `-DGGML_IQK_FA_ALL_QUANTS=ON`.

- **Cleanup unused CUDA images**: If you experiment with several `CUDA_VERSION`, delete unused images (they are several GB):
  ```bash
  podman image rm docker.io/nvidia/cuda:12.4.0-runtime-ubuntu22.04 && \
    podman image rm docker.io/nvidia/cuda:12.4.0-devel-ubuntu22.04
  ```

- **Build without `llama-swap`**: Change `--target swap` to `--target server` in docker-bake or Containerfiles.

- **Pre-made quants**: Look for premade quants from [ubergarm](https://huggingface.co/ubergarm/models).

- **GGUF tools**: Build custom quants with [Thireus](https://github.com/Thireus/GGUF-Tool-Suite)'s tools.

- **Download prebuilt binaries**: Download from [ik_llama.cpp's Thireus fork with release builds for macOS/Windows/Ubuntu CPU and Windows CUDA](https://github.com/Thireus/ik_llama.cpp).

- **KoboldCPP experience**: [Croco.Cpp is a fork of KoboldCPP inferring GGUF/GGML models on CPU/Cuda with KoboldAI's UI. It's powered partly by IK_LLama.cpp, and compatible with most of Ikawrakow's quants except Bitnet.](https://github.com/Nexesenex/croco.cpp)

## Credits

All credits to the awesome community:

[llama-swap](https://github.com/mostlygeek/llama-swap)
