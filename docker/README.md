# Build and use ik_llama.cpp with CPU or CPU+CUDA

Built on top of [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and [llama-swap](https://github.com/mostlygeek/llama-swap)

All commands are provided for Podman and Docker.

CPU or CUDA sections under [Build](#Build) and [Run]($Run) are enough to get up and running. 

## Overview

- [Build](#Build)
- [Run](#Run)
- [Troubleshooting](#Troubleshooting)
- [Extra Features](#Extra)
- [Credits](#Credits)

# Build

Builds two image tags:

- `swap`: Includes only `llama-swap` and `llama-server`.
- `full`: Includes `llama-server`, `llama-quantize`, and other utilities.

Start: download the 4 files to a new directory (e.g. `~/ik_llama/`) then follow the next steps.

```
└── ik_llama
    ├── ik_llama-cpu.Containerfile
    ├── ik_llama-cpu-swap.config.yaml
    ├── ik_llama-cuda.Containerfile
    └── ik_llama-cuda-swap.config.yaml
```

## CPU

```
podman image build --format Dockerfile --file ik_llama-cpu.Containerfile --target full --tag ik_llama-cpu:full && podman image build --format Dockerfile --file ik_llama-cpu.Containerfile --target swap --tag ik_llama-cpu:swap
```

```
docker image build --file ik_llama-cpu.Containerfile --target full --tag ik_llama-cpu:full . && docker image build --file ik_llama-cpu.Containerfile --target swap --tag ik_llama-cpu:swap .
```

## CUDA

```
podman image build --format Dockerfile --file ik_llama-cuda.Containerfile --target full --tag ik_llama-cuda:full && podman image build --format Dockerfile --file ik_llama-cuda.Containerfile --target swap --tag ik_llama-cuda:swap
```

```
docker image build --file ik_llama-cuda.Containerfile --target full --tag ik_llama-cuda:full . && docker image build --file ik_llama-cuda.Containerfile --target swap --tag ik_llama-cuda:swap .
```

# Run

- Download `.gguf` model files to your favorite directory (e.g. `/my_local_files/gguf`).
- Map it to `/models` inside the container.
- Open browser `http://localhost:9292` and enjoy the features.
- API endpoints are available at `http://localhost:9292/v1` for use in other applications.

## CPU

```
podman run -it --name ik_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro localhost/ik_llama-cpu:swap
```

```
docker run -it --name ik_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro ik_llama-cpu:swap
```

## CUDA

- Install Nvidia Drivers and CUDA on the host.
- For Docker, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- For Podman, install [CDI Container Device Interface](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)

```
podman run -it --name ik_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro --device nvidia.com/gpu=all --security-opt=label=disable localhost/ik_llama-cuda:swap
```

```
docker run  -it --name ik_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro --runtime nvidia ik_llama-cuda:swap
```

# Troubleshooting

- If CUDA is not available, use `ik_llama-cpu` instead.
- If models are not found, ensure you mount the correct directory: `-v /my_local_files/gguf:/models:ro`
- If you need to install `podman` or `docker` follow the [Podman Installation](https://podman.io/docs/installation) or [Install Docker Engine](https://docs.docker.com/engine/install) for your OS.

# Extra

- `CUSTOM_COMMIT` can be used to build a specific `ik_llama.cpp` commit (e.g. `1ec12b8`).

```
podman image build --format Dockerfile --file ik_llama-cpu.Containerfile --target full --build-arg CUSTOM_COMMIT="1ec12b8" --tag ik_llama-cpu-1ec12b8:full && podman image build --format Dockerfile --file ik_llama-cpu.Containerfile --target swap --build-arg CUSTOM_COMMIT="1ec12b8" --tag ik_llama-cpu-1ec12b8:swap
```

```
docker image build --file ik_llama-cuda.Containerfile --target full --build-arg CUSTOM_COMMIT="1ec12b8" --tag ik_llama-cuda-1ec12b8:full . && docker image build --file ik_llama-cuda.Containerfile --target swap --build-arg CUSTOM_COMMIT="1ec12b8" --tag ik_llama-cuda-1ec12b8:swap .
```

- Using the tools in the `full` image:

```
$ podman run -it --name ik_llama_full --rm -v /my_local_files/gguf:/models:ro --entrypoint bash localhost/ik_llama-cpu:full
# ./llama-quantize ...
# python3 gguf-py/scripts/gguf_dump.py ...
# ./llama-perplexity ...
# ./llama-sweep-bench ...
```

```
docker run  -it --name ik_llama_full --rm -v /my_local_files/gguf:/models:ro --runtime nvidia --entrypoint bash ik_llama-cuda:full
# ./llama-quantize ...
# python3 gguf-py/scripts/gguf_dump.py ...
# ./llama-perplexity ...
# ./llama-sweep-bench ...
```

- Customize `llama-swap` config: save the `ik_llama-cpu-swap.config.yaml` or `ik_llama-cuda-swap.config.yaml` localy  (e.g. under `/my_local_files/`) then map it to `/app/config.yaml` inside the container appending `-v /my_local_files/ik_llama-cpu-swap.config.yaml:/app/config.yaml:ro` to your`podman run ...` or `docker run ...`.
- To run the container in background, replace `-it` with `-d`: `podman run -d ...` or `docker run -d ...`. To stop it: `podman stop ik_llama` or `docker stop ik_llama`.
- If you build the image on the same machine where will be used, change `-DGGML_NATIVE=OFF` to `-DGGML_NATIVE=ON` in the `.Containerfile`.
- For a smaller CUDA build, identify your GPU [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus) (e.g. `8.6` for RTX30*0) then change `CUDA_DOCKER_ARCH` in `ik_llama-cuda.Containerfile` from `default` to your GPU architecture (e.g. `CUDA_DOCKER_ARCH=86`).
- If you build only for your GPU architecture and want to make use of more KV quantization types, build with `-DGGML_IQK_FA_ALL_QUANTS=ON`.
- Look for premade quants (and imatrix files) that work well on most standard systems and are designed around ik_llama.cpp (with helpful metrics in the model card) from [ubergarm](https://huggingface.co/ubergarm/models).
- Usefull graphs and numbers on @magikRUKKOLA [Perplexity vs Size Graphs for the recent quants (GLM-4.7, Kimi-K2-Thinking, Deepseek-V3.1-Terminus, Deepseek-R1, Qwen3-Coder, Kimi-K2, Chimera etc.)](https://github.com/ikawrakow/ik_llama.cpp/discussions/715) topic.
- Build custom quants with [Thireus](https://github.com/Thireus/GGUF-Tool-Suite)'s tools.
- Download from [ik_llama.cpp's Thireus fork with release builds for macOS/Windows/Ubuntu CPU and Windows CUDA](https://github.com/Thireus/ik_llama.cpp) if you cannot build.
- For a KoboldCPP experience [Croco.Cpp is fork of KoboldCPP infering GGML/GGUF models on CPU/Cuda with KoboldAI's UI. It's powered partly by IK_LLama.cpp, and compatible with most of Ikawrakow's quants except Bitnet. ](https://github.com/Nexesenex/croco.cpp)

# Credits

All credits to the awesome community:  

[llama-swap](https://github.com/mostlygeek/llama-swap)
