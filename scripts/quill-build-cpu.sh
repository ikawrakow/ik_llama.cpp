#!/usr/bin/env bash
# quillama.cpp: CPU-only release build tuned for the machine it runs on.
# Build on the server that will run the model: GGML_NATIVE targets the local CPU (AVX-512, VNNI, ...).
# The Intel AMX kernels are compiled in any x86-64 Linux build and switched on at run time.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build}
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF "$@"
cmake --build "$BUILD_DIR" --config Release -j "$(nproc)" --target llama-server llama-cli llama-bench llama-quantize llama-perplexity llama-sweep-bench
echo
echo "built: $BUILD_DIR/bin/llama-server (see docs/quillama.md)"
if grep -q amx_bf16 /proc/cpuinfo 2>/dev/null; then
    echo "this CPU has AMX: prompt processing uses the AMX kernels"
else
    echo "no AMX on this CPU: prompt processing uses ik_llama.cpp's AVX2/AVX-512 kernels"
fi
