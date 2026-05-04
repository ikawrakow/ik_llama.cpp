#!/bin/sh
# CPU-only build helper for AVX-512-capable CPUs (AMD Zen4 / Intel
# Sapphire Rapids+). Enables the IQK GEMM kernels gated by HAVE_FANCY_SIMD
# (see docs/build.md "CPU build flags for AVX-512").
#
# Usage:
#   ./scripts/build-zen.sh [build-dir]
#
# Default build directory is "build". A subsequent
#
#   objdump -d <build-dir>/bin/llama-cli | grep -c vpdpbusd
#
# should report a non-trivial count if VNNI was compiled in. The runtime
# banner of any built binary will print "HAVE_FANCY_SIMD is defined" when
# the AVX-512 quantized matmul path is active.

set -e

BUILD_DIR=${1:-build}

cmake -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DGGML_AVX512=ON \
    -DGGML_AVX512_VBMI=ON \
    -DGGML_AVX512_VNNI=ON \
    -DGGML_AVX512_BF16=ON

cmake --build "$BUILD_DIR" --config Release -j
