# Qwen3Next Performance-Differences Report (`ik_llama.cpp` vs `llama.cpp`)

## Scope

This report documents:

- Measured behavior observed during bring-up and benchmarking.
- Code-level differences likely affecting performance.
- Fixes already applied in `ik_llama.cpp`.
- Remaining bottlenecks and concrete next steps.

All numbers below were collected on this machine in Docker with the model:

- `/models/qwen3-next-coder.gguf`

Date of measurements: 2026-02-06.

## Environment Notes

- GPU setup: RTX 5060 Ti + RTX 3060.
- Early slow runs were partially confounded by low free memory on GPU1 in one session (`~201 MiB` free at init).
- Later checks confirmed GPUs can be mostly free (`~15.8 GiB` and `~11.9 GiB` free) before starting runs.

## What Was Validated

### Numerical sanity/parity check (perplexity)

Using identical prompt text, `c=256`, `b=64`, `ub=64`, CPU model weights (`-ngl 0`), no warmup:

- `ik` (`llama-perplexity`) `chunks=1`:
  - `[1]1.0009`
  - `Final estimate: PPL over 1 chunks for n_ctx=256 = 1.0009 +/- 0.00045`
- mainline (`llama-perplexity`) `chunks=1`:
  - `[1]1.0008`
  - `Final estimate: PPL = 1.0008 +/- 0.00036`

And for `chunks=2`:

- `ik`: `[1]1.0009,[2]1.0009`, `Final estimate ... = 1.0009 +/- 0.00026`
- mainline: `[1]1.0008,[2]1.0008`, `Final estimate ... = 1.0008 +/- 0.00020`

Interpretation: current `ik` Qwen3Next path is numerically very close to mainline for this test.

## Measured Performance Signals

### `ik` sweep at long context

`llama-sweep-bench` with `c=65536`, `b=1024`, `ub=128` started successfully and produced low TG values in observed rows (roughly `~2.2` to `~4.1` t/s) and PP mostly in `~27` to `~60` t/s depending on `n_kv` occupancy.

This run was intentionally stopped by user before completion.

### Scheduler limits hit at larger batch

`ik` with `c=65536`, `b=4096`, `ub=1024` failed with:

- `GGML_ASSERT(i_split < GGML_SCHED_MAX_SPLITS)` in `ggml-backend.cpp`.

This indicates high graph split pressure for this configuration.

## Code-Level Differences Relevant to Performance

## 1) Recurrent-state storage model differs from mainline

Mainline Qwen3Next uses recurrent memory abstractions (`llama_memory_recurrent`) with `R` and `S` state buffers in F32:

- `llama.cpp/src/llama-model.cpp:7505`
- `llama.cpp/src/models/qwen3next.cpp:686`
- `llama.cpp/src/models/qwen3next.cpp:687`

`ik` path originally used KV cache-tail handling; this was adjusted to dedicated per-layer state tensors (`s_l`) in F32:

- `ik_llama.cpp/src/llama-context.h:59`
- `ik_llama.cpp/src/llama.cpp:771`
- `ik_llama.cpp/src/llama.cpp:817`
- `ik_llama.cpp/src/llama-build-context.cpp:4617`

Impact: avoids repeated cast in/out of recurrent state for Qwen3Next and aligns closer to mainline state precision behavior.

## 2) `ggml_sub` broadcast semantics differ

Mainline allows repeat/broadcast in `ggml_sub`:

- `llama.cpp/ggml/src/ggml.c:2129`

`ik` currently enforces same-shape inputs:

- `ik_llama.cpp/ggml/src/ggml.c:6406`

Consequence: in Qwen3Next chunking, `ik` must materialize explicit repeats for tensors used in `sub`, increasing graph materialization overhead.

## 3) Qwen3Next chunking path has extra explicit repeats in `ik`

Current `ik` chunking path repeats `g_cumsum` and `g_last` before subtraction:

- `ik_llama.cpp/src/llama-build-context.cpp:4234`
- `ik_llama.cpp/src/llama-build-context.cpp:4287`

Mainline path uses broadcasted subtraction without those explicit materializations:

- `llama.cpp/src/models/qwen3next.cpp:200`
- `llama.cpp/src/models/qwen3next.cpp:264`

Consequence: additional memory traffic and nodes in high-frequency path.

## 4) Graph split count is higher in `ik` for tested Qwen3Next context

Observed logs for `c=256` showed:

- `ik`: graph splits `1227`
- mainline: graph splits `975`

Higher split count usually implies more sync/copy overhead and can reduce PP/TG.

## Fixes Already Applied in `ik`

These are included in commit:

- `a7df116` (`qwen3next: add architecture support and recurrent-state fixes`)

Applied items:

- Added Qwen3Next architecture and kernels in `ik`.
- Added dedicated F32 recurrent-state storage (`s_l`) for Qwen3Next recurrent layers.
- Updated Qwen3Next build path to read/write from dedicated state storage when available.
- Ensured numerical sanity vs mainline with perplexity checks above.
- Kept conservative explicit-repeat logic in chunking where `ik` `ggml_sub` currently requires same-shape (after testing showed global broadcast change caused instability in this fork).

## Why Current `ik` Can Still Be Slower

Most probable remaining reasons:

- Extra repeat materializations in chunking path.
- Higher graph split count in scheduler/backend path.
- Less optimized Qwen3Next integration path compared to mainline recurrent-memory abstractions.
- Run configuration sensitivity at long context and very large batch (`SCHED_MAX_SPLITS` boundary).

## Priority Next Fixes

1. Reduce split pressure and keep benchmark configs inside stable split envelope at 64k.
2. Eliminate or fuse high-cost repeat materializations in Qwen3Next chunking path without changing math.
3. Align more of Qwen3Next recurrent memory/update flow with mainline memory-recurrent pattern where possible.
4. Validate after each change:
   - PPL/outputs against mainline.
   - PP/TG against the same benchmark parameters.

## Current Status

- Qwen3Next is integrated and functionally running in `ik`.
- Precision is close to mainline on tested perplexity cases.
- Performance gap remains and requires targeted optimization work listed above.

## 2026-02-06 Optimization Update

### Newly applied performance changes

1. Enabled broadcast-capable `ggml_sub` and aligned it with existing `ggml_mul` broadcast behavior.
2. Reworked CPU `ggml_compute_forward_sub_f32` to use threaded row-splitting and contiguous broadcast loops.
3. Enabled `GGML_OP_SUB` multi-task scheduling in `ggml_get_n_tasks`.
4. Removed two avoidable repeat materializations in Qwen3Next chunking path:
   - `gcs_i = repeat(g_cumsum, ...)` -> `gcs_i = g_cumsum`
   - `g_last_repeat` in `g_diff` path removed, using direct broadcasted subtract.
5. Added a CUDA fast path in `ggml_cuda_op_ssm_conv` for single-sequence recurrent updates (`n_kv == 1`), with token-block parallelization and explicit final-state reconstruction.

### Post-change validation

#### CPU parity vs mainline (`-ngl 0`)

`c=256`, `b=64`, `ub=64`, `--no-warmup`:

- `chunks=1`
  - `ik`: `[1]1.0007`, final `1.0007 +/- 0.00042`
  - mainline: `[1]1.0007`, final `1.0007 +/- 0.00049`
- `chunks=2`
  - `ik`: `[1]1.0007,[2]1.0007`, final `1.0007 +/- 0.00023`
  - mainline: `[1]1.0007,[2]1.0008`, final `1.0008 +/- 0.00028`

#### CUDA sanity parity vs mainline (`CUDA_VISIBLE_DEVICES=1`, `-ngl 1`)

`c=256`, `b=64`, `ub=64`, `--no-warmup`, `chunks=1`:

- `ik`: `[1]1.0011`, final `1.0011 +/- 0.00071`
- mainline: `[1]1.0011`, final `1.0011 +/- 0.00074`

Interpretation: precision parity remains intact after CPU and CUDA optimizations.

### Updated long-context speed signal (`ik`, no KV quantization)

Config: `llama-sweep-bench -c 65536 -b 1024 -ub 128 -ctk f16 -ctv f16`

Observed rows after the changes show:

- PP generally in `~82` to `~91` t/s range once `n_kv` grows (`~768` to `~3328` in sampled rows).
- TG generally in `~6.2` to `~6.6` t/s range in the same sampled region.

This is substantially improved versus earlier observed TG (`~2` to `~4` t/s) in the prior slow run.

### Remaining performance risks

- Some runs still offload few/no layers depending on available VRAM at run time, which can mask CUDA-path gains.
- `SCHED_MAX_SPLITS` limits at very aggressive `(b, ub)` settings are still a separate scaling constraint.
- Additional backend-level profiling is still needed to determine whether remaining gap to top-end mainline numbers is dominated by offload limits, scheduler split overhead, or other kernels.

## 2026-02-06 CUDA MoE/SSM Optimization Update

### Applied changes in this update

1. MoE row mapping in CUDA `mul_mat_id` paths (`ggml/src/ggml-cuda.cu`):
   - Replaced per-call `ids` device->host copy, host-side count/build, and mapping host->device copy.
   - Added device-side count + exclusive prefix sum + scatter kernels:
     - `k_moe_row_count`
     - `k_moe_row_exclusive_scan`
     - `k_moe_row_scatter`
   - Kept existing call-site logic intact by copying only compact metadata back (`moe_counts`, `cum_moe_counts`, invalid-id flag).
   - Net effect: removes large host round-trip traffic from a hot MoE routing path.

2. Qwen3Next SSM conv path for `n_kv > 1` (`ggml/src/ggml-cuda/ssm-conv.cu`):
   - Added a guarded fast path for decode-like multi-sequence batches where each token maps to one unique sequence (no multi-sequence fan-out per token).
   - Added:
     - `ssm_conv_validate_unique_seq_map`
     - `ssm_conv_multi_seq_unique_f32_kernel`
     - `ssm_conv_multi_seq_unique_f32_kernel_nc4`
   - If the input pattern does not satisfy fast-path constraints, execution falls back to the existing kernel path unchanged.

3. Top-k MoE fusion verification:
   - No matcher change was required in this update.
   - Qwen3Next MoE build path still emits the expected `SOFT_MAX -> ... -> ARGSORT -> VIEW -> GET_ROWS` form used by current CUDA fusion checks.

### Parity validation (required checks)

Tests were run in Docker (`iktest-dev:latest`) with:
- model: `/models/qwen3-next-coder.gguf`
- text corpus: `/tmp/qnext_ppl.txt` (same file for `ik` and mainline)
- params: `-c 256 -b 64 -ub 64 --no-warmup`

CPU parity (`-ngl 0`, threshold `<= 5e-4`):
- `chunks=1`: `ik 1.0041` vs `mainline 1.0037` (`delta=4e-4`) -> PASS
- `chunks=2`: `ik 1.0025` vs `mainline 1.0023` (`delta=2e-4`) -> PASS

CUDA sanity parity (`-ngl 1`, threshold `<= 1e-3`):
- `chunks=1`: `ik 1.0041` vs `mainline 1.0037` (`delta=4e-4`) -> PASS
- `chunks=2`: `ik 1.0025` vs `mainline 1.0023` (`delta=2e-4`) -> PASS

### Quick performance matrix (`llama-sweep-bench`)

Config: `-c 512 -b 1024 -ub 128 -n 16 -ctk f16 -ctv f16 -ngl 999 --cpu-moe`

| Profile | Baseline maxPP | Baseline maxTG | New maxPP | New maxTG | Delta maxPP | Delta maxTG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16GB a) `CUDA_VISIBLE_DEVICES=0` | 129.83 | 26.45 | 122.91 | 26.79 | -6.92 | +0.34 |
| 16GB b) `CUDA_VISIBLE_DEVICES=0 -no-ooae` | n/a | n/a | 132.02 | 26.84 | n/a | n/a |
| 28GB a) `CUDA_VISIBLE_DEVICES=0,1 --tensor-split 0.85,0.15` | 127.66 | 22.95 | 127.48 | 23.97 | -0.18 | +1.02 |
| 28GB b) `CUDA_VISIBLE_DEVICES=0,1` | n/a | n/a | 104.61 | 21.17 | n/a | n/a |

### Command log (exact forms)

Build:

```bash
docker run --rm --gpus all \
  -v /home/yurko/Code/ik_llama.cpp:/ik_llama.cpp \
  iktest-dev:latest \
  bash -lc 'cmake --build /ik_llama.cpp/build-cuda13-fresh --config Release -j 56 --target llama-perplexity llama-bench'
```

Parity (`ik`):

```bash
docker run --rm --gpus all \
  -v /home/yurko/Code/ik_llama.cpp:/ik_llama.cpp \
  -v /home/yurko/.cache/llama.cpp:/models \
  -v /tmp:/tmp \
  iktest-dev:latest \
  bash -lc 'export LD_LIBRARY_PATH=/ik_llama.cpp/build-cuda13-fresh/src:/ik_llama.cpp/build-cuda13-fresh/ggml/src:$LD_LIBRARY_PATH; \
  /ik_llama.cpp/build-cuda13-fresh/bin/llama-perplexity -m /models/qwen3-next-coder.gguf -f /tmp/qnext_ppl.txt -c 256 -b 64 -ub 64 --no-warmup --chunks {1|2} -ngl {0|1} -ctk f16 -ctv f16'
```

Parity (mainline):

```bash
docker run --rm --gpus all \
  -v /home/yurko/Code/llama.cpp:/llama.cpp \
  -v /home/yurko/.cache/llama.cpp:/models \
  -v /tmp:/tmp \
  iktest-dev:latest \
  bash -lc 'export LD_LIBRARY_PATH=/llama.cpp/build/src:/llama.cpp/build/ggml/src:$LD_LIBRARY_PATH; \
  /llama.cpp/build/bin/llama-perplexity -m /models/qwen3-next-coder.gguf -f /tmp/qnext_ppl.txt -c 256 -b 64 -ub 64 --no-warmup --chunks {1|2} -ngl {0|1} -ctk f16 -ctv f16'
```

Quick matrix:

```bash
# 16GB a
CUDA_VISIBLE_DEVICES=0 /ik_llama.cpp/build-cuda13-fresh/bin/llama-sweep-bench \
  -m /models/qwen3-next-coder.gguf -c 512 -b 1024 -ub 128 -n 16 -ctk f16 -ctv f16 -ngl 999 --cpu-moe

# 16GB b
CUDA_VISIBLE_DEVICES=0 /ik_llama.cpp/build-cuda13-fresh/bin/llama-sweep-bench \
  -m /models/qwen3-next-coder.gguf -c 512 -b 1024 -ub 128 -n 16 -ctk f16 -ctv f16 -ngl 999 --cpu-moe -no-ooae

# 28GB a
CUDA_VISIBLE_DEVICES=0,1 /ik_llama.cpp/build-cuda13-fresh/bin/llama-sweep-bench \
  -m /models/qwen3-next-coder.gguf -c 512 -b 1024 -ub 128 -n 16 -ctk f16 -ctv f16 -ngl 999 --cpu-moe --tensor-split 0.85,0.15

# 28GB b
CUDA_VISIBLE_DEVICES=0,1 /ik_llama.cpp/build-cuda13-fresh/bin/llama-sweep-bench \
  -m /models/qwen3-next-coder.gguf -c 512 -b 1024 -ub 128 -n 16 -ctk f16 -ctv f16 -ngl 999 --cpu-moe
```

### Status after this update

- Precision parity: PASS on all required checks.
- Performance:
  - 16GB profile improved TG but not PP vs baseline.
  - 28GB split profile improved TG and preserved PP.
- Remaining likely bottlenecks for 16GB PP:
  - MoE routing still limited by per-expert launches/host-side per-expert loop in `mul_mat_id`.
  - Scheduler split / backend-crossing overhead remains visible at this config.

## 2026-02-06 Follow-up Hotspot Pass (this session)

### Additional code changes

1. `ggml/src/ggml-cuda.cu`
   - Removed an unused `ids` device->host copy + stream sync in `ggml_cuda_moe_up_gate_unary` fallback path.
   - Reduced row-mapping host transfer volume by deriving `moe_counts` from host-side prefix bounds (`cum_moe_counts`) instead of copying both arrays from device.
   - Added `build_active_experts(...)` and switched per-expert loops to iterate only active experts.
2. `ggml/src/ggml-cuda/ssm-conv.cu`
   - Removed host-side `cudaMemcpyAsync(...D2H...) + cudaStreamSynchronize` for multi-seq fast-path eligibility.
   - Made fast/fallback dispatch fully async by gating both kernels with a device-side `fast_path_ok` flag.
3. `ggml/src/ggml-backend.cpp`
   - Reduced unnecessary split churn when a weight tensor is on another backend but the current backend can consume that buffer type directly.
   - Increased `GGML_SCHED_MAX_SPLITS` from `2048` to `4096` for large-graph headroom.
4. `src/llama.cpp`
   - Added a Qwen3Next-specific default split guard for heterogeneous dual-GPU layer mode: clamp to at least `75/25` on 2-GPU auto-split when GPU0 has more free memory.
5. `scripts/qwen3next-eval.sh`
   - Fixed CLI compatibility (`mainline: llama-completion`, `ik: llama-cli` completion path).
   - Made evaluation resilient to missing binaries (`gpu_sweep_mainline` is skipped if unavailable).
   - Fixed complexity-token regex.
   - Switched PPL corpus generation to a stable deterministic pattern to reduce chunk-level variance.

### Validation rerun

Run artifact: `/tmp/qwen3next-eval/20260206_064339`

- CPU PPL parity:
  - chunks=1: mainline `1.0009`, ik `1.0009`, delta `0.000000`
  - chunks=2: mainline `1.0005`, ik `1.0005`, delta `0.000000`
- CUDA sanity parity:
  - `gpu_ppl_chunks1_mainline`: `OK`
  - `gpu_ppl_chunks1_ik`: `OK`
- Generation smoke:
  - both mainline and ik contain Fibonacci token(s)
  - mainline contains complexity token(s), ik did not in this sample output
- Notes:
  - `gpu_sweep_mainline` skipped in this environment because `/home/yurko/Code/llama.cpp/build/bin/llama-sweep-bench` is not present.
  - `gpu_sweep_ik` (`c=2048`, `n=32`) in this run peaked at approximately `maxPP=137.02`, `maxTG=24.81`.

### Quick matrix (exact required configs)

Run artifact: `/tmp/qwen3next-matrix/20260206_063957`

| Profile | Baseline maxPP | Baseline maxTG | New maxPP | New maxTG | Delta maxPP | Delta maxTG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16GB a) `CUDA_VISIBLE_DEVICES=0 --cpu-moe` | 129.83 | 26.45 | 115.56 | 25.74 | -14.27 | -0.71 |
| 16GB b) `CUDA_VISIBLE_DEVICES=0 --cpu-moe -no-ooae` | n/a | n/a | 136.21 | 26.00 | n/a | n/a |
| 28GB a) `CUDA_VISIBLE_DEVICES=0,1 --cpu-moe --tensor-split 0.85,0.15` | 127.66 | 22.95 | 129.70 | 22.72 | +2.04 | -0.23 |
| 28GB b) `CUDA_VISIBLE_DEVICES=0,1 --cpu-moe` | n/a | n/a | 117.54 | 22.99 | n/a | n/a |

### Variance note for single-GPU default (`--cpu-moe`)

Repeated measurements show substantial run-to-run variance in this environment:

Run artifact: `/tmp/qwen3next-repeat-20260206_064133`

- `single_cpu_moe` maxPP/maxTG:
  - run1: `113.84 / 25.86`
  - run2: `135.29 / 26.88`
  - run3: `113.95 / 23.54`
- `single_cpu_moe_no_ooae` maxPP/maxTG:
  - run1: `135.33 / 26.49`
  - run2: `133.64 / 24.92`
  - run3: `126.33 / 23.42`

Interpretation: in this setup, `-no-ooae` is currently more stable and generally faster for PP; default OOAE shows large variance and occasional severe PP drops.
