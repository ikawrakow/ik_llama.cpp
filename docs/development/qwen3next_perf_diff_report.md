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
