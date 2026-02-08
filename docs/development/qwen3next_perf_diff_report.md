# Qwen3Next Review and Benchmark Summary (`ik_llama.cpp` vs `llama.cpp`)

Date: 2026-02-08

## Scope

This document captures:

- Current upstream PR alignment for Qwen3Next-related work.
- What is already strong in `ik_llama.cpp` and what still needs adjustment.
- Recommended runtime settings for this machine (single GPU target, long context).
- Final apples-to-apples benchmark matrix for `ik_llama.cpp` vs `../llama.cpp`.

## Upstream PR Check (as of 2026-02-08)

Reviewed PRs:

- https://github.com/ggml-org/llama.cpp/pull/18102 (`open`): Delta-Net CUDA op + integration.
- https://github.com/ggml-org/llama.cpp/pull/18792 (`open`): unified DeltaNet handling (`src/models/delta.cpp`).
- https://github.com/ggml-org/llama.cpp/pull/19375 (`open`, `draft`): Qwen3Next graph optimization in model builder.

### Current alignment in `ik_llama.cpp`

Already present and/or functionally covered:

- CUDA DeltaNet op path exists in GGML (`ggml/src/ggml-cuda/delta-net.cu`).
- Solve-tri and backend op support are present for the fused path.
- Qwen3Next fused DeltaNet builder path exists (and is now runtime-toggleable via env).
- Existing ik optimizations remain available (`-rtr`, grouped/fused paths, no-offload-only-active-experts switches).

Not directly mirrored yet (by design divergence from mainline model layout):

- Mainline `src/models/delta.cpp` structure from PR #18792.
- Mainline `src/models/qwen3next.cpp` graph-form from PR #19375.

## Required Adjustments (remaining)

1. Keep non-fused as the strict safety baseline in defaults, and use `LLAMA_QWEN3NEXT_FUSED_DELTA=1` (prefill-only fused) as the explicit acceleration mode.
2. Add a first-class runtime flag/CLI plumb for Qwen3Next fused mode (`LLAMA_QWEN3NEXT_FUSED_DELTA`) so serving does not depend on raw env wiring.
3. Continue using `scripts/qwen3next-regression.sh` as the release gate for this model path, and wire it into CI or pre-merge checks.
4. Treat the remaining PR #19375 autoregressive rewrite as deferred: direct porting into current ik graph builder is not layout-compatible without broader contiguity/reshape refactoring.
5. Revisit PR #18792 (`src/models/delta.cpp`) only if we need unified GDA/KDA support for additional architectures; for Qwen3Next-only it is optional.

## Strong Points of `ik_llama.cpp` to Preserve

- More runtime controls than mainline for this workload (`-rtr`, backend toggles, MoE/OOAE controls).
- Strong CUDA path for this model family once offload routing is tuned (`--n-cpu-moe` thresholding).
- Better TG throughput than current mainline in matched CUDA and CPU tests on this host.

## Best Runtime Configuration (this host)

Model: `/models/qwen3-next-coder.gguf`

Single-GPU long-context finding:

- `-c 65536` on GPU0 (16 GB) requires at least `--n-cpu-moe 47` to fit reliably.

8k sweep proxy (single GPU, tuned path):

- `b=2048,ub=512` -> `avg_tg ~27.9 tok/s`
- `b=3072,ub=768` -> `avg_tg ~28.4 tok/s` (best TG)
- `b=4096,ub=1024` -> `avg_tg ~26.9 tok/s`

Recommended serving baseline:

- `CUDA_VISIBLE_DEVICES=0`
- `-c 65536 -b 3072 -ub 768 -t 8 -fa on -ngl 999 --n-cpu-moe 47 -rtr`

## Final Benchmark Matrix (8k context proxy)

All four builds were benchmarked with matched parameters and explicit `-mmp 0` for fairness.

Common args:

- `-m /models/qwen3-next-coder.gguf -p 8192 -n 128 -b 3072 -ub 768 -t 8 -r 1`
- CUDA runs: `CUDA_VISIBLE_DEVICES=0 -fa 1 -ngl 999 --n-cpu-moe 47 -mmp 0`
- CPU runs: `-fa 0 -ngl 0 --n-cpu-moe 0 -mmp 0`

| Build | PP (tok/s) | TG (tok/s) |
|---|---:|---:|
| `ik` CUDA | 204.614 | 28.979 |
| mainline CUDA | 184.521 | 22.012 |
| `ik` CPU | 49.795 | 12.681 |
| mainline CPU | 51.674 | 7.299 |

Relative (`ik` vs mainline):

- CUDA PP: `+10.9%`
- CUDA TG: `+31.7%`
- CPU PP: `-3.6%`
- CPU TG: `+73.7%`

## Notes

- CPU-only Qwen3Next with `-fa 1` is now guarded in ik: FA is auto-disabled with a warning for `n_gpu_layers == 0` to avoid the prior `iqk_fa_templates.h` assert path.
- `ik` benchmark JSON currently includes some non-JSON log lines in stdout around context creation; parsing should tolerate that.
- Fused DeltaNet mode mapping has been updated in code:
  - `0` / unset: non-fused
  - `1`: fused only for `n_tok > 1` (safe mode)
  - `2`: fused on all token counts (experimental; decode-quality regression observed)
- Added manual regression runner for fused-mode safety checks:
  - `scripts/qwen3next-fused-regression.sh`
  - Example:
    - `BIN=./build-qwen3next-fix/bin/llama-perplexity scripts/qwen3next-fused-regression.sh --model /models/qwen3-next-coder.gguf --ctx 2048 --decode-b 1 --decode-ub 1 --prefill-b 2048 --prefill-ub 512 --ngl 47 --n-cpu-moe 40`
- Also integrated into the broader eval harness:
  - `scripts/qwen3next-eval.sh --with-gpu --with-fused-regression ...`
  - Results are surfaced in `SUMMARY.md` under `IK Fused Delta Regression`.
- Fused regression now enforces absolute non-fused sanity too:
  - mode0 decode/prefill PPL must stay below configurable thresholds (defaults: `10.0` / `10.0`).
- Added unified Qwen3Next regression entrypoint for ongoing checks:
  - `scripts/qwen3next-regression.sh --model /path/to/qwen3-next-coder.gguf`
  - Outputs `SUMMARY.md` + per-step logs under `/tmp/qwen3next-regression/<timestamp>/`.
