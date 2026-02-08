# Qwen3Next Benchmark: PP 16384 / TG 128 (`ik_llama.cpp` vs `llama.cpp`)

Date: 2026-02-08

## Setup

- Container: `iktest2`
- Model: `/models/qwen3-next-coder.gguf`
- Prompt processing: `-p 16384`
- Token generation: `-n 128`
- Batch settings: `-b 3072 -ub 768`
- Threads: `-t 8`
- Repetitions: `-r 1`
- Mmap: `-mmp 0`

CUDA runs:

- `CUDA_VISIBLE_DEVICES=0`
- `-fa 1 -ngl 999 --n-cpu-moe 47`

CPU-only runs:

- `-fa 0 -ngl 0 --n-cpu-moe 0`

Hardware note:

- GPU0 (bench target): `NVIDIA GeForce RTX 5060 Ti`, `16311 MiB` total (`CUDA_VISIBLE_DEVICES=0` for CUDA runs).
- GPU1 (not used for these runs): `NVIDIA GeForce RTX 3060`, `12288 MiB` total.
- Observed during active `ik` CUDA run (`p=8192,b=2048,ub=512,n-cpu-moe=45`): GPU0 memory used `~12074 MiB` (`~3775 MiB` free), from `nvidia-smi`.

## Results

| Build | Backend | PP 16384 (tok/s) | TG 128 (tok/s) |
|---|---|---:|---:|
| `ik_llama.cpp` | CUDA | 207.891304 | 27.263562 |
| `llama.cpp` | CUDA | 185.764649 | 24.145662 |
| `ik_llama.cpp` | CPU-only | 45.739881 | 12.172113 |
| `llama.cpp` | CPU-only | 47.835420 | 6.991398 |

## Relative (`ik` vs `llama.cpp`)

- CUDA PP: `+11.91%`
- CUDA TG: `+12.91%`
- CPU PP: `-4.38%`
- CPU TG: `+74.10%`

## Raw outputs

- `/tmp/ik_cuda_bench_16k.json`
- `/tmp/mainline_cuda_bench_16k.json`
- `/tmp/ik_cpu_bench_16k.json`
- `/tmp/mainline_cpu_bench_16k.json`

## Additional CUDA rerun (requested lower `n-cpu-moe` ballpark)

Adjusted config:

- `-p 8192 -n 128 -b 2048 -ub 512 -t 8 -fa 1 -ngl 999 -mmp 0`
- single GPU: `CUDA_VISIBLE_DEVICES=0`

Fit checks on `ik`:

- `--n-cpu-moe 25` -> fail to load model
- `--n-cpu-moe 40` -> fail to create context
- `--n-cpu-moe 45` -> works

Working comparison at `--n-cpu-moe 45`:

| Build | Backend | PP 8192 (tok/s) | TG 128 (tok/s) |
|---|---|---:|---:|
| `ik_llama.cpp` | CUDA | 201.613283 | 24.884600 |
| `llama.cpp` | CUDA | 145.100895 | 24.595058 |

`ik` rerun with `-rtr 1` at the same config (`--n-cpu-moe 45`):

| Build | Backend | PP 8192 (tok/s) | TG 128 (tok/s) |
|---|---|---:|---:|
| `ik_llama.cpp` (`-rtr 1`) | CUDA | 232.340508 | 27.895722 |

## Historical Fused DeltaNet Check (obsolete)

Date: 2026-02-08

Setup:

- Container: `iktest2`
- Device: `CUDA_VISIBLE_DEVICES=0` (RTX 5060 Ti)
- Common args: `-c 2048 -b 2048 -ub 512 --chunks 1 --no-warmup -ngl 999 --n-cpu-moe 47 -t 8 -fa on`
- Switch under test: `LLAMA_QWEN3NEXT_FUSED_DELTA`

Results (Wikitext2 sample file `/tmp/ppl_wikitext2_test.txt`):

| Model | `LLAMA_QWEN3NEXT_FUSED_DELTA=0` | `LLAMA_QWEN3NEXT_FUSED_DELTA=1` |
|---|---:|---:|
| `/models/qwen3-next-coder.gguf` | `PPL 3.9378` | `PPL 15.3628` |
| `/models/qwen-3-coder-next-mxfp4.gguf` | `PPL 3.9860` | `PPL 15.0740` |

Conclusion:

- This run is kept for history only and is superseded by the later `Fused DeltaNet Safety Update (Superseding)` section below.
- Use the superseding section as source of truth for mode mapping and quality guidance.

## Upstream PR #19375 Trial (Selective Port) Outcome

Date: 2026-02-08

What was tried:

- Ported selected non-fused qwen3next graph changes from `ggml-org/llama.cpp#19375` (broadcast/repeat and autoregressive matmul rewrite), then benchmarked and re-tested perplexity.

Outcome:

- No stable speed win in our setup after repeated runs.
- Direct autoregressive rewrite attempts from PR #19375 were not compatible with current ik graph-layout/contiguity assumptions and were reverted.
- Final code keeps only safe chunk-shape fixes plus fused-mode safety controls.

## Decode-Only Fused Mode Trial (`LLAMA_QWEN3NEXT_FUSED_DELTA=2`)

Date: 2026-02-08

Code change:

- Added mode `2` for `LLAMA_QWEN3NEXT_FUSED_DELTA`:
  - prompt / multi-token path: non-fused
  - single-token decode path: fused

Perplexity validation (`-c 2048`, GPU config as above):

| Model | `=0` non-fused | `=2` decode-only fused |
|---|---:|---:|
| `/models/qwen3-next-coder.gguf` | `3.9378` | `3.9378` |
| `/models/qwen-3-coder-next-mxfp4.gguf` | `3.9860` | `3.9860` |

`llama-bench` at `-p 8192 -n 128 -b 2048 -ub 512 -r 3 -rtr 1`:

| Mode | PP 8192 (tok/s) | TG 128 (tok/s) |
|---|---:|---:|
| `LLAMA_QWEN3NEXT_FUSED_DELTA=0` | `170.090` | `25.465` |
| `LLAMA_QWEN3NEXT_FUSED_DELTA=2` | `166.212` | `29.599` |

Notes:

- Decode-only fused mode preserves prompt-quality metrics in this test.
- TG improved significantly in this run; PP variance was higher, so PP delta should be treated as noisy.

## Fused DeltaNet Safety Update (Superseding)

Date: 2026-02-08

This section supersedes the earlier `LLAMA_QWEN3NEXT_FUSED_DELTA` mode mapping.

Updated env behavior in `src/llama-build-context.cpp`:

- `0` / unset: non-fused for all token counts
- `1`: fused only for `n_tok > 1` (prefill/chunking), non-fused for single-token decode
- `2`: fused for all token counts (experimental)

Reason:

- Fused path has a known decode-path quality regression when forced on single-token steps.
- The safer default acceleration is therefore prefill-only fused mode (`=1`).

Validation (CUDA, `qwen3-next-coder.gguf`, `-c 2048 -b 1 -ub 1 -fa on -ngl 47 --n-cpu-moe 40 --chunks 1 --no-warmup`):

| Mode | PPL |
|---|---:|
| `LLAMA_QWEN3NEXT_FUSED_DELTA=0` | `3.9148 +/- 0.31093` |
| `LLAMA_QWEN3NEXT_FUSED_DELTA=1` | `3.9148 +/- 0.31093` |
| `LLAMA_QWEN3NEXT_FUSED_DELTA=2` | `6.1277 +/- 0.54810` |

Quick throughput check (`-p 8192 -n 128 -b 2048 -ub 512 -r 1 -rtr 1`, same CUDA settings):

| Mode | PP 8192 (tok/s) | TG 128 (tok/s) |
|---|---:|---:|
| `0` | `179.30` | `24.69` |
| `1` | `252.12` | `22.99` |
| `2` | `245.71` | `27.94` |

Interpretation:

- Use `=1` for production-safe quality with strong PP gain.
- Reserve `=2` for experiments only until decode-path correctness is fixed.
