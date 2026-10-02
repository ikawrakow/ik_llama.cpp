# CUDA Bit-Exact Quantization for GGUF

Follow-up to PR 2572 (Joel's IQ4_KT/IQ3_KT encoder). Adds bit-exact CUDA
encoders for 12 GGUF block quants, merged into Joel's single entry
`ggml_cuda_quantize(device, type, ..., nslice, imatrix)` in
`ggml/src/ggml-cuda/kt-encoder.cu`, which stays the lead plumbing.
Order after KT: Q8_0, Q6_0, Q5_0, Q4_0, Q5_1, Q4_1, IQ4_NL, IQ4_XS.
Q5_0/Q4_0 sections are bannered for one-shot removal.

## 1. Bit-exactness model

A quantized block is byte-identical across backends iff every stored byte
depends only on exact order-independent reductions (max/min, argmax with
lowest-index tie-break), per-element rounding with fixed semantics
(truncate, round-half-away `roundf`, round-half-even `nearest_int` magic),
nearest-even FP16, and correctly-rounded division (`__fdiv_rn`, immune to
`-use_fast_math`). No FMA contraction and no cross-element FP accumulation
except where replayed verbatim: row `sigma2` sums are host-computed in exact
CPU order, and the `make_qx`/`make_qkx3`/codebook optimizers run
single-threaded in exact CPU order with RN intrinsics (float and double).

## 2. Per-type algorithms (all match `ggml_quantize_chunk` CPU paths)

| Type | Plain (no imatrix) | With imatrix | Fudge |
|------|--------------------|--------------|-------|
| Q8_0 | max/\|x\|, `d=amax/127`, `roundf(x*id)` | ignored (CPU ignores it; CUDA routes to plain) | Q6_0 quirk (`ggml-quants.c:915`) |
| Q6_0 | OLS `make_qx` (w=x*x) — no plain path on CPU | OLS `make_qx` (w=qw*sqrt) | yes |
| Q5_0 | argmax signed-max, `d=max/-16`, `MIN(31,(int8_t)(x*id+16.5))`, qh bitmap | OLS `make_qx` (nmax=16) | yes |
| Q4_0 | argmax signed-max, `d=max/-8`, `MIN(15,(int8_t)(x*id+8.5))` | OLS `make_qx` (nmax=8) | yes |
| Q5_1 | min/max, `d=(max-min)/31`, `(uint8_t)((x-min)*id+0.5)` no clamp, qh bitmap | `make_qkx3` (nmax=31, doubles) | no |
| Q4_1 | min/max, `d=(max-min)/15`, `MIN(15,…)` (ignores symmetric like CPU) | `make_qkx3` (nmax=15, doubles) | no |
| IQ4_NL | codebook LUT grid (ntry=7) + hill-climb, w=x*x | same with w=qw*sqrt | 1.0 |
| IQ4_XS | 8x block optimizers + global `d=-max/32` fit + re-quant, w=x*x | same with w=qw*sqrt | 1.0 |

Scales store `FP16(fudge*d)` (bit-twiddle conversion, NaN-payload-exact).
Degenerate overflow blocks (|x| > ~1e19, x*x overflows f32) store non-finite
scales whose payload bytes are platform UB (x86 vs CUDA integer casts); the
harness skips whole blocks whose `d` is non-finite on both sides. Real
tensors never hit this (proven by e2e SHAs).

## 3. Kernel design (`ggml/src/ggml-cuda/quantize_gguf.cu`)

- Plain: warp per 32-block, shuffle reductions, lane 0 stores scales.
- OLS/imatrix/codebook: one thread per block (superblock for XS), shared
  helpers (`iq4nl_opt_block_device`, pack helpers, warp argmax/minmax/nibble
  helpers); `make_qkx3` replay uses double RN intrinsics.
- Shared chunked drivers (~128 MiB F32/chunk, per-chunk base indexing,
  every CUDA call checked, 0 = CPU fallback); double-buffered
  H2D/compute/D2H overlap via streams+events; `-ftz=false` like kt-encoder.
- Dispatcher (lookup table in kt-encoder.cu): per-`nslice` fan-out with
  per-slice imatrix offset (`imatrix + s*n_per_row`); HIP/MUSA and bad
  device index return 0 with warn-once CPU fallback.

## 4. Integration

`llama_model_quantize_params` gains `cuda_quantize` (default false) and
`cuda_device` (default 0); CLI `--cuda-quantize` plus `--cuda-device N`
(alias `--device N`). `do_quantize()` calls the single entry (symmetric
Q4_0 stays on CPU); nonzero output is validated (`ggml_validate_row_data`),
zero falls back to `ggml_quantize_chunk`. Non-CUDA builds warn and ignore.

## 5. Validation (`examples/unit_test_cuda`)

16 specs (plain+imatrix per type) compare GPU vs `ggml_quantize_chunk` vs
local refs on identical input: random-uniform, weight-like, edge-cases
(zeros, outlier, ±max, .5 ties, denormals, huge/tiny, mixed signs),
q5/q6-boundary fills, >1M-block chunk loop, `ne[2]` slice reproduction.
Diffs are pair-labeled (`gpu/cpu`, `cpu/ref`). Result: ALL PASS.

End-to-end: `llama-quantize --pure` twice per type (CPU vs `--cuda-quantize`,
plain and imatrix) on Llama-3.2-1B BF16 and Qwen2.5-0.5B F16 — 64/64
SHA256-identical. Spot PPL sane (e.g. Qwen q6_0 9.67 vs F16 9.51;
imatrix helps up to -0.6 on Q4_1).

## 6. Follow-ups

- Q6_1, K-quants, IQ1/IQ2: CPU-only (scan-order-dependent searches).
- Host-level read/write pipelining in llama-quantize untouched (shared code).
- Q4_0/Q5_0 (and Q5_1/Q4_1) removable via bannered sections if requested.
