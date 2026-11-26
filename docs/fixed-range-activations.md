# Fixed-range activation quantization (CPU experiment)

This fork already has per-tensor `input_scale` metadata, used by the FP8/NVFP4
paths.  The CPU activation quantizer now honours it for the Q8 activation
formats used by `mul_mat` / `mul_mat_id`:

- `Q8_K` (`quantize_row_q8_K_fixed`)
- `Q8_0` (`quantize_row_q8_0_fixed`)
- `Q8_0_X4` (`quantize_row_q8_0_x4_fixed`)
- `Q8_2_X4` (`quantize_row_q8_2_x4_fixed`)

When a weight tensor has a positive `input_scale`, the activation row is
quantized with that fixed scale and clamped to `[-127, 127]`, instead of
computing the amax of the row.  This is CPU-only for now.

## 1. Calibrate the FFN down projections

```bash
IK_ACT_CALIB=/tmp/act-calib.txt \
    ./build/bin/llama-cli -m model.gguf -p "calibration text" -n 0
```

Only tensors whose name contains `ffn_down` are observed.  For MoE tensors the
amax is accumulated separately for every expert.  The output is:

```
# tensor expert scale count median p99 p999 max mean
blk.0.ffn_down_exps.weight 3 10.3747168 1000 5.18735838 9.93486214 10.3747168 9.98999977 4.99499989
```

`scale` is the Q8 delta (`amax/127`), using the p99.9 bin edge by default.  Use
`IK_ACT_CALIB_QUANTILE=median|p99|p999|max` to select another statistic.
`median`, `p99`, `p999` and `max` are the raw amax values, so they remain easy
to compare across experts.

The per-expert columns tell you whether one scalar per tensor is enough.  With
Inkling-2-Mini the p99.9 varies by roughly 2-4x between experts of the same
layer, and by up to ~35x between the smallest and largest expert, so the loader
uses one scale **per expert** (indexed by the routing ids) and falls back to
the scalar value only for experts that were not observed.

## 2. Run with the calibrated scales

```bash
IK_ACT_SCALES=/tmp/act-calib.txt \
    ./build/bin/llama-cli -m model.gguf -p "hello" -n 128
```

The loader reads only the first three columns (`tensor expert scale`).  It
installs a `float[n_expert]` table on the expert weight; the quantization loop
picks the entry for the expert selected by `ids`.  Experts missing from the
calibration (scale 0) keep the dynamic amax path.  A scalar `input_scale`
fallback is still set for dense tensors and for any expert outside the table.

**MoE cache caveat.** With `--moe-cache-mib N > 0`, `id` is rewritten through
`slot_map` and the cached weight is slot-indexed.  In that case the per-expert
table cannot be indexed by `id` safely, so cache tensors only receive the
scalar fallback (the maximum over experts), and calibration skips
`moe_cache.*` tensors.  Run the calibration with the MoE cache disabled
(`--moe-cache-mib 0`, the default) if you want per-expert scales.

## 3. What to measure

- PPL with `IK_ACT_SCALES` on/off.
- Prefill and token generation speed.
- The ratio `p999 / max` per expert in the calibration file.  A large ratio
  means a few outliers dominate the global scale.

## 4. First measurements on Inkling-2-Mini

`Inkling-2-Mini-AD-IQ3_XXS`, 4 threads, `n_ctx=512`, held-out 4 chunks of
documentation text (calibration used 16 chunks of a different text):

| activation scales | PPL | delta |
|---|---:|---:|
| dynamic amax (baseline) | 4.0284 | - |
| fixed, p99.9, **per expert** (SIMD Q8_2_X4) | 4.0701 | +1.03% |
| fixed, p99.9, one scalar (max) per layer | 4.1330 | +2.60% |

Per-expert p99.9 varies by about 2-3x between experts of the same layer
(max/min up to ~35x), which is exactly why the scalar version loses more
quality.  With the AVX2 `Q8_2_X4` fixed quantizer (and `Q8_0_X4`), `pp512` is
no longer slower than the dynamic path within measurement noise (25.6/25.0 vs
26.5 t/s on this machine), and `tg128` is also at parity within noise
(~3.79 vs ~3.74 t/s over 3 repetitions).  `Q8_K` still uses the scalar fixed
path; it is the next SIMD candidate for K-quant models.  The remaining win
would come from fusing the quantization into the SiTU GLU output, which avoids
the F32 round-trip.

The clamp is intentional: `SiTU` bounds the GLU output by
`beta * beta_linear` in theory, but the useful range is much smaller.  Start
with p99.9 and compare against `max`.

## 5. Experimental fused up-gate (opt-in)

`IK_FUSE_ENABLE=1` selects `GGML_OP_MOE_UP_GATE_QUANT`: the SiTU GLU output is
quantized into the tensor that `down` consumes, so the down matmul skips its
activation quantization pass.  The implementation uses a shared F32 scratch
and a barrier per expert; it is numerically correct (PPL 4.0284 with dynamic
scales, 4.0701 with per-expert p99.9, identical to the non-fused path) but it
is currently slower on prefill (21-22 vs ~29 t/s) because of the per-expert
barriers.  It is therefore off by default; the next step is a tile-level
quantization inside `mul_mat_up_gate_NxM` that removes the scratch/barriers.
