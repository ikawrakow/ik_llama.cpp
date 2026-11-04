# Nemotron-H, Mamba-2 and NVFP4

This file is the reference for the support added in the "Nemotron-H / NVFP4" pull-request series: the
architecture, the Mamba-2 tensor layouts, the NVFP4 format and its coverage, the limits, and how to
run and quantize a model. The PR descriptions stay short and point here.

This fork supports NVIDIA's Nemotron-H family and the NVFP4 quantization format that its
official GGUFs use.

## Nemotron-H / Nemotron-H-MoE

`nemotron_h` (dense) and `nemotron_h_moe` are hybrid models: a mix of Mamba-2 state-space
layers, attention layers **without positional encoding**, and relu² feed-forward layers. In the
MoE variants the experts are routed by a sigmoid router with an optional shared expert, optional
low-rank projections (`moe_latent_size`) and — for the "Puzzle"-style GGUFs such as the
Nemotron 3.5 Bolt 74B-A9B — **a different expert count and expert FFN width per layer**
(240…496 experts in that model). The per-layer values are read from the `expert_used_count` /
`expert_feed_forward_length` arrays and from the router tensor metadata.

Supported artifacts:

- `nemotron_h`: Nemotron-3-Nano-4B and similar.
- `nemotron_h_moe`: NVIDIA's official Nemotron 3.5 Bolt 74B-A9B (`NVFP4` projections + `IQ2_XXS`
  experts) is the reference file.

### Limitations

- **One sequence per context.** The Mamba-2 path keeps a single recurrent state (conv window in
  the k-side cache slot, SSM state in the s slot), so `n_seq_max > 1` is rejected at context
  creation with a message. Use one context per sequence; `llama-server` needs `-np 1`.
- **`-sm graph` / `-sm attn` are not supported** for these archs: the split mode is downgraded to
  `layer` with a warning, because the SSM and norm tensors are not replicated per device.
- A K-shift / context shift is refused for the same reason — the rolling conv window is keyed by
  absolute position.

### Sizing

The recurrent state per Mamba-2 layer is `(d_conv - 1) * (d_inner + 2 * n_group * d_state)` floats
for the conv window plus `d_state * d_inner` for the SSM state; both are counted in the model's
cache-size estimate. On a 12 GB card the Bolt model fits with `--n-cpu-moe 28` and
`--moe-cache-mib 1024` (see `bench_results` in the matching pull request for measurements).

## NVFP4 (type 40)

NVFP4 is a 4-bit float format from NVIDIA's ModelOpt toolchain: 64-value super-blocks, four
**UE4M3** scales (one per 16 values) and E2M1 magnitudes, 36 bytes per block. The layout is
byte-compatible with mainline, so official exports load unchanged.

| backend | decode (n ≤ 8) | prefill (n > 8) | MoE (`mul_mat_id`) |
|---|---|---|---|
| CUDA | MMVQ (`vec_dot_nvfp4_q8_1`) | MMQ, int8-MMA and dp4a | MMVQ-ID ≤ 8 rows, MMQ-ID above |
| CPU | AVX2/AVX `ggml_vec_dot_nvfp4_q8_0` | IQK `mul_mat_nvfp4_q8_2` (`nrc_y` 1..8) | same IQK kernel |
| fallback | dequantize to f16/f32 for ops without a per-type kernel | | |

Not implemented: AVX-512 kernels, the repacked R4/R8/R16 layouts, and per-channel scales (the
`.scale` companion is a single per-tensor value).

### Quantizing

`llama-quantize` accepts `NVFP4`:

```bash
llama-quantize model-f16.gguf model-nvfp4.gguf NVFP4
```

Preparation is round-to-nearest per 16-value group; ModelOpt exports derive those scales from
calibration data instead, so a locally quantized file is not identical to an official one. Rows
whose width is not a multiple of 64 fall back to F16. Requantizing an already quantized model
needs `--allow-requantize` like any other type.
