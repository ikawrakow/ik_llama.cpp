# Nemotron-H and Mamba-2

This file is the reference for the support added in the Nemotron-H pull request: the architecture,
the Mamba-2 tensor layouts, the limits, and how to run a model. The PR description stays short and
points here.

This fork supports NVIDIA's Nemotron-H family (`nemotron_h` and `nemotron_h_moe`) together with the
Mamba-2 state-space ops it needs: reference CPU implementations (`ggml_ssm_conv_m2`,
`ggml_ssm_scan_m2`) and CUDA kernels; the rest of the graph runs the fork's usual IQK/quantized
kernels.

## Nemotron-H / Nemotron-H-MoE

`nemotron_h` (dense) and `nemotron_h_moe` are hybrid models: a mix of Mamba-2 state-space
layers, attention layers **without positional encoding**, and relu² feed-forward layers. In the
MoE variants the experts are routed by a sigmoid router with an optional shared expert, optional
low-rank projections (`moe_latent_size`) and — for the "Puzzle"-style GGUFs such as the
Nemotron 3.5 Bolt 74B-A9B — **a different expert count and expert FFN width per layer**
(240…496 experts in that model). The per-layer values are read from the `expert_used_count` /
`expert_feed_forward_length` arrays and from the router tensor metadata.

Supported artifacts:

- `nemotron_h`: NVIDIA's Nemotron3-Nano-4B (Q4_K_M and similar).
- `nemotron_h_moe`: Nemotron 3.5 Bolt 74B-A9B. The reference file used here is the Unsloth
  `UD-IQ1_S` quant (763 tensors, 18 GB, standard quant types).

NVIDIA's official 24 GB Bolt export ships its 80 SSM projections as **NVFP4**, each with a
`.scale` companion, so it needs NVFP4 support before it can load here; that is not part of this
change. Everything else in that file is already covered, and any normal quant of the same model —
e.g. the Unsloth `UD-IQ1_S` above — runs today.

### Running

```bash
llama-cli -m NVIDIA-Nemotron-3.5-Bolt-74B-A9B-UD-IQ1_S.gguf -ngl 99 \
    --n-cpu-moe 28 --moe-cache-mib 1024 \
    --cuda-params "offload-batch-size=8" -ub 2048 \
    -c 2048 -t 6 -p "The history of computing is"
```

On a 12 GB card with 31 GB of RAM the 74B-A9B fits with 28 expert layers on the CPU and a 1 GiB
device expert cache.

`--cuda-params "offload-batch-size=8"` is the prefill switch, and it is easy to miss: with
`--n-cpu-moe` the host-resident experts are *computed* on the CPU by default, which makes prefill
CPU-bound. This parameter lets the scheduler stream the active expert ranges to the GPU instead.
Measured with this file on a 3060 (`llama-bench`, pp2048, greedy): **145 t/s** without it (ub 512),
**360 t/s** with it at `-ub 512`, **518 t/s** at `-ub 1024` and **682 t/s** at `-ub 2048`; decode is
unchanged (~25 t/s). The parameter's default (32) never fires at these ubatch sizes.

### Limitations

- **One sequence per context.** The Mamba-2 path keeps a single recurrent state (conv window in
  the k-side cache slot, SSM state in the s slot), so `n_seq_max > 1` is rejected at context
  creation with a message. Use one context per sequence; `llama-server` needs `-np 1`.
- **`-sm graph` / `-sm attn` are not supported** for these archs: they are not on the split-mode
  allow-list, so the split mode falls back to `layer` with the generic warning.
- A K-shift / context shift is refused for the same reason — the rolling conv window is keyed by
  absolute position.

### Sizing

The recurrent state per Mamba-2 layer is `(d_conv - 1) * (d_inner + 2 * n_group * d_state)` floats
for the conv window plus `d_state * d_inner` for the SSM state; both are counted in the model's
cache-size estimate.

## Mamba-2 ops

Two tensor layouts exist for `ssm_conv` and `ssm_scan`: the legacy one (8-argument
`ggml_ssm_scan`, packed `s_l` state) and the Mamba-2 layout. The legacy functions are untouched —
`ggml_ssm_scan` keeps setting `K = -1`, so existing models dispatch to the old path — and the
Mamba-2 form gets its own entry points, `ggml_ssm_conv_m2` and `ggml_ssm_scan_m2`.

| backend | `ssm_conv_m2` | `ssm_scan_m2` |
|---|---|---|
| CPU | reference, incl. strided `conv_x` views | reference, chunked prefill == token-by-token |
| CUDA | `ssm-conv-m2.cu` (ported from mainline, ref `95887577`) | `ssm-scan.cu` (ported, PDL stripped) |
| fallback | other shapes are declined by `supports_op` and run on the CPU | |

The CUDA `supports_op` checks follow upstream's declared kernel constraints — `d_state`
96/128/256 with `d_head % 16 == 0`, `d_conv` 3/4/5/9/15 with `d_inner % 128 == 0` — plus a
contiguous `A`; anything else falls back to the CPU.

### State

The rolling conv window is part of the sequence state: `llama_state_seq_get/set` serialise it
(state format version 5, session format 11). The speculative checkpoint modes that assume a packed
`s_l` are declined for Mamba-2 models instead of silently producing a wrong state.

## Tests

- `tests/test-mamba2.cpp` — chunked prefill against token-by-token decode for `d_conv` 3/4/5 and
  for `d_state` 8 and 96, including a `n_head = 4, n_group = 2` case that exercises the head→group
  mapping; plus the snapshot mapping and resume. Runs in `ctest`, on the CPU reference.
