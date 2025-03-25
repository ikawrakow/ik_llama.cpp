# ik_llama.cpp/example/sweep-bench

Benchmark the prompt processing and token generation performance of `ik_llama.cpp`
by doing a sweep over a whole context size and gathering performance metrics
in each ubatch-sized window. Only a single token sequence is used.

The benchmark steps are:

for each ubatch-sized window in context:

    1. generate ubatch/4 tokens (not the whole window to save some time)
    2. measure generation performance
    3. remove generated tokens from KV cache
    4. prepare a ubatch-sized batch of random tokens
    4. process prepated batch
    5. measure prompt processing performance

The purpose of the benchmark is to visualize how the performance changes with
the context size without averaging the metrics values over the whole context.

## Usage

./llama-sweep-bench -c 8704 -ub 512 -m models/Meta-Llama-3.2-3B-Instruct-Q8_0.gguf

## Sample results

- `PP` - prompt tokens per ubatch
- `TG` - generated tokens per ubatch
- `N_KV` - current KV cache size
- `T_PP` - prompt processing time (i.e. time to first token)
- `S_PP` - prompt processing speed (`(B*PP)/T_PP` or `PP/T_PP`)
- `T_TG` - time to generate all batches
- `S_TG` - text generation speed (`(B*TG)/T_TG`)

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.100 |   465.51 |    2.311 |    55.38 |
|   512 |    128 |    512 |    1.183 |   432.97 |    1.895 |    67.55 |
|   512 |    128 |   1024 |    1.305 |   392.38 |    2.071 |    61.81 |
|   512 |    128 |   1536 |    1.279 |   400.42 |    2.164 |    59.14 |
|   512 |    128 |   2048 |    1.571 |   325.96 |    2.280 |    56.14 |
|   512 |    128 |   2560 |    1.431 |   357.87 |    2.418 |    52.94 |
|   512 |    128 |   3072 |    1.515 |   337.93 |    2.566 |    49.88 |
|   512 |    128 |   3584 |    1.588 |   322.34 |    2.722 |    47.03 |
|   512 |    128 |   4096 |    1.675 |   305.70 |    2.864 |    44.69 |
|   512 |    128 |   4608 |    1.769 |   289.50 |    2.999 |    42.68 |
|   512 |    128 |   5120 |    1.845 |   277.48 |    3.102 |    41.26 |
|   512 |    128 |   5632 |    1.893 |   270.46 |    3.219 |    39.76 |
|   512 |    128 |   6144 |    1.953 |   262.20 |    3.348 |    38.23 |
|   512 |    128 |   6656 |    2.018 |   253.71 |    3.474 |    36.84 |
|   512 |    128 |   7168 |    2.078 |   246.34 |    3.589 |    35.66 |
|   512 |    128 |   7680 |    2.140 |   239.22 |    3.717 |    34.43 |
|   512 |    128 |   8192 |    2.196 |   233.15 |    3.854 |    33.21 |

### JSONL output

Pass `--output-format jsonl` to output JSONL instead of Markdown, á la

```json lines
{"n_kv_max": 8704, "n_batch": 2048, "n_ubatch": 512, "flash_attn": 0, "n_gpu_layers": -1, "n_threads": 32, "n_threads_batch": 32, "pp": 512, "tg": 128, "n_kv": 0, "t_pp": 1.093814, "speed_pp": 468.086884, "t_tg": 1.780312, "speed_tg": 71.897514 }
{"n_kv_max": 8704, "n_batch": 2048, "n_ubatch": 512, "flash_attn": 0, "n_gpu_layers": -1, "n_threads": 32, "n_threads_batch": 32, "pp": 512, "tg": 128, "n_kv": 512, "t_pp": 1.169302, "speed_pp": 437.868073, "t_tg": 1.897474, "speed_tg": 67.458099 }
{"n_kv_max": 8704, "n_batch": 2048, "n_ubatch": 512, "flash_attn": 0, "n_gpu_layers": -1, "n_threads": 32, "n_threads_batch": 32, "pp": 512, "tg": 128, "n_kv": 1024, "t_pp": 1.183700, "speed_pp": 432.542053, "t_tg": 2.059179, "speed_tg": 62.160694 }
{"n_kv_max": 8704, "n_batch": 2048, "n_ubatch": 512, "flash_attn": 0, "n_gpu_layers": -1, "n_threads": 32, "n_threads_batch": 32, "pp": 512, "tg": 128, "n_kv": 1536, "t_pp": 1.428625, "speed_pp": 358.386566, "t_tg": 2.160639, "speed_tg": 59.241734 }
{"n_kv_max": 8704, "n_batch": 2048, "n_ubatch": 512, "flash_attn": 0, "n_gpu_layers": -1, "n_threads": 32, "n_threads_batch": 32, "pp": 512, "tg": 128, "n_kv": 2048, "t_pp": 1.360647, "speed_pp": 376.291595, "t_tg": 2.274003, "speed_tg": 56.288403 }
```
