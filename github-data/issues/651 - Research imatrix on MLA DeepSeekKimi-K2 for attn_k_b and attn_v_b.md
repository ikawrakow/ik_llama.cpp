## 📌 [Issue #651](https://github.com/ikawrakow/ik_llama.cpp/issues/651) - Research: imatrix on MLA DeepSeek/Kimi-K2 for `attn_k_b` and `attn_v_b`

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-26 |
| **Updated** | 2025-07-26 |

---

## 📄 Description

### Previous existing literature and research

We've had some discussions on how to use imatrix on DeepSeek/Kimi-K2 models with MLA to make sure it applies to `attn_k_b` and `attn_v_b` tensors.

When using an imatrix to quantize these messages are expected:
```
====== llama_model_quantize_internal: did not find weights for token_embd.weight
====== llama_model_quantize_internal: did not find weights for blk.0.attn_kv_b.weight
```

This is fine as the token_embd does not use imatrix data, and the strategy is to set `attn_kv_b` always to Q8_0 as it is only used for PP assuming end user mode of `-mla 3`.

However the following messages *not* okay and mean there maybe was an issue with how imatrix data was collected:
```bash
====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
====== llama_model_quantize_internal: did not find weights for blk.0.attn_v_b.weight
```

Given these tensors are good to quantize to speed up TG, ideally there should be imatrix data there.

In the past I had mistakenly left off `-mla 1` while creating imatrix and so was missing the data for `attn_k_b` and `attn_v_b` oops!

More recently with Kimi-K2 I *did* use `-mla 1` but still was missing data for both tensors. One tensors seems because it is named `attn_k_b.weight (reshaped)`. And `attn_v_b` does not appear in the verbose logs. For now I am just quantizing `attn_kv_b` as designed, but now also `attn_k_b`, and `attn_v_b` to Q8_0 on Kimi-K2-Instruct models to preserve perplexity at the cost of some TG speed.

See attached log files, too long to add into details fold:

[imat-kimi-no-mla.log](https://github.com/user-attachments/files/21441623/imat-kimi-no-mla.log)

[imat-kimi-mla-1.log](https://github.com/user-attachments/files/21441622/imat-kimi-mla-1.log)

A few things I can check would be:

### 1. Try it on DeepSeek-V2-Lite
So I just tried llama-imatrix on this model and it seems to use that `attn_k_b.weight (reshaped)` name and I don't ever see `attn_v_b`, though when actually quantizing it doesn't throw a message about missing `attn_v_b`.

```
# llama-quantize
[  11/ 431]                blk.0.attn_k_b.weight - [  128,  8192,     1,     1], type =   bf16, Using custom type q4_0 for tensor blk.0.attn_k_b.weight
====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
converting to q4_0 .. size =     2.00 MiB ->     0.56 MiB
[  12/ 431]                blk.0.attn_v_b.weight - [  512,  2048,     1,     1], type =   bf16, Using custom type q4_0 for tensor blk.0.attn_v_b.weight
converting to q4_0 .. size =     2.00 MiB ->     0.56 MiB
```

```
CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-imatrix \
    -mla 1 \
    --verbosity 2 \
    --layer-similarity \
    -m /mnt/raid/models/ubergarm/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite-64x1.6B-BF16.gguf \
    -f ubergarm-imatrix-calibration-corpus-v02.txt \
    -o /tmp/imatrix-tmp.dat \
    -ngl 99 \
    --ctx-size 512 \
    --threads 1

...

collect_imatrix[1]:     blk.20.ffn_down_shexp.weight, MUL_MAT,  2816 x   512, 0
collect_imatrix[1]:      blk.21.attn_kv_a_mqa.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:             blk.21.attn_q.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]: blk.21.attn_k_b.weight (reshaped), MUL_MAT,   128 x   512, 0
collect_imatrix[1]:        blk.21.attn_output.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:       blk.21.ffn_gate_inp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:      blk.21.ffn_gate_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:        blk.21.ffn_up_exps.weight, MUL_MAT_ID,  2048 x   512, 0
collect_imatrix[1]:      blk.21.ffn_down_exps.weight, MUL_MAT_ID,  1408 x   512, 0
collect_imatrix[1]:     blk.21.ffn_gate_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:       blk.21.ffn_up_shexp.weight, MUL_MAT,  2048 x   512, 0
collect_imatrix[1]:     blk.21.ffn_down_shexp.weight, MUL_MAT,  2816 x   512, 0
collect_imatrix[1]:      blk.22.attn_kv_a_mqa.weight, MUL_MAT,  2048 x   512, 0
```

---

We've discussed this topic across a number of discussions and PRs hah, here is most recent relevent comments:

## References
* https://github.com/ikawrakow/ik_llama.cpp/pull/642#issuecomment-3109818995
* https://github.com/ikawrakow/ik_llama.cpp/issues/601#issuecomment-3070185792