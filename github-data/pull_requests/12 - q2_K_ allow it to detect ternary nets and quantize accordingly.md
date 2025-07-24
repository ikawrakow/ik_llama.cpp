### 🔀 [#12](https://github.com/ikawrakow/ik_llama.cpp/pull/12) - q2_K: allow it to detect ternary nets and quantize accordingly

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-08-05 |
| **Updated** | 2024-08-05 |

---

#### Description

It looks like they have abandoned the Bitnet quants in PR-8151 in `llama.cpp` and are now going for quantization types in blocks of 256 similar to k- and i-quants. This of course removes support for 3B Bitnet (number of columns is not a multiple of 256) without clunky stuff such as padding, so they are going for [TriLM](https://huggingface.co/collections/SpectraSuite/trilms-unpacked-668d5f62afe0f4036925b1d2) instead, being excited about the newly added `TQ1_0` and `TQ2_0` quantizations, and `TQ2_0` being the fastest quant around on `AVX2`. So, I decided to check how it compares to the CPU implementation here.

The `IQ1_BN` and `IQ2_BN` quants in this repo rely on the tensors in the model converted to `GGUF`  being prepared as ternary, with separate tensors holding the scales. Instead of adding yet another hack to the `convert_hf_to_gguf.py` conversion script, for a quick comparison I added to the `Q2_K` quantization function a ternary net detection. If a ternary net is detected, the quants only take values `0, 1, 2`, all block scales and mins are set to one, and the super-block scale/min are set to the max value found in the row. But to be able to quantize to `Q2_K_S` without an imatrix, I also needed the ability to ignore the build-in imatrix rules, which I added to the `llama-quantize` tool and to `llama.cpp`. With these changes, a `Q2_K_S` quantization of the 3.9B TriLM model matches `fp16` perplexity (using `Q6_K` for `output.weight` and `Q4_K` for `token_embedding.weight`). It is actually even slightly better than `fp16`, I'm getting `PPL = 11.1531` for `fp16` and `PPL = 11.1240` for `Q2_K_S`.

We can now compare performance of `Q2_K_S` to the new `TQ_2` quantization in `llama.cp`. I'm using the 3.9B TriLM variant. The command line to quantize with this PR is
```
./bin/llama-quantize --pure --output-weight-type q6_K --token-embedding-type q4_K --ignore-imatrix-rules $trilm_model $output_file q2_K_S`
```

Here is what I find for `PR-8151` on my Ryzen-7950X CPU:

| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| llama ?B TQ2_0 - 2.06 bpw ternary |   1.08 GiB |     3.99 B | CPU        |      16 |         pp512 |    275.78 ± 0.68 |
| llama ?B TQ2_0 - 2.06 bpw ternary |   1.08 GiB |     3.99 B | CPU        |       2 |         tg128 |     29.69 ± 0.07 |
| llama ?B TQ2_0 - 2.06 bpw ternary |   1.08 GiB |     3.99 B | CPU        |       4 |         tg128 |     46.65 ± 0.07 |
| llama ?B TQ2_0 - 2.06 bpw ternary |   1.08 GiB |     3.99 B | CPU        |       8 |         tg128 |     48.15 ± 0.03 |
| llama ?B TQ2_0 - 2.06 bpw ternary |   1.08 GiB |     3.99 B | CPU        |      16 |         tg128 |     46.13 ± 0.03 |

And here is what I get for `Q2_K_S` in this repo:

| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| llama ?B Q2_K - Small          |   1.33 GiB |     3.99 B | CPU        |      16 |         pp512 |    360.60 ± 0.92 |
| llama ?B Q2_K - Small          |   1.33 GiB |     3.99 B | CPU        |       2 |         tg128 |     25.81 ± 0.04 |
| llama ?B Q2_K - Small          |   1.33 GiB |     3.99 B | CPU        |       4 |         tg128 |     39.91 ± 0.35 |
| llama ?B Q2_K - Small          |   1.33 GiB |     3.99 B | CPU        |       8 |         tg128 |     38.77 ± 2.11 |
| llama ?B Q2_K - Small          |   1.33 GiB |     3.99 B | CPU        |      16 |         tg128 |     38.55 ± 0.02 |

So, despite wasting time for unnecessary block scale multiplications, we still outperform `TQ2_0` by 30% for prompt processing. Token generation is off course memory bound and, with the `Q2_K_S` quantized model being ~25% larger than `TQ2_0`, peak TG performance is ~15% lower.