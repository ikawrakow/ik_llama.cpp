### 🔀 [#302](https://github.com/ikawrakow/ik_llama.cpp/pull/302) - Quantization improvements (2)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-31 |
| **Updated** | 2025-04-02 |

---

#### Description

This PR is a follow up of #295. It applies the same approach to type-1 quants (`Q2_K, Q4_K, Q5_K, Q4_1, Q5_1`) and to `IQ3_K`. Quantization speed for `IQ3_K` is improved by a significant margin (up to 40%). Quantization speed for type-1 quants is also slightly improved ($\le 15$%). The changes do not result in PPL improvement for all tested models, but do improve PPL for the models that are more difficult to quantize (e.g., the LLaMA-3 series of models), and avoid a near catastrophic failure of `IQ3_K` on DeepSeek-Lite. 

The following table shows PPL comparisons between the main branch and this PR for LLaMA-v1-7B<sup>1</sup>(L1-7B in the table), LLaMA-v2-7B<sup>1</sup> (L2-7B), Mistral-7B<sup>1</sup> (M-7B), LLaMA-3.1-8B-Instruct (L3-8B), and DeepSeek-V2-Lite (DSL). Context is always 512 tokens. Also given are the quantization times (Q-time for short in the table) in seconds on a Ryzen-7950X CPU. Tested is "pure" quantization (i.e., using the `--pure` option of `llama-quantize`) with token embeddings and output tensor set to `Q8_0`. The quantization command line is
```
./bin/llama-quantize --imatrix $imatrix --token-embedding-type q8_0 --output-tensor-type q8_0 --pure $model $output $quant
```

| Model |  Quantization |  PPL (main) |  PPL (this PR) | Q-time (main) | Q-time (this PR) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| L1-7B | Q4_1 | 5.9773 | 5.9760 | N/A<sup>2</sup> | N/A<sup>2</sup> |
| L2-7B | Q4_1 | 5.8676 | 5.8691 | 33.6 | 29.9 |
|M-7B | Q4_1 | 5.7452 | 5.7471 | 36.7 | 32.3 |
| L3-8B | Q4_1 | 7.5309 | 7.5277 | 38.1 | 34.0 |
| DSL | Q4_1 | 6.8639 | 6.8584 | 84.1 | 75.3 |
| L1-7B | Q5_1 | 5.9183 | 5.9182 | N/A<sup>2</sup> |  N/A<sup>2</sup> |
| L2-7B | Q5_1 | 5.8164 | 5.8175 | 35.6 | 30.8 |
|M-7B | Q5_1 | 5.7067 | 5.7074 | 37.6 | 33.6 |
| L3-8B | Q5_1 | 7.3749 | 7.3759 | 38.7 | 34.7 |
| DSL | Q5_1 | 6.7881 | 6.7875 | 86.4 | 76.5 |
| L1-7B | Q2_K | 7.3154 | 7.2989 | N/A<sup>2,3</sup> |  N/A<sup>2</sup> |
| L2-7B | Q2_K | 7.3044 | 7.2558 | 36.4 | 32.2 |
|M-7B | Q2_K | 6.9507 | 6.9273 | 38.4 |  35.0 |
| L3-8B | Q2_K | 11.546 | 11.458 | 40.1 | 36.5 |
| DSL | Q2_K | 8.3822 | 8.3346 | 89.6 | 83.4 |
| L1-7B | Q4_K | 5.9801 | 5.9779 | N/A<sup>2</sup> |  N/A<sup>2</sup> |
| L2-7B | Q4_K | 5.8675 | 5.8673 |  34.1 | 30.7 |
|M-7B | Q4_K | 5.7449 |  5.7406  | 37.0 | 32.8 |
| L3-8B | Q4_K | 7.5192 | 7.5157 | 38.2 | 34.5 |
| DSL | Q4_K | 6.8607 | 6.8570 | 75.7 | 68.5 |
| L1-7B | Q5_K | 5.9314 | 5.9299 |  N/A<sup>2</sup> |  N/A<sup>2</sup> |
| L2-7B | Q5_K | 5.8144 | 5.8196 |  35.6 | 31.2 |
|M-7B | Q5_K | 5.7030 |  5.7064 | 37.3 | 34.1 |
| L3-8B | Q5_K | 7.3941 | 7.3812 |  38.9 | 34.6 |
| DSL | Q5_K | 6.7929 | 6.7903 | 76.5 | 69.5 |
| L1-7B | IQ3_K | 6.1393 | 6.1377 |  N/A<sup>2</sup> |  N/A<sup>2</sup> |
| L2-7B | IQ3_K | 6.0251 | 6.0227 |  44.7 | 36.9 |
|M-7B | IQ3_K | 5.8835 |  5.8855 | 54.6 | 39.5 |
| L3-8B | IQ3_K | 7.9148 | 7.9189 |  56.3 | 41.4 |
| DSL | IQ3_K | 7.3143 | 7.0409 | 116.4 | 92.5 |

___
<sup>1</sup> Why use such ancient models? The LLaMA-v1 models were the basis for k-quants development. I-quants were developed using LLaMA-v1, LLaMA-v2 and Mistral-7B. In my experience, if a quantization technique does well on all 3 of these, it is (almost) guaranteed to do well on any other model out there. 

<sup>2</sup> I have this model on an old HDD. In this case quantization time is dominated by the time needed to read the data from the HDD. I could have copied the model to the SSD drive, but I think the timing for the other models gives enough indication of the relative performance.

---

#### 💬 Conversation

👤 **saood06** commented the **2025-04-02** at **10:55:25**:<br>

>and avoid a near catastrophic failure of IQ3_K on DeepSeek-Lite.

Interestingly IQ3_K before this PR was actually worse than Q3_K before #295 for DSL.