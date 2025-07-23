### 🔀 [#312](https://github.com/ikawrakow/ik_llama.cpp/pull/312) - Improved IQ2_XS quantization

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-05 |
| **Updated** | 2025-04-07 |

---

#### Description

The table shows PPL comparisons between the main branch and this PR for LLaMA-v1-7B<sup>1</sup>(L1-7B in the table), LLaMA-v2-7B<sup>1</sup> (L2-7B), Mistral-7B<sup>1</sup> (M-7B), LLaMA-3.1-8B-Instruct (L3-8B), and DeepSeek-V2-Lite (DSL). Context is always 512 tokens. Also given are the quantization times (Q-time for short in the table) in seconds on a Ryzen-7950X CPU. Tested is "pure" quantization (i.e., using the `--pure` option of `llama-quantize`) with token embeddings and output tensor set to `Q8_0`. The quantization command line is
```
./bin/llama-quantize --imatrix $imatrix --token-embedding-type q8_0 --output-tensor-type q8_0 --pure $model $output iq2_xs
```

| Model |  Quantization |  PPL (main) |  PPL (this PR) | Q-time (main) | Q-time (this PR) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| L1-7B | IQ2_XS | 8.2767 | 8.2773 | N/A<sup>2</sup> | N/A<sup>2</sup> |
| L2-7B | IQ2_XS | 8.0856 | 8.1669 | 156.4 | 132.6 |
|M-7B | IQ2_XS | 7.3882 | 7.3447 | 169.1 | 143.3 |
| L3-8B | IQ2_XS | 13.4294 | 13.0922 | 171.3 | 145.8 |
| DSL | IQ2_XS | 9.8273 | 9.4692 | 302.7 | 257.0 |

All models are improved except LLaMA-v2 (but I might have given it too much importance when fine tuning the hyper parameters in the original `IQ2_XS` PR). Quantization time is reduced by about 18%.
 
___
<sup>1</sup> Why use such ancient models? The LLaMA-v1 models were the basis for k-quants development. I-quants were developed using LLaMA-v1, LLaMA-v2 and Mistral-7B. In my experience, if a quantization technique does well on all 3 of these, it is (almost) guaranteed to do well on any other model out there. 

<sup>2</sup> I have this model on an old HDD. In this case quantization time is dominated by the time needed to read the data from the HDD. I could have copied the model to the SSD drive, but I think the timing for the other models gives enough indication of the relative performance.