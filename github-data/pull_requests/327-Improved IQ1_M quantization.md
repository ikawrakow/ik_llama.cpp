### 🔀 [#327](https://github.com/ikawrakow/ik_llama.cpp/pull/327) - Improved IQ1_M quantization

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-13 |
| **Updated** | 2025-04-13 |

---

#### Description

I was experimenting with LlaMA-4-Scout quantization and was bothered by the extremely long quantization time of `IQ1_M`, so looked into speeding things up.

This PR improves `IQ1_M` quantization speed by a huge margin. There is also a minor improvement in quantization accuracy. 

The table shows PPL comparisons between the main branch and this PR for LLaMA-v1-7B<sup>1</sup>(L1-7B in the table), LLaMA-v2-7B<sup>1</sup> (L2-7B), Mistral-7B<sup>1</sup> (M-7B), LLaMA-3.1-8B-Instruct (L3-8B), and DeepSeek-V2-Lite (DSL). Context is always 512 tokens. Also given are the quantization times (Q-time for short in the table) in seconds on a Ryzen-7950X CPU. Unlike earlier quantization improvement PRs, which used "pure" quantization (`--pure` command line option in `llama-quantize`), tested is the default `IQ1_M` quantization mix. 

| Model |  Quantization |  PPL (main) |  PPL (this PR) | Q-time (main) | Q-time (this PR) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| L1-7B | IQ1_M | 10.9274 | 10.8046 | N/A<sup>2</sup> | N/A<sup>2</sup> |
| L2-7B | IQ1_M | 10.7642 | 10.6809 | 129.4 | 52.8 |
|M-7B | IQ1_M | 9.6336 | 9.6236 | 146.1 | 58.4 |
| L3-8B | IQ1_M | 22.7422 | 21.9715 | 148.1 | 60.0 |
| DSL | IQ1_M | 9.2758 | 9.1137 | 267.4 | 109.2 |

Speedup for the default `IQ1_M` quantization mix is in the range of 2.5X. When quantizing pure `IQ1_M`, the speedup is about 3X.

___
<sup>1</sup> Why use such ancient models? The LLaMA-v1 models were the basis for k-quants development. I-quants were developed using LLaMA-v1, LLaMA-v2 and Mistral-7B. In my experience, if a quantization technique does well on all 3 of these, it is (almost) guaranteed to do well on any other model out there. 

<sup>2</sup> I have this model on an old HDD. In this case quantization time is dominated by the time needed to read the data from the HDD. I could have copied the model to the SSD drive, but I think the timing for the other models gives enough indication of the relative performance.