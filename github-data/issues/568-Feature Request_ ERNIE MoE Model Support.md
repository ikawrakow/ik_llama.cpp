### ✨ [#568](https://github.com/ikawrakow/ik_llama.cpp/issues/568) - Feature Request: ERNIE MoE Model Support

| **Author** | `Downtown-Case` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-01 |
| **Updated** | 2025-07-18 |

---

#### Description

New MoE series from Baidu: https://github.com/PaddlePaddle/ERNIE

> ...We designed a heterogeneous MoE structure, incorporated modality-isolated routing, and employed router orthogonal loss and multimodal token-balanced loss...

This bit caught my eye:

>  ...For inference, we propose multi-expert parallel collaboration method and convolutional code quantization algorithm to achieve **4-bit/2-bit lossless quantization...**

https://github.com/PaddlePaddle/ERNIE?tab=readme-ov-file#model-development

> ERNIE-4.5-300B-A47B: BF16 / W4A16C16 / W8A16C16 / W4A8C8 / FP8 / **2Bits**

https://huggingface.co/baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle

2 bit QAT on a 300B? Now that's *interesting.*

I am leaving this as a drive by request, as I still have other issues (like testing Hunyuan!) in my queue.

Related issue: https://github.com/ggml-org/llama.cpp/pull/14408

***

Unrelated, but Huawei just dropped a 72B MoE trained on NPUs: https://huggingface.co/IntervitensInc/pangu-pro-moe-model

Seems to be *specifically* designed for even multi-device distribution:

> We proposed a new type of Mixture of Grouped Experts (MoGE), which groups experts in the expert selection stage and constrains tokens to activate equal experts in each group, thereby achieving natural load balancing between devices.

LG is about to release EXAONE 4.0 as well: https://github.com/ggml-org/llama.cpp/issues/14474

I can't keep up with any of this, lol.

---

#### 💬 Conversation

👤 **Downtown-Case** commented the **2025-07-01** at **19:48:31**:<br>

From the paper:

https://yiyan.baidu.com/blog/publication/ERNIE_Technical_Report.pdf

```

To address the aforementioned issues, we propose Convolutional Code Quantization (CCQ), a scalar
quantization algorithm based on the convolutional code. The approach not only retains the high-precision
data quantization capability of vector quantization but also preserves the low computational complexity
of scalar quantization. Combined with scale quantization and optimization, we achieve the highest
possible compression ratio while simultaneously minimizing inference overhead.
Convolutional Codebook. Inspired by QTIP (Tseng et al., 2024b), we innovatively integrate convolutional code with scalar quantization through a series of meticulously designed coding structures. Based
on convolutional codes, we construct a lookup-free codebook that achieves a linear mapping between the
codebook and weight vectors, thereby optimizing inference performance. Meanwhile, by drawing on the
concept of data mapping from vector quantization, we minimize the performance degradation of the
model under extremely low-bit conditions.
Hybrid Encoding. We employ convolutional codes with varying coding configurations to accommodate the storage of encoded values in INT8 and INT16 formats. As a result, we successfully compress
4-bit scalar quantization to an equivalent of 2.75 bits and 3-bit scalar quantization to 2.5 bits.
Code Clustering. Furthermore, by analyzing the distribution of encoded values across each channel,
we observe that they conform to a normal distribution, enabling deeper compression along the coding
dimension. Through clustering of the convolutional codes, we can compress any coding configuration to
an equivalent of 2 bits, thereby further enhancing the model compression rate.

```

(sorry for formatting).

There's also details on KV cache quantization.

---

👤 **Ph0rk0z** commented the **2025-07-11** at **12:25:10**:<br>

I think we're going to be stuck trying to run Paddle. If it does also quant kv, that means fully offloaded ernie on 4x3090. Their deepseek quant size is impressive too.. only 184GB.

There's a PR: https://github.com/ggml-org/llama.cpp/pull/14658 that can be ported now.

---

👤 **Ph0rk0z** commented the **2025-07-11** at **12:25:10**:<br>

I think we're going to be stuck trying to run Paddle. If it does also quant kv, that means fully offloaded ernie on 4x3090. Their deepseek quant size is impressive too.. only 184GB.

---

👤 **fizzAI** commented the **2025-07-18** at **02:17:16**:<br>

The above PR (https://github.com/ggml-org/llama.cpp/pull/14658) was just finalized and merged into mainline, would be nice to see if anyone is smart enough to port it properly :3