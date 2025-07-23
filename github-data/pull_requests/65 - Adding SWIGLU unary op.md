### 🔀 [#65](https://github.com/ikawrakow/ik_llama.cpp/pull/65) - Adding SWIGLU unary op

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-28 |
| **Updated** | 2024-09-28 |

---

#### Description

Phi-3(.5) (and also ChatGLM) uses a "SWIGLU" operation in its FFN. There is nothing special about "SWIGLU", it is just that the `ffn_up` tensor is actually a combination of the usual `ffn_up` and `ffn_gate` tensors, where in each row the first half contains the `ffn_up` weights and the second half has the `ffn_gate` weights. So that, to implement
```
silu(ffn_up * A) * (ffn_gate * A)
```
(`A` are the activations passed into the FFN network), which is common for many LLMs, one needs `swiglu(ffn_up * A) `. In a typical `ggml` style, instead of adding a dedicated op for that, `ggml` models it as 4 (!) operations
```
x1 = ggml_cont(ffn_up, first row half)
x2 = ggml_cont(ffn_up, second row half)
x3 = ggml_silu(x1)
x4 = ggml_mul(x2, x3)
```
`ggml_cont(x)` is basically a copy operation. The result of this is that on my Ryzen-7950X CPU more than 5% (!) of PP time is spent in `ggml_cont`, i.e., in completely unnecessary copies<sup>1</sup> 

To remedy this unfortunate `ggml` implementation detail, this PR adds a dedicated `ggml_swiglu` operation, implemented for the CPU, CUDA, and Metal back-ends. We get
* ~4% PP speedup on the CPU (Ryzen-7950X, Ryzen-5975WX, M2-Max) 
* ~3% PP speedup on Metal (M2-Max GPU)
* ~12% PP speedup on CUDA (RTX-4080)
* ~1-2% speedup for TG on all tested platforms

**Of note**: Phi-3.5 has been trained in `bf16`. To make sure that my `ggml_swiglu` implementation is correct, I ran a full Wikitext2 perplexity calculation on the CPU. The Ryzen-7950X CPU has native `bf16` support, so I used a GGUF converted directly to `bf16` from the safetensors on HF. As FA with `bf16` KV-cache is slightly faster when there is native `bf16` support, I also used that. The final PPL for a context of 512 tokens is `6.5556`. In comparison, the `fp16` CUDA result is `6.5816`. The difference is small but definitely outside of what one would expect from numerical roundoff errors alone. I guess, there are a few model weights in Phi-3.5-mini, as well as some activations, that fall outside of the `fp16` range.

===
<sup>1</sup> Phi-3(-5) also uses a combined `QKV` tensor, which triggers additional `ggml_cont` operations as implemented in `llama.cpp`:
```
cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, attn_norm_output); // this is the QKV * A matrix multiplication
Qcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd,     n_tokens, cur->nb[1], 0 * sizeof(float) * (n_embd)));      
Kcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1 * sizeof(float) * (n_embd)));      
Vcur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa)));      
Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
```
The ` ggml_reshape_3d` op requires the tensor being reshaped to be contiguous, so `Qcur` and `Kcur` are created by copying the appropriate data out of `QKV * A`. The `Vcur` copy is completely unnecessary. The exact same result can be achieved, without using any copies, via
```
Qcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head, n_tokens, n_embd_head*sizeof(float), cur->nb[1], cur, 0 * sizeof(float) * (n_embd));
Kcur = ggml_view_3d(ctx0, cur, n_embd_head, n_head_kv, n_tokens, n_embd_head*sizeof(float), cur->nb[1], 1 * sizeof(float) * (n_embd));
Vcur = ggml_view_2d(ctx0, cur, n_embd_gqa, n_tokens, cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa));
```
This results in an additional 2-3% speedup of PP-512(Phi-3.5-mini) when running on the CPU. Unfortunately CUDA becomes massively slower, so I need to investigate and hence have left this change for a future PR.

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2024-09-28** at **10:07:59**:<br>

OK, Phi-3.5 has a 128k context, so let's run a benchmark with a longer context, say, 8k tokens. Here is what I get after this PR on a Ryzen-7950X CPU for Phi-3.5-mini:

| model                          |       size | backend    | threads | type_k | type_v | fa |          test |              t/s |
| ------------------------------ | ---------: | ---------- | ------: | -----: | -----: | -: | ------------: | ---------------: |
| phi3 3B BF16                   |   7.12 GiB | CPU        |      16 |    - |  -  |  0 |     pp8192 |    218.01 ± 0.37 |
| phi3 3B BF16                   |   7.12 GiB | CPU        |      16 |   bf16 |   bf16 |  1 |        pp8192 |    307.62 ± 1.23 |

Mainline `llama.cpp` has no `bf16` support, so we need to use `fp16` (`bf16` will run but it is infinitely slow). Here is what I get with the `llama.cpp` version from this morning (`build: 44f59b43 (3829)`)

| model                          |       size | backend    | threads | fa |          test |                  t/s |
| ------------------------------ | ---------: |  ---------- | ------: | -: | ------------: | -------------------: |
| phi3 3B F16                    |   7.12 GiB | CPU        |      16 |  1 |        pp8192 |         32.28 ± 0.01 |
| phi3 3B F16                    |   7.12 GiB | CPU        |      16 |  0 |        pp8192 |         81.05 ± 0.05 |

The best calculation here (FA with `bf16` for K- and V-cache) is 3.8X faster than the best `llama.cpp` has to offer (no FA). Out FA speeds things up by 41%, `llama.cpp` FA slows things down 2.5X. A user who has not taken the time to investigate FA performance in `llama.cpp`, and is running on a Zen4 CPU, will observe a 9.5X difference in processing speed between here and mainline `llama.cpp`.