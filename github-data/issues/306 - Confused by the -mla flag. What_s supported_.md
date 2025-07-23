### 📝 [#306](https://github.com/ikawrakow/ik_llama.cpp/issues/306) - Confused by the -mla flag. What's supported?

| **Author** | `Downtown-Case` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-02 |
| **Updated** | 2025-04-02 |

---

#### Description

Trying to load Deepseek 32B (specifically an IQ4_KS_RQ quantization I just made) with the -mla 2 (or -mla any value) flag gives me a segfault.

`./build/bin/llama-server --model /Models/GGUF/Deepseek-32B-IQ4_KS_R4.gguf --ctx-size 2048 -mla 2 -fa --n-gpu-layers 65 --parallel 1 --threads 1 --host 127.0.0.1 --port 8080`


```
...
llama_kv_cache_init: layer 63: n_embd_head_qk_rope = 128, kv_lora_rank = 0
llama_kv_cache_init:      CUDA0 KV buffer size =    32.00 MiB
llama_new_context_with_model: KV self size  =   32.00 MiB, c^KV (f16):   32.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
fish: Job 1, './build/bin/llama-server --mode…' terminated by signal SIGSEGV (Address boundary error)

```
Is that only supported by full Deepseek MoE, not the Qwen distills?

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-04-02** at **14:55:01**:<br>

As far as I know, the distilled models use a standard attention mechanism (same as the underlying model used to prepare the distillation, i.e., Qwen, LLaMA-3, etc.). At least [this one](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF) does.

I guess, I should add checks to only allow MLA when we have a model using MLA.

---

👤 **Downtown-Case** commented the **2025-04-02** at **14:59:41**:<br>

Interesting, thanks. I'm playing catch up here, and did find the MLA paper.


What major models *do* support MLA? Just the MoE deepseek releases? Adapted finetunes hiding on huggingface?

---

👤 **Downtown-Case** commented the **2025-04-02** at **14:59:41**:<br>

Interesting, thanks. I'm playing catch up here, and did find the MLA paper.


What major models *do* support MLA? Just the MoE deepseek releases?

---

👤 **ikawrakow** commented the **2025-04-02** at **15:02:38**:<br>

As far as I know, DeepSeek-V2/V3/R1/Lite are the models that use MLA.

---

👤 **Downtown-Case** commented the **2025-04-02** at **15:17:53**:<br>

Thanks! And I appreciate you posting this repo.