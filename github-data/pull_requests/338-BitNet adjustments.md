### 🔀 [#338](https://github.com/ikawrakow/ik_llama.cpp/pull/338) - BitNet adjustments

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-22 |
| **Updated** | 2025-04-22 |

---

#### Description

Two small tweaks to #337:
* Use `create_tensor` instead of `ml.create_tensor`. This is necessary for tensor overrides to work (in case one would ever want to use tensor overrides with a BitNet model)
* Use `output.weight` instead of `token_embd.weight` for the final matrix multiplication. This improves CUDA performance quite a bit as `token_embd.weight` is on the host, so needs to be copied to the GPU each time it is needed (or the matrix multiplication is done on the CPU when running TG). I see that MicroSoft have decided to have `output.weight` stored in the model, even though it is identical to `token_embd.weight` (in the initial BitNet models one simply reused `token_embd.weight`). This makes the model quite a bit larger than it needs to be. Go figure.

---

#### 💬 Conversation

👤 **saood06** commented the **2025-04-22** at **07:01:54**:<br>

>     * Use `create_tensor` instead of `ml.create_tensor`. This is necessary for tensor overrides to work (in case one would ever want to use tensor overrides with a BitNet model)

Yes I noticed that, I just didn't want to change until I tested if it worked first.


>Use `output.weight` instead of `token_embd.weight` for the final matrix multiplication. This improves CUDA performance quite a bit as `token_embd.weight` is on the host, so needs to be copied to the GPU each time it is needed (or the matrix multiplication is done on the CPU when running TG). I see that MicroSoft have decided to have `output.weight` stored in the model, even though it is identical to `token_embd.weight` (in the initial BitNet models one simply reused `token_embd.weight`). This makes the model quite a bit larger than it needs to be. Go figure.

Interesting. There is a discussion on the huggingface that the model is larger than it has to be. Can we have change this to have smaller model size or is the performance benefit worth it (if it can't be duplicated on runtime for CUDA)?

I also noticed when converting the two tensors ended up with different quants.

```
[   1/ 333]                        output.weight - [ 2560, 128256,     1,     1], type =    f16, converting to q6_K .. size =   626.25 MiB ->   256.86 MiB
[   2/ 333]                    token_embd.weight - [ 2560, 128256,     1,     1], type =    f16, converting to iq4_nl .. size =   626.25 MiB ->   176.13 MiB
```