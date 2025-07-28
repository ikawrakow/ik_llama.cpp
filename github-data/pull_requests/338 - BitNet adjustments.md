## 🔀 [Pull Request #338](https://github.com/ikawrakow/ik_llama.cpp/pull/338) - BitNet adjustments

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/bitnet_adjustments` |
| **Target Branch** | `main` |
| **Created** | 2025-04-22 |
| **Updated** | 2025-04-22 |
| **Merged** | 2025-04-22 |

---

## 📄 Description

Two small tweaks to [#337](https://github.com/ikawrakow/ik_llama.cpp/issues/337):
* Use `create_tensor` instead of `ml.create_tensor`. This is necessary for tensor overrides to work (in case one would ever want to use tensor overrides with a BitNet model)
* Use `output.weight` instead of `token_embd.weight` for the final matrix multiplication. This improves CUDA performance quite a bit as `token_embd.weight` is on the host, so needs to be copied to the GPU each time it is needed (or the matrix multiplication is done on the CPU when running TG). I see that MicroSoft have decided to have `output.weight` stored in the model, even though it is identical to `token_embd.weight` (in the initial BitNet models one simply reused `token_embd.weight`). This makes the model quite a bit larger than it needs to be. Go figure.

---

## 💬 Conversation

👤 **saood06** commented on **2025-04-22** at **07:01:54**

>Use `create_tensor` instead of `ml.create_tensor`. This is necessary for tensor overrides to work (in case one would ever want to use tensor overrides with a BitNet model)

Yes I noticed that, I just didn't want to change until I tested if it worked first.


>Use `output.weight` instead of `token_embd.weight` for the final matrix multiplication. This improves CUDA performance quite a bit as `token_embd.weight` is on the host, so needs to be copied to the GPU each time it is needed (or the matrix multiplication is done on the CPU when running TG). I see that MicroSoft have decided to have `output.weight` stored in the model, even though it is identical to `token_embd.weight` (in the initial BitNet models one simply reused `token_embd.weight`). This makes the model quite a bit larger than it needs to be. Go figure.

Interesting. There is a discussion on the huggingface that the model is larger than it has to be. Can we have change this to have smaller model size or is the performance benefit worth it (if it can't be duplicated on runtime for CUDA)?

I also noticed when converting the two tensors ended up with different quants.

```
[   1/ 333]                        output.weight - [ 2560, 128256,     1,     1], type =    f16, converting to q6_K .. size =   626.25 MiB ->   256.86 MiB
[   2/ 333]                    token_embd.weight - [ 2560, 128256,     1,     1], type =    f16, converting to iq4_nl .. size =   626.25 MiB ->   176.13 MiB
```

---

👤 **ikawrakow** commented on **2025-04-22** at **07:10:08**

> I also noticed when converting the two tensors ended up with different quants.

These are the built-in defaults. If one wants to have something else one needs to use `--token-embedding-type` and `--output-tensor-type` (or `--custom-q`). 

> Interesting. There is a discussion on the huggingface that the model is larger than it has to be. Can we have change this to have smaller model size or is the performance benefit worth it (if it can't be duplicated on runtime for CUDA)?

The two tensors are stored in the model. If we wanted to avoid the duplication, we need to add logic that checks if `output.weight` and `token_embd.weight` are the same. But if one is running on CUDA, one wants to have `output.weight` offloaded to the GPU to avoid the copy on each evaluation. `token_embd.weight` needs to stay on the host because in `llama.cpp` the input (token embedding, attention mask, etc.) is always prepared on the host. So, the only situation where we would gain is for CPU-only inference, where we wouldn't have the same tensor stored twice in memory.