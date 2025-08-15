### 🔀 [#19](https://github.com/ikawrakow/ik_llama.cpp/pull/19) - Skip barriers of noops

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-08-14 |
| **Updated** | 2024-08-14 |

---

#### Description

`GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_PERMUTE, GGML_OP_TRANSPOSE`, along with `GGML_OP_NONE`, are all noops in `ggml`. I.e., nothing happens. But `ggml` still has a thread barrier after them, which wastes time. The waste is not too bad for large models where computations are long compared to the time taken for thread synchronization. But for small models skipping those unnecessary waits makes a noticeable difference.

Let's look at a really tiny model - the [99M parameter TriLM ternary model](https://huggingface.co/SpectraSuite/TriLM_99M_Unpacked) quantized with `IQ2_TN`.  The following table compares performance for PP-512 and TG-128 with and without the change in this PR

| CPU        | threads |          test |    t/s (main)    |  t/s (PR)        |  Speedup |
| ---------- | ------: | ------------: | ---------------: | ---------------: | -------: |
| Ryzen-7950X|      16 |         pp512 | 11386.75 ± 19.08 | 11587.58 ± 34.26 |  1.018   |   
| Ryzen-7950X|       8 |         tg128 |   1312.25 ± 1.02 |   1460.80 ± 1.69 |  1.113   |   
| M2-Max     |       8 |         pp512 |  7642.81 ± 22.07 |   7680.29 ± 9.29 |  1.005   |   
| M2-Max     |       8 |         tg128 |   992.83 ± 18.17 |  1096.47 ± 14.45 |  1.104   |

So, basically, for such a small model `ggml` spends 10% of its time waiting for threads to pass through a barrier after a noop when generating tokens.

There are other barriers that can be eliminated. E.g., the typical attention block involves matrix multiplications of the `Q, K` and `V` tensors with the **same** activations, so there is no need to synchronize threads after each such matrix multiplications. In a similar way, in the feed-forward portion of the network the `ffn_up` and `ffn_gate` tensors multiply the same activations, so one can save another barrier there. This is left for a future PR.