### 🔀 [#84](https://github.com/ikawrakow/ik_llama.cpp/pull/84) - Better model info

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-10-10 |
| **Updated** | 2024-10-10 |

---

#### Description

In the quantization literature they always ignore the token embedding and output tensors (they leave them as `f16`). But when `llama.cpp` loads a model, it prints a bits-per-weight (bpw) value that is basically `total file size on disk / total number of parameters`. As this includes the output tensor, which is almost always quantized with more bpw, this makes the i- and k-quants appear not competitive.

So, this PR adds an additional print out that tells us the model size excluding `token_embd.weight` and `output.weight`, and the corresponding bpw. Here is an example from LLaMA-3.1-8B-Instruct quantized with `IQ2_XS`:
```
...
llm_load_print_meta: model type       = 8B
llm_load_print_meta: model ftype      = IQ2_XS - 2.3125 bpw
llm_load_print_meta: model params     = 8.030 B
llm_load_print_meta: model size       = 3.880 GiB (4.150 BPW) 
llm_load_print_meta: repeating layers = 1.923 GiB (2.366 BPW, 6.980 B parameters)
llm_load_print_meta: general.name     = Meta Llama 3.1 8B Instruct
...
```

I also added one extra digit (two decimal places is a bit too little for bpw values).