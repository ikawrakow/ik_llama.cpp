### ✨ [#626](https://github.com/ikawrakow/ik_llama.cpp/issues/626) - Feature Request: Add IQK GEMM for IQ1_M

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-18 |
| **Updated** | 2025-07-18 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Quite a few people are trying to run Unsloth models that contain tensors quantized with `IQ1_M`. In addition, there are now the quantization recipes prepared by the @Thireus GGUF suite, which also tend to contain `IQ1_M` when a low-bpw has been requested.

When a model contains `IQ1_M` FFN tensors and `-fmoe` is specified, `ik_llama.cpp` will crash with an assert when the number of tokens processed by one of the routed experts is less than 32. This is due to the fused `ffn_up+ffn_gate` op assuming the presence of an IQK GEMM kernel, which is not implemented.

So, either add IQK GEMM for `IQ1_M`, or at least quard against the absence of a GEMM kernel in the fused `ffn_up+ffn_gate` op CPU implementation.    

### Motivation

Quite a few people are trying to run Unsloth models that contain tensors quantized with `IQ1_M`. In addition, there are now the quantization recipes prepared by the @Thireus GGUF suite, which also tend to contain `IQ1_M` when a low-bpw has been requested.

When a model contains `IQ1_M` FFN tensors and `-fmoe` is specified, `ik_llama.cpp` will crash with an assert when the number of tokens processed by one of the routed experts is less than 32. This is due to the fused `ffn_up+ffn_gate` op assuming the presence of an IQK GEMM kernel, which is not implemented.

### Possible Implementation

Either add IQK GEMM for `IQ1_M`, or at least quard against the absence of a GEMM kernel in the fused `ffn_up+ffn_gate` op CPU implementation.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-07-18** at **14:43:32**:<br>

I'll not open a new issue regarding unsloths Kimi-K2-Instruct-IQ1_S failing with `-fmoe` as discussed on other threads here and [reported on hugging face here](https://github.com/ikawrakow/ik_llama.cpp/issues/626). I also recreated the issue and observed removing `-fmoe` allows that model to run.

I confirmed using gguf-dump.py script that the model in question indeed has a handfull of IQ1_M ffn tensors:
```bash
$ cat logs/gguf-dump-Kimi-K2-Instruct-UD-IQ1_S-0000* | grep IQ1_M
    163: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.18.ffn_gate_exps.weight
    167: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.18.ffn_up_exps.weight
    111: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.50.ffn_gate_exps.weight
    115: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.50.ffn_up_exps.weight
    129: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.51.ffn_gate_exps.weight
    133: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.51.ffn_up_exps.weight
    147: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.52.ffn_gate_exps.weight
    151: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.52.ffn_up_exps.weight
    165: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.53.ffn_gate_exps.weight
    169: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.53.ffn_up_exps.weight
    183: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.54.ffn_gate_exps.weight
    187: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.54.ffn_up_exps.weight
     21: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.56.ffn_gate_exps.weight
     25: 5637144576 |  7168,  2048,   384,     1 | IQ1_M   | blk.56.ffn_up_exps.weight
```

Given the "unsloth dynamic" is to change the tensor size up and down across layers for the same tensor name, it wasn't obvious from the first GGUF splits that it contained IQ1_M.

---

👤 **ikawrakow** commented the **2025-07-18** at **14:46:02**:<br>

I created issue #626 for this, so no need to add another one.

---

👤 **ubergarm** commented the **2025-07-18** at **17:34:41**:<br>

Confirmed I can now run unsloths `Kimi-K2-Instruct-UD-IQ1_S-00001-of-00006.gguf` with `-fmoe`! Thanks!

```
$ ./build/bin/llama-server --version
version: 3808 (38012f72)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
```