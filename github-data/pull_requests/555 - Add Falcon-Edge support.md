### 🔀 [#555](https://github.com/ikawrakow/ik_llama.cpp/pull/555) - Add Falcon-Edge support

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-25 |
| **Updated** | 2025-06-26 |

---

#### Description

Closes #551 

How to use:

1. Grab a GGUF containing Microsoft's `i2_s` quant packing. E.g.,
```
huggingface-cli download --local-dir falcon tiiuae/Falcon-E-3B-Instruct-GGUF
```

2. Convert to `ik_llama.cpp` quants `iq2_bn` or `iq1_bn`. `iq2_bn` uses 2 bits per weight (bpw), `iq1_bn` uses 1.625 bpw. `iq2_bn` is faster for prompt processing, and may also be faster for token generation (TG) on devices with limited computing power. `iq1_bn` uses 20% less RAM, so that if TG is memory bound, it will be slightly faster than `iq2_bn`. Command to convert is
```
./bin/llama-quantize --allow-requantize falcon/ggml-model-i2_s.gguf falcon_iq2_bn.gguf iq2_bn 
```
(replace `iq2_bn` with `iq1_bn` if you prefer the smaller variant.

3. Utilize the just created model file in the usual way with `llama-cli, llama-server`, etc.