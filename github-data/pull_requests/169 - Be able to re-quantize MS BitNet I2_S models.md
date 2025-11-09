## 🔀 [Pull Request #169](https://github.com/ikawrakow/ik_llama.cpp/pull/169) - Be able to re-quantize MS BitNet I2_S models

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/convert_i2s` |
| **Target Branch** | `main` |
| **Created** | 2025-01-10 |
| **Updated** | 2025-01-10 |
| **Merged** | 2025-01-10 |

---

## 📄 Description

Closes [#167](https://github.com/ikawrakow/ik_llama.cpp/issues/167) 

I also saw requests for `Falcon3-10B-1.58b` being made in the mainline `llama.cpp` and `llamafile` repositories, so decided to add the ability to use this model with `ik_llama.cpp`.

1. Get a ternary model in Microsoft's `I2_S` format. E.g., for  ` Falcon3-10B-1.58b`
```
huggingface-cli download tiiuae/Falcon3-10B-Instruct-1.58bit-GGUF
```

2. Re-quantize to one of the ternary quantization types in this repository. E.g.,
```
./bin/llama-quantize --allow-requantize path_to_model/ggml-model-i2_s.gguf output.gguf iq2_bn
```

Works on the CPU **and** GPU (CUDA or Metal)

Enjoy!

I see perplexity is quite high (higher than the Falcon3 7B Instruct ternary model), so not sure how useful this model is in practice.