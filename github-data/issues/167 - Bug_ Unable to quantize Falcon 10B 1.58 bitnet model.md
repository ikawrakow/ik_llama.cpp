### 🐛 [#167](https://github.com/ikawrakow/ik_llama.cpp/issues/167) - Bug: Unable to quantize Falcon 10B 1.58 bitnet model

| **Author** | `raymond-infinitecode` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-09 |
| **Updated** | 2025-01-11 |

---

#### Description

### What happened?


Model Source
https://huggingface.co/tiiuae/Falcon3-10B-Instruct-1.58bit/tree/main


llama-quantize ggml-model-f32.gguf output.gguf IQ1_BN

output
main: build = 3525 (3e685162)
main: built with MSVC 19.37.32825.0 for x64
main: quantizing 'd:\llamafile-0.9.0\ggml-model-f32.gguf' to 'output.gguf' as IQ1_BN
ggml_calloc: failed to allocate   0.00 MB
D:\ik_llama.cpp\ggml\src\ggml.c:378: fatal error

### Name and Version

D:\ik_llama.cpp\build\bin\Release>llama-cli --version
version: 3525 (3e685162)
built with MSVC 19.37.32825.0 for x64

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

_No response_

---

#### 💬 Conversation

👤 **raymond-infinitecode** commented the **2025-01-09** at **15:39:01**:<br>

How to convert that model to gguf that can be used with ik_llama.cpp ?

---

👤 **raymond-infinitecode** commented the **2025-01-09** at **15:39:01**:<br>

How to conver that model to gguf that can be used with ik_llama.cpp ?

---

👤 **ikawrakow** commented the **2025-01-09** at **15:48:13**:<br>

I haven't looked into this model at all. Does it work in mainline `llama.cpp`? I see them talking about cloning a Microsoft BitNet repository to use this model, so this does not look like a standard `llama.cpp` GGUF to me.

---

👤 **raymond-infinitecode** commented the **2025-01-10** at **03:02:26**:<br>

Hi Ikawrakow, it doesn't work with llama.cpp but it works with bitnet repository https://github.com/microsoft/BitNet
To be percise it works with 
https://github.com/Eddie-Wang1120/llama.cpp.git  [merge-dev] branch only

---

👤 **ikawrakow** commented the **2025-01-10** at **07:14:34**:<br>

When a ternary Falcon3 model is released in a more standard format, it will be supported also here. In the meantime you can use the quoted Microsoft BitNet repository.

---

👤 **raymond-infinitecode** commented the **2025-01-10** at **11:17:41**:<br>

The problem with Microsoft Bitnet repository is that llama-server is not build. I wonder if they did it on intention.

---

👤 **ikawrakow** commented the **2025-01-10** at **11:34:24**:<br>

And the problem with the model that you want to run is that it is stored quantized as `I2_S`, which is Microsoft BitNet specific, and does not exist anywhere else. There is no `f16` or `f32` or `q8_0` GGUF. If I follow the BitNet setup instructions,  running 
```
python setup_env.py --hf-repo tiiuae/Falcon3-7B-Instruct-1.58bit -q i2_s
```
actually fetches an `f32` version of `Falcon3-7B-Instruct-1.58bit` from `tiiuae/Falcon3-7B-Instruct-1.58bit`. Qunatizing that model to `IQ1_BN` or `IQ2_BN` works just fine. There is a minor modification required in `llama.cpp` to add the Falcon3 pre-tokenizer configuration, and then all works.

But to use the 10B model, which appears to be available only as BitNet `I2_S` quants, one would need to write a `I2_S -> IQ2_BN or IQ1_BN or F16/32` converter. I think it is much easier to ask `tiiuae` to post the model in a standard `llama.cpp` type (`f16, f32, q8_0`) than to write converters from obscure quantization types.

---

👤 **ikawrakow** commented the **2025-01-10** at **15:46:36**:<br>

OK, it doesn't seem to be that hard. WIP on [this branch](https://github.com/ikawrakow/ik_llama.cpp/tree/ik/convert_i2s)

---

👤 **raymond-infinitecode** commented the **2025-01-11** at **05:09:18**:<br>

wow, you are really a genius, complete the conversion implementation in less than half a day !