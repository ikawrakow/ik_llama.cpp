## 📌 [Issue #507](https://github.com/ikawrakow/ik_llama.cpp/issues/507) - Compatible gguf models ?

| **Author** | `lbarasc` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-09 |
| **Updated** | 2025-06-14 |

---

## 📄 Description

Hi,

I want to use some compatible 1bit gguf models, like microsoft bitnet 1b or falcon 1b with your software.
Where can i found these models ? can you send me links to download ?

Thank you for your help.

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-06-09** at **12:23:07**

See [#401](https://github.com/ikawrakow/ik_llama.cpp/issues/401)

---

👤 **lbarasc** commented on **2025-06-09** at **16:47:49**

Here is my command under win10 64bits  (with latest ik_lama with xeon e5 and rtx 3060 cuda : 

D:\ik_lama>llama-server.exe -m ggml-model-i2_s.gguf -p "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n"

the result :

INFO [                    main] build info | tid="21032" timestamp=1749487602 build=1 commit="02272cd"
INFO [                    main] system info | tid="21032" timestamp=1749487602 n_threads=12 n_threads_batch=-1 total_threads=24 system_info="AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "

D:\ik_lama>

I have no error but nothing at all !
Please help me.

---

👤 **ikawrakow** commented on **2025-06-09** at **16:53:40**

You need to convert the `i2_s` model to `ik_llama.cpp` quants as described in [#401](https://github.com/ikawrakow/ik_llama.cpp/issues/401). You missed this step:
```
./build/bin/llama-quantize --allow-requantize ./models/ggml-model-i2_s.gguf ./models/bitnet.gguf iq2_bn_r4
```
Then your server command should use the newly created file, not the `i2_s` file.

---

👤 **lbarasc** commented on **2025-06-09** at **17:09:08**

I do this : 
D:\ik_lama>llama-quantize --allow-requantize ggml-model-i2_s.gguf bitnet.gguf iq2_bn_r4

the result is 
main: build = 1 (02272cd)
main: built with MSVC 19.29.30159.0 for
main: quantizing 'ggml-model-i2_s.gguf' to 'bitnet.gguf' as IQ2_BN_R4

but i cannot retrieve bitnet.gguf file ?

---

👤 **saood06** commented on **2025-06-11** at **07:00:39**

Not sure why the requantize didn't work for you, but I have provided pre-converted models you can use [here](https://huggingface.co/tdh111/bitnet-b1.58-2B-4T-GGUF).

---

👤 **ikawrakow** commented on **2025-06-14** at **12:02:29**

Nothing more that we can do here.