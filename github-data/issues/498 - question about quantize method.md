## 📌 [Issue #498](https://github.com/ikawrakow/ik_llama.cpp/issues/498) - question: about quantize method

| **Author** | `nigelzzz` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-06 |
| **Updated** | 2025-06-14 |

---

## 📄 Description

Hi,
  the project is amazing and interesting, looks like it better thank origin llama.cpp.

I would like to study the repo and because there are a lot of quantize method same with origin llama.cpp, can i know how to choose quantize method to study.

my env is rpi5, and i often test bitnet and llama3.2 1b or 3b. 

thanks

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-06-06** at **16:39:00**

For BitNet take a look at `IQ1_BN` and `IQ2_BN`. The packing in `IQ2_BN` is simpler and easier to understand, but uses 2 bits per weight. `IQ1_BN` uses 1.625 bits per weight, which is very close to the theoretical 1.58 bits for a ternary data type. 

Otherwise not sure what to recommend. Any of the quantization types should be OK for LlaMA-3.1-1B/3B on Rpi5. If you are new to the subject, it might be better to look into the simpler quantization types (e.g., `QX_K`) first.

---

👤 **aezendc** commented on **2025-06-09** at **10:48:49**

> For BitNet take a look at `IQ1_BN` and `IQ2_BN`. The packing in `IQ2_BN` is simpler and easier to understand, but uses 2 bits per weight. `IQ1_BN` uses 1.625 bits per weight, which is very close to the theoretical 1.58 bits for a ternary data type.
> 
> Otherwise not sure what to recommend. Any of the quantization types should be OK for LlaMA-3.1-1B/3B on Rpi5. If you are new to the subject, it might be better to look into the simpler quantization types (e.g., `QX_K`) first.

I like the iq1_bn quantize. Its good and I am using it. Is there a way we can make this support function calling?

---

👤 **ikawrakow** commented on **2025-06-09** at **11:01:33**

See [#407](https://github.com/ikawrakow/ik_llama.cpp/issues/407)

---

👤 **ikawrakow** commented on **2025-06-14** at **12:01:58**

I think we can close it.