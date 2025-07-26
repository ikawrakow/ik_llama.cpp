### [Pull Request #483](https://github.com/ikawrakow/ik_llama.cpp/pull/483) -  convert_hf_to_gguf.py : conversion from hf weights to Q6_0

| **Author** | `Nexesenex` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `conv_q6_0` |
| **Target Branch** | `main` |
| **Created** | 2025-06-02 |
| **Updated** | 2025-06-03 |
| **Merged** | 2025-06-03 |

---

#### Description

This quantization script is obtained by making a sort of "cross multiplication" with the python code for q5_0, and the C code for q5_0 and q6_0 in order to get through trial and error the code for the q6_0 conversion script, this with the help of a 7xB parameters AI model.

It was an interesting experiment!

Tested on Llama 3.2 instruct 1B and Qwen 2.5 instruct 1.5B.
Bitrate of this q6_0 conversion is 6.50BPW straight.
PPL equivalent (+/-0.5%) to a regular q6_0 quant from a fp16 gguf.
Inference is working as intended in my Croco.cpp.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [x] Medium
  - [ ] High