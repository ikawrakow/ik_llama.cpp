### 🗣️ [#104](https://github.com/ikawrakow/ik_llama.cpp/discussions/104) - Convenience improvements for llama-quantize

| **Author** | `Nexesenex` |
| :--- | :--- |
| **Created** | 2024-10-23 |
| **Updated** | 2024-10-23 |

---

#### Description

Hey IK.

Here are some ideas of potential features for llama-quantize, that I'm not capable to code myself :

- Create a directory when it doesn't exist for the output file.

- Interrupt the quantization (or even **quantize each tensor in a directory**, so the quantization can be resumed on crash, or even a single series of tensor can be requantized (like attn_q weight for example, or even a function of use_more_bits if one of the part of the ternary statement deciding the quantization of a given tensor is not met when you change the quant of a part of the ternary, but not the other). The monolithic approach makes a pretty monster-file, but at the same time, wastes a lot of space, time and compute.

- integrate the formulas like use_more_bits (we have one, I intend to PR more of those) to the tensors that we manually specify with arguments in CLI to customize a FTYPE. 

- A pre-check of the available space on disk before the quantization, ideally coupled with a dry-run giving the final size of the desired quant.