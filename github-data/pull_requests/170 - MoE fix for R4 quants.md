### 🐛 [#170](https://github.com/ikawrakow/ik_llama.cpp/pull/170) - MoE fix for R4 quants

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-12 |
| **Updated** | 2025-01-12 |

---

#### Description

This PR adds two fixes:
* Make sure number of tensor rows being processed by one thread is a multiple of the number of interleaved rows when using `R4` quants also in `iqk_mul_mat_mow`
* Fix logic when we have a matrix multiplication kernel that processes 16 columns of the right matrix per kernel call (introduced on 907cde6be). The bug shows up when the number of columns in the right matrix is greater than 16 (so this kernel gets used), and the number of columns is not divisible by 16 (so there are leftover columns to be processed), so did not get caught by the usual `TG-128` and `PP-512` testing.

If quantized to `R4` quants, MoE models now work. But if run-time-repacking is used (`-rtr` command line option) to repack non-`R4` quants to `R4`, something goes wrong for MoE models that I'm not able to figure out. It is really bizarre because in the former case (quantize directly into `R4`) four rows are quantized to the corresponding non-`R4` quant in a temporary buffer and then repacked to `R4`. In the later case, 4 rows are copied into a temporary buffer and then repacked, storrng the repacked data into the memory from where the data was copied. The exact same repacking function is used in both cases, so I don't see how `rtr` can fail. What is even more bizarre is that `rtr` always works for non-MoE models, and also works for some quantization types for MoE models.