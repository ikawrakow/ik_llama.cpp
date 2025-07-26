### [Pull Request #610](https://github.com/ikawrakow/ik_llama.cpp/pull/610) - q8_k_r8: experimental AVX512 version

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `ik/q8_k_r8_avx512` |
| **Target Branch** | `main` |
| **Created** | 2025-07-14 |
| **Updated** | 2025-07-23 |

---

#### Description

@ubergarm This is specifically for your 9950X CPU.

On my 7950X this is ~10% slower than what we have on the main branch. The 7950X supports `AVX512`, but 512-bit instructions get executed as two 256-bit instructions. Hence, I'm expecting (hoping?) this `Q8_K_R8` GEMM version to be significantly faster on a CPU with "real" 512-bit instructions such as the 9950X.

Please benchmark it so I can decide if it is worth adding this to the main branch.

---

#### 🔀 Conversation

👤 **ubergarm** commented on **2025-07-14** at **17:27:49**

Wow! :rocket: this little amd 9950x can really rip with its "real" 512-bit instruction!!!

The chart is getting too busy, but left everything to show how crazy it is to see faster PP on my 16x gaming rig that the 24x core thread ripper pro! 😮🎉🥂

*EDIT* the title is a bit misleading, that commit was used for the earlier tests. The actual commit used is shown in the legend in tiny tiny hard to read font. thanks.

<img width="4176" height="2217" alt="sweep-bench-activations-speed-testing-pr610" src="https://github.com/user-attachments/assets/220a4c85-e563-407a-9bf7-708b72bb6a78" />

<details>

<summary>👈 Details</summary>

The other data and info is over on #602 

# Q8_K_R8 9950X 16x PR610 ik/q8_k_r8_avx512@c462c5bd
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    1.593 |   321.47 |   21.403 |     5.98 |
|   512 |    128 |    512 |    1.658 |   308.80 |   21.594 |     5.93 |
|   512 |    128 |   1024 |    1.719 |   297.82 |   21.855 |     5.86 |
|   512 |    128 |   1536 |    1.797 |   284.99 |   22.093 |     5.79 |
|   512 |    128 |   2048 |    1.866 |   274.35 |   22.337 |     5.73 |
|   512 |    128 |   2560 |    1.948 |   262.82 |   22.605 |     5.66 |
|   512 |    128 |   3072 |    2.008 |   254.93 |   22.899 |     5.59 |
|   512 |    128 |   3584 |    2.084 |   245.66 |   23.271 |     5.50 |
|   512 |    128 |   4096 |    2.152 |   237.93 |   23.333 |     5.49 |

</details>

---

👤 **ikawrakow** commented on **2025-07-14** at **17:42:43**

OK, then, I'll create a way to select one of the two kernels at build time.

Yes, the 9950X is really nice. I was tempted to upgrade when it came out, but at the end didn't because AMD didn't do anything for memory bandwidth.

---

👤 **Ph0rk0z** commented on **2025-07-18** at **13:11:56**

Wish there was a way to use AVX-512 without the ML extensions. Or would it not provide any benefit over AVX2?

---

👤 **sousekd** commented on **2025-07-22** at **20:58:13**

Do I need specific model quants to test it? I tried using **anikifoss/Kimi-K2-Instruct-DQ4_K** and **bartowski/Qwen3-235-A22B-Q8_0** with `-rtr`, but I didn't notice any difference compared to the main branch on my EPYC 9355. It might be due to how I compiled it on Windows, though.

---

👤 **ubergarm** commented on **2025-07-23** at **01:03:05**

@sousekd 

> Do I need specific model quants to test it?

If I understand correctly this only effects quants that use q8_k_r8 path so I don't think your Q8_0 would be effected nor your q4_K/q6_K quants which use different paths [as i tried to find a way to describe here in an older buried comment](https://github.com/ikawrakow/ik_llama.cpp/pull/495#issuecomment-2985633815).

I think this would be a list of the current quants that if are in your mix you might see a boost in PP using this PR on a Zen5 CPU:

<details>

<summary>👈 supported quants</summary>

```bash
$ grep Q8_K_R8 ggml/src/iqk/iqk_mul_mat.cpp  | grep type
            case GGML_TYPE_IQ2_XXS: return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_XS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_S  : return nrc_y >= 16 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_XXS: return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_XS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_S  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ1_S  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ1_M  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_Q2_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_Q3_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_KL : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ5_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ5_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ6_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_Q2_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_Q3_K   : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ1_S  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ1_M  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_XXS: return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_XS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_S  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_XXS: return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_S  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_XS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_KL : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ5_KS : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ2_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ3_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ4_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ5_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
            case GGML_TYPE_IQ6_K  : return nrc_y >= 32 ? GGML_TYPE_Q8_K_R8 : type;
```

</details>

I'm not sure how `-rtr` would effect it or not, I'd suggest leave it off if you are testing and just attempt to boost `-ub 4096 -b 4096` for max PP as is my practice.

psure your CPU should support it though as it is Zen5, and i have no idea about windows compiling. on linux i run `lscpu | grep avx_vnni` to check for the flag in question.

You could possibly give this quant a try as it is mostly quants from this list: https://huggingface.co/ubergarm/Qwen3-235B-A22B-Instruct-2507-GGUF#iq5_k-161722-gib-5909-bpw

I measured slightly better PPL than the larger DQ4_K, but I used an imatrix so go with whatever you prefer.

---

👤 **ikawrakow** commented on **2025-07-23** at **07:31:38**

Yes, pick one of the quantization types in the list provided by @ubergarm to see if it makes a difference on your Zen5 CPU.

> I'm not sure how -rtr would effect it or not

Do not use `-rtr`. With `-rtr` it will repack the quants to the corresponding row-interleaved `*_R4` or `*_R8` variant while loading the model. The row-interleaved quants do not get repacked to `Q8_K_R8` for large matrix multiplications, so the PR will have no effect on performance in that case.