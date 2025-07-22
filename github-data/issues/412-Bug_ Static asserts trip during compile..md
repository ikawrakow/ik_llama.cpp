### 🐛 [#412](https://github.com/ikawrakow/ik_llama.cpp/issues/412) - Bug: Static asserts trip during compile.

| **Author** | `Ph0rk0z` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-12 |
| **Updated** | 2025-05-12 |

---

#### Description

### What happened?

I have some compile time asserts from the recent commits. It built when I commented them out. Have not tested yet to see if there is some issue when running models. I build all fa kernels to have q8/q4 cache when I need it so maybe related?


```
/home/supermicro/ai/ik_llama.cpp/ggml/src/ggml-cuda/fattn-new-mma.cu(859): error: static assertion failed with "bad nbatch_K2, nbatch_V2 for MLA"
      static_assert(!mla || nbatch_K2 >= nbatch_V2, "bad nbatch_K2, nbatch_V2 for MLA");
      ^
          detected during:
            instantiation of "void flash_attn_ext_f16_process_tile<DKQ,DV,ncols1,ncols2,nwarps,ntiles,use_logit_softcap,mla,needs_fixup,is_fixup>(const float2 *, const half2 *, const half2 *, const half2 *, float2 *, float2 *, float, float, float, int, int, int, int, int, int, int, int, int, int) [with DKQ=576, DV=512, ncols1=1, ncols2=16, nwarps=2, ntiles=2, use_logit_softcap=false, mla=true, needs_fixup=false, is_fixup=false]" at line 1331
            instantiation of "void flash_attn_ext_f16<DKQ,DV,ncols1,ncols2,nwarps,ntiles,use_logit_softcap,mla>(const char *, const char *, const char *, const char *, float *, float2 *, float, float, float, float, float, uint32_t, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int) [with DKQ=576, DV=512, ncols1=1, ncols2=16, nwarps=2, ntiles=2, use_logit_softcap=false, mla=true]" at line 1783
            instantiation of "void ggml_cuda_flash_attn_ext_mma_f16_case<DKQ,DV,ncols1,ncols2>(ggml_backend_cuda_context &, ggml_tensor *) [with DKQ=576, DV=512, ncols1=1, ncols2=16]" at line 1821
            instantiation of "void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ,DV,ncols2>(ggml_backend_cuda_context &, ggml_tensor *) [with DKQ=576, DV=512, ncols2=16]" at line 1884

/home/supermicro/ai/ik_llama.cpp/ggml/src/ggml-cuda/fattn-new-mma.cu(475): error: static assertion failed with "bad nbatch_K2, nbatch_V2 for MLA"
      static_assert(!mla || nbatch_K2 >= nbatch_V2, "bad nbatch_K2, nbatch_V2 for MLA");
      ^
          detected during:
            instantiation of "void flash_attn_ext_f16_iter<DKQ,DV,ncols1,ncols2,nwarps,ntiles,use_logit_softcap,mla,needs_fixup,is_fixup,last_iter>(const float2 *, const half2 *, const half2 *, const half2 *, float2 *, float2 *, float, float, float, int, int, int, int, int, int, half2 *, half2 *, half2 *, half2 *, const tile_B *, tile_C_VKQ *, float *, float *, int) [with DKQ=576, DV=512, ncols1=1, ncols2=16, nwarps=2, ntiles=2, use_logit_softcap=false, mla=true, needs_fixup=false, is_fixup=false, last_iter=false]" at line 963
            instantiation of "void flash_attn_ext_f16_process_tile<DKQ,DV,ncols1,ncols2,nwarps,ntiles,use_logit_softcap,mla,needs_fixup,is_fixup>(const float2 *, const half2 *, const half2 *, const half2 *, float2 *, float2 *, float, float, float, int, int, int, int, int, int, int, int, int, int) [with DKQ=576, DV=512, ncols1=1, ncols2=16, nwarps=2, ntiles=2, use_logit_softcap=false, mla=true, needs_fixup=false, is_fixup=false]" at line 1331
            instantiation of "void flash_attn_ext_f16<DKQ,DV,ncols1,ncols2,nwarps,ntiles,use_logit_softcap,mla>(const char *, const char *, const char *, const char *, float *, float2 *, float, float, float, float, float, uint32_t, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int) [with DKQ=576, DV=512, ncols1=1, ncols2=16, nwarps=2, ntiles=2, use_logit_softcap=false, mla=true]" at line 1783
            instantiation of "void ggml_cuda_flash_attn_ext_mma_f16_case<DKQ,DV,ncols1,ncols2>(ggml_backend_cuda_context &, ggml_tensor *) [with DKQ=576, DV=512, ncols1=1, ncols2=16]" at line 1821
            instantiation of "void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ,DV,ncols2>(ggml_backend_cuda_context &, ggml_tensor *) [with DKQ=576, DV=512, ncols2=16]" at line 1884
```

### Name and Version

git

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-12** at **11:41:48**:<br>

What is the architecture?

---

👤 **Ph0rk0z** commented the **2025-05-12** at **11:51:28**:<br>

The system? It's a xeon 5120 w/cuda. I tested qwen 235 with the binary that came out and it worked. Haven't tried deepseek yet.

---

👤 **Ph0rk0z** commented the **2025-05-12** at **11:51:28**:<br>

The system? It's a xeon 5120. I tested qwen 235 with the binary that came out and it worked. Haven't tried deepseek yet.

---

👤 **ikawrakow** commented the **2025-05-12** at **11:53:12**:<br>

I mean the CUDA architecture (Turing, Ampere, etc.). Or simpler, what is the GPU?

---

👤 **ikawrakow** commented the **2025-05-12** at **11:53:12**:<br>

I mean the CUDA architecture (Turing, Ampere, etc.)

---

👤 **Ph0rk0z** commented the **2025-05-12** at **12:03:40**:<br>

I have ampere and turning but only inferencing on ampere. I guess turning gets picked up during compile.

---

👤 **ikawrakow** commented the **2025-05-12** at **12:04:28**:<br>

Does #413 fix it?

---

👤 **Ph0rk0z** commented the **2025-05-12** at **12:08:03**:<br>

Yep, just undid my comments and changed it to CC_TURNING