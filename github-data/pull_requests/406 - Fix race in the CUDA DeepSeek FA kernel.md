## 🔀 [Pull Request #406](https://github.com/ikawrakow/ik_llama.cpp/pull/406) - Fix race in the CUDA DeepSeek FA kernel

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_cuda_fa_race` |
| **Target Branch** | `main` |
| **Created** | 2025-05-11 |
| **Updated** | 2025-05-13 |
| **Merged** | 2025-05-11 |

---

## 📄 Description

Reference: https://github.com/ggml-org/llama.cpp/pull/13438

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-05-12** at **15:59:39**

Just saw what looks like a small patch in mainline's [earlier ggml-org/llama.cpp[#13438](https://github.com/ikawrakow/ik_llama.cpp/issues/13438) just updated in [#13469](https://github.com/ikawrakow/ik_llama.cpp/issues/13469) (linked here)](https://github.com/ggml-org/llama.cpp/pull/13469)

Could be related to my issue with `DDDD` showing up for longer contexts which I attributed to `-ser` [as we were discussing here](https://github.com/ikawrakow/ik_llama.cpp/pull/386#issuecomment-2869078136)?

Though hrmm, yours has this in a similar area already, so may not be relevent.
```
      if (np > 1) {
          __syncthreads();
      }
```

fwiw I tested the following small change and still am seeing `DDDD` with longer context and `-ser` so might not be related.

```
--- a/ggml/src/ggml-cuda/fattn-mma-f16.cuh
+++ b/ggml/src/ggml-cuda/fattn-mma-f16.cuh
@@ -734,9 +734,10 @@ static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
             float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
             dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
         }
-    }
-
-    if (np > 1) {
+    } else if (np > 1) {
+        // Warps with threadIdx.y % np == 0 execute a __syncthreads() in the if branch.
+        // Therefore, all other warps also need to execute a __syncthreads().
+        // Otherwise the points at which warps synchronize with each other would become misaligned.
         __syncthreads();
     }
```

---

👤 **ikawrakow** commented on **2025-05-13** at **04:34:01**

> Could be related to my issue with DDDD showing up for longer contexts which I attributed to -ser https://github.com/ikawrakow/ik_llama.cpp/pull/386#issuecomment-2869078136?

Thanks for the alert.  But isn't it easier to rerun without `-ser` to not have 2 potential causes at the same time? There has been [a new report](https://github.com/ikawrakow/ik_llama.cpp/discussions/385#discussioncomment-13125043) about SER not working, this time CPU only.