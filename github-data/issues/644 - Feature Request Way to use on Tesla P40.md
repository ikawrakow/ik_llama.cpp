## 📌 [Issue #644](https://github.com/ikawrakow/ik_llama.cpp/issues/644) - Feature Request: Way to use on Tesla P40

| **Author** | `narikm` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-24 |
| **Updated** | 2025-07-27 |
| **Labels** | `enhancement` |

---

## 📄 Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Is there a way to use this code on legacy Tesla P40?

### Motivation

Is there a way to use this repo on a old Tesla P40? I tried to deactivate flash attention and use:
cmake -B build ^
  -DGGML_CUDA=ON ^
  -DGGML_BLAS=OFF ^
  -DGGML_CUDA_ARCH=61 ^
  -DGGML_CUDA_GRAPH=OFF ^
  -DGGML_CUDA_FORCE_MMQ=OFF ^
  -DGGML_CUDA_DMMV_X=32 ^
  -DGGML_CUDA_MMQ_ENABLE=OFF ^

according to Chat GPT, is there a way to compile it for old devices?

I only get CUDA errors like:
CUDA error: an illegal memory access was encountered
  current device: 0, in function launch_mul_mat_q at D:\ik_llama.cpp\ggml\src\ggml-cuda\template-instances\../mmq.cuh:4008
  cudaFuncSetAttribute(mul_mat_q<type, mmq_x, 8, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem)
D:\ik_llama.cpp\ggml\src\ggml-cuda.cu:110: CUDA error

### Possible Implementation

_No response_

---

## 💬 Conversation

👤 **firecoperana** commented on **2025-07-25** at **11:59:14**

P40 should work. Try this command:
-DGGML_CUDA=ON ^
-DGGML_BLAS=OFF ^
-DCMAKE_CUDA_ARCHITECTURES="61" ^
-DGGML_CUDA_USE_GRAPHS=OFF ^
-DGGML_CUDA_FORCE_MMQ=OFF ^
-DGGML_CUDA_DMMV_X=32 ^
-DGGML_CUDA_MMQ_ENABLE=OFF 

Flash attention should work too with -DGGML_CUDA_FA_ALL_QUANTS=ON

---

👤 **narikm** commented on **2025-07-25** at **14:35:54**

Same error.
INFO [            update_slots] kv cache rm [p0, end) | tid="5232" timestamp=1753453942 id_slot=0 id_task=0 p0=0
CUDA error: an illegal memory access was encountered
  current device: 0, in function launch_mul_mat_q at G:\ik_llama.cpp\ggml\src\ggml-cuda\template-instances\../mmq.cuh:4008
  cudaFuncSetAttribute(mul_mat_q<type, mmq_x, 8, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem)
G:\ik_llama.cpp\ggml\src\ggml-cuda.cu:110: CUDA error

The compiler said -DGGML_CUDA_MMQ_ENABLE=OFF was not used. Is this related?

---

👤 **firecoperana** commented on **2025-07-25** at **15:12:29**

Can you post your full command line including the model name?

---

👤 **narikm** commented on **2025-07-25** at **15:49:47**

The launch command line:
cd /d G:\ik_llama.cpp\build\bin\Release

llama-server ^
    --alias DeepSeek-R1-0528-IQ2_K_R4 ^
    --model "G:\DeepSeek-R1-0528-IQ2_K_R4\DeepSeek-R1-0528-IQ2_K_R4-00001-of-00005.gguf" ^
    -rtr ^
    --ctx-size 12288 ^
    -ctk q8_0 ^
    -mla 2 ^
    -amb 512 ^
    -fmoe ^
    --n-gpu-layers 63 ^
    --override-tensor exps=CPU ^
    --parallel 1 ^
    --threads 32 ^
    --host 0.0.0.0 ^
    --port 8008 


The build line is what you gave me:
cmake -B build ^
-DGGML_CUDA=ON ^
-DGGML_BLAS=OFF ^
-DCMAKE_CUDA_ARCHITECTURES="61" ^
-DGGML_CUDA_USE_GRAPHS=OFF ^
-DGGML_CUDA_FORCE_MMQ=OFF ^
-DGGML_CUDA_DMMV_X=32 ^
-DGGML_CUDA_MMQ_ENABLE=OFF

---

👤 **firecoperana** commented on **2025-07-25** at **16:40:52**

You are using the ik quants file. Add -DGGML_IQK_FA_ALL_QUANTS=1 to see if it works. You can also use regular quants here.

---

👤 **narikm** commented on **2025-07-25** at **16:48:36**

Will do. Do i need -fa in the launch ? The compiler said:

Manually-specified variables were not used by the project:

    GGML_CUDA_MMQ_ENABLE.

Is this OK?

---

👤 **firecoperana** commented on **2025-07-25** at **17:11:32**

-fa is supported, but since you are not using ctv, it's not required I think. I would remove GGML_CUDA_MMQ_ENABLE since it's off by default. If you still have issue, you can remove -ctk q8_0.

---

👤 **firecoperana** commented on **2025-07-25** at **17:17:24**

You can add -DGGML_CUDA_FA_ALL_QUANTS=ON to make sure all quants are supported.

---

👤 **ikawrakow** commented on **2025-07-25** at **17:42:44**

DeepSeek flash attention does not work on a P40.

---

👤 **narikm** commented on **2025-07-25** at **17:55:14**

ChatGPT said the same, so i removed the -fa in the launch args before posting. It changed the error to the one i posted. I still get the same error each time. It load the model but when i ask something, error, the cpu run for ten seconds, then the program crash.

---

👤 **firecoperana** commented on **2025-07-25** at **18:54:27**

It most likely not gonna work, but if you like to try, can you remove line 4008-4009 or 4008-4010 in mmq.cuh, and run it without fa?

---

👤 **Ph0rk0z** commented on **2025-07-25** at **20:33:58**

>DeepSeek flash attention does not work on a P40.

Is this because of having to use BF16? Or something else like not enough smem?

>-DGGML_CUDA_FA_ALL_QUANTS=ON

I think this is just to use different K and V quantization. Like Q4/Q8.

This thread saved me from re-installing the P40s I have at least. Any other caveats, say for qwen coder or others? Assume P100 is the same story too. What about turning cards?

---

👤 **firecoperana** commented on **2025-07-25** at **21:34:39**

> ChatGPT said the same, so i removed the -fa in the launch args before posting. It changed the error to the one i posted. I still get the same error each time. It load the model but when i ask something, error, the cpu run for ten seconds, then the program crash.

Does your card work in llama.cpp if you use regular quants of Deepseek R1?

---

👤 **narikm** commented on **2025-07-25** at **21:37:54**

I only tried with webui and a 7b llama3, where it works. I want to use the better IK quant, for faster inference AFAIK.

---

👤 **narikm** commented on **2025-07-25** at **23:26:49**

> It most likely not gonna work, but if you like to try, can you remove line 4008-4009 or 4008-4010 in mmq.cuh, and run it without fa?

Another error:
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_cuda_op_mul_mat at G:\ik_llama.cpp\ggml\src\ggml-cuda.cu:1733
  ggml_cuda_cpy_tensor_2d( src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream)
G:\ik_llama.cpp\ggml\src\ggml-cuda.cu:110: CUDA error

---

👤 **firecoperana** commented on **2025-07-26** at **00:55:28**

I haven't used IQ2_K_R4 quants, but I can use my 1080ti for DeepSeek V3 UD IQ1_S with -rtr -fa -fmoe -mla 1 without the above code change in ik_llama.cpp.

---

👤 **narikm** commented on **2025-07-26** at **01:11:03**

What args did you use to compile? Can you give me your launchs commands so i can try to repicate?

---

👤 **firecoperana** commented on **2025-07-26** at **01:36:39**

cmake.exe -B build -DGGML_CUDA=ON  -DGGML_BLAS=OFF -DCMAKE_CUDA_ARCHITECTURES="86;61" -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON  -DGGML_RPC=ON  -DLLAMA_CURL=OFF  -DBUILD_SHARED_LIBS=ON -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_IQK_FA_ALL_QUANTS=0


llama-server.exe ^
        --model "F:\DeepSeek-V3 UD IQ1_S\DeepSeek V3 UD IQ1_S.gguf"  ^
        --host 292.138.3.201 ^
        --port 6703 ^
        --n-gpu-layers 99 ^
        --tensor-split 99 ^
        --split-mode layer --main-gpu 0 ^
        --threads 10  --ctx-size 100 --cache-type-k q8_0  --seed 1234 -ot exps=CPU  --ubatch-size 16 --batch-size 16 ^
        -rtr -fa -fmoe -mla 1

---

👤 **narikm** commented on **2025-07-26** at **23:34:24**

Ok, so i tested with your your args and model: it crash with -fa. Without it doesn't but neither does it output tokens, despite the cpu and gpu working. So there is no current way to use a P40 with ik llama.

---

👤 **saood06** commented on **2025-07-26** at **23:37:51**

>Without it doesn't but neither does it output tokens, despite the cpu and gpu working.

Are you using the text_completion endpoint? If so that may be the reason see [#654](https://github.com/ikawrakow/ik_llama.cpp/issues/654).

---

👤 **narikm** commented on **2025-07-26** at **23:42:30**

Yes, on silly tavern, will test again once fix is merged. But it still crash with Ik quant like the Ubergarm deepseek.

---

👤 **firecoperana** commented on **2025-07-26** at **23:44:34**

You can test from the built-in webui first. This is still working. Just type ip and port in the browser. If you didn't set it, default is 127.0.0.1:8080.

---

👤 **narikm** commented on **2025-07-26** at **23:56:13**

I tested, it works locally with the standard quant (unsloth/DeepSeek-V3-0324-GGUF) but crash with the ik quant [ubergarm/DeepSeek-V3-0324-GGUF] (https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF). Is there a way to use the better ik quant?

---

👤 **saood06** commented on **2025-07-27** at **00:03:29**

>I tested, it works locally with the standard quant (unsloth/DeepSeek-V3-0324-GGUF) but crash with the ik quant [ubergarm/DeepSeek-V3-0324-GGUF] (https://huggingface.co/ubergarm/DeepSeek-V3-0324-GGUF). Is there a way to use the better ik quant?

Can you give more details about this crash?

---

👤 **narikm** commented on **2025-07-27** at **00:05:42**

The same as always, CUDA error: an illegal memory access was encountered
current device: 0, in function launch_mul_mat_q at D:\ik_llama.cpp\ggml\src\ggml-cuda\template-instances../mmq.cuh:4008
cudaFuncSetAttribute(mul_mat_q<type, mmq_x, 8, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem)
D:\ik_llama.cpp\ggml\src\ggml-cuda.cu:110: CUDA error
Chatgpt wanted me to desactivate mmq to fall back on older matrix multiplication, but the args he gave me were not taken into account.

---

👤 **saood06** commented on **2025-07-27** at **00:39:53**

>Yes, on silly tavern, will test again once fix is merged. But it still crash with Ik quant like the Ubergarm deepseek.

I know you already tested, but just as a heads up, I merged the fix in.