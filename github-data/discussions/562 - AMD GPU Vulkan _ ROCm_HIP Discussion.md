### 🗣️ [#562](https://github.com/ikawrakow/ik_llama.cpp/discussions/562) - AMD GPU Vulkan & ROCm/HIP Discussion

| **Author** | `ubergarm` |
| :--- | :--- |
| **Created** | 2025-06-28 |
| **Updated** | 2025-07-06 |

---

#### Description

## Background
I've been asked a few times now about AMD GPU support with ik's fork. I recently got access to an AMD RX 7900 XTX to try it out, and as discussed on [Issue 503](https://github.com/ikawrakow/ik_llama.cpp/issues/503#issuecomment-2953557243) the Vulkan and ROCm backends are *not* the focus of this fork hence limited support on AMD GPU hardware.

I'm starting this discussion to have a place to point folks who might be interested the current state AMD GPU backend support, and especially if they wanted to attempt updates and work on it at all.

## Current State
ik_llama.cpp actually *does* compile with Vulkan and can do some limited inferencing. As it is unmaintained, it is slower than mainline at the moment. However I couldn't get it to compile with ROCm/HIP support. I only tried the AMD official open source AMDVLK backend and not the community open source RADV backend.

There is a [good benchmarking discussion on mainline](https://github.com/ggml-org/llama.cpp/discussions/10879#discussioncomment-13606581) maintained by @netrunnereve which was very helpful for establishing baseline expectations and trying to understand the various AMD GPU driver development environments.

## Benchmarks
I did a comparison between mainline llama.cpp and ik_llama.cpp at the given sha's for what I could get working.

![sweep-bench-amd-gpu-mainline-vs-ik](https://github.com/user-attachments/assets/9a9c2fcc-24db-46bb-8131-9c47fce36084)

## Methodology
To keep things somewhat consistent with the establish methodologies I used [TheBloke's now vintage Llama-2-7B at classic Q4_0 quantization](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_0.gguf). The following is how compilation was done as well as running `llama-sweep-bench` with and without flash attention `-fa`:

### Compiling
```bash
# compile for Vulkan
cmake -B build -DGGML_HIP=OFF -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)

# couldn't find a combination that worked below
# compile for ROCm/HIP
export HIPCXX="$(hipconfig -l)/clang"
export HIP_PATH="$(hipconfig -R)"
#cmake -B build -DGGML_VULKAN=0 -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 -DGGML_HIP_ROCWMMA_FATTN=ON -DCMAKE_BUILD_TYPE=Release
cmake -B build -DGGML_VULKAN=0 -DGGML_HIPBLAS=ON -DAMDGPU_TARGETS=gfx1100 -DGGML_HIP_ROCWMMA_FATTN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
In file included from /home/w/projects/ik_llama.cpp/ggml/src/ggml-cuda/fattn.cu:15:
In file included from /home/w/projects/ik_llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:3:
/home/w/projects/ik_llama.cpp/ggml/src/ggml-cuda/mma_new.cuh:49:27: error: use of undeclared identifier '__shfl_sync'
   49 |     const int ret_low  = (__shfl_sync(0xFFFFFFFF, x, src_laneid_low,  WARP_SIZE) >> shift_low)  & 0x0000FFFF;
      |                           ^
/home/w/projects/ik_llama.cpp/ggml/src/ggml-cuda/mma_new.cuh:50:27: error: use of undeclared identifier '__shfl_sync'
   50 |     const int ret_high = (__shfl_sync(0xFFFFFFFF, x, src_laneid_high, WARP_SIZE) << shift_high) & 0xFFFF0000;
      |                           ^
4 errors generated when compiling for gfx1100.
```

#### sweep-bench
```bash
export model=/models/TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_0.gguf
# try with and without -fa
./build/bin/llama-sweep-bench \
  --model "$model" \
  -fa \
  -c 18432 \
  -ngl 99 \
  --warmup-batch \
  --threads 1
```

### Observations
1. Surprisingly Vulkan without FA managed to complete the benchmark and even give similar performance as mainline for the no FA token generation at longer context lengths.
2. However, Vulkan with FA enabled shows very poor performance and consistently crashes at `N_KV=7680`.  `iqk_fa_templates.h:1146: GGML_ASSERT(fms.S[j] > 0) failed`
3. I did *not* test any other quantizations especially the newer ik exclusive quants.
4. I did do a quick vibe check and confirm the model was at least valid tokens, however the chat template seemed odd or could be due to my client settings for temp etc but the responses seemed wrong and had `<|im_start|>` and `<|im_end|>` type tokens which don't usually come back from the chat endpoint.

## Conclusion
Well, sorry if you have AMD GPU hardware and were hoping to try out the latest greatest stuff on ik's fork. You can still make use of the CPU only optimizations fwiw. You can see the relative performance of native CUDA in the linked benchmark thread for one of my other tests, and ik's fork does run faster than mainline for CUDA.

Finally, I saw [and interesting NVIDIA slide deck from the Vulkanised 2025 Developer Conference](https://vulkan.org/user/pages/09.events/vulkanised-2025/T47-Jeff-Bolz-NVIDIA.pdf) which discusses llama.cpp on pages 14 and 15 even showing what looks like [some of ik's IQ4_NL code](https://github.com/ggml-org/llama.cpp/pull/5590) with implementation discussions. I was surprised that some models benchmark faster on NVIDIA GPUs using vulkan backend beating out the native CUDA implementation, but perhaps that is for another day...

Thanks and curious if anyone else has tried this or is interested in improving support here. Cheers!

---

#### 🗣️ Discussion

👤 **OneOfOne** replied the **2025-06-29** at **01:50:14**:<br>

llama.cpp's vulkan backend is faster and uses less memory on my 7900xtx as well (I'm using latest rocm on Arch so it's not that).

> 👤 **ubergarm** replied the **2025-06-29** at **14:41:47**:<br>
> Yup, this is to be expected given ik's fork prioritizes a couple CPU types and CUDA implementations and does not focus on maintaining Vulkan nor ROCm/HIP backends.

---

👤 **firecoperana** replied the **2025-06-29** at **14:50:07**:<br>

I'm working on bringing ik_llama.cpp up to date with llama.cpp's vulkan backend. It is actually easier than I expected.

> 👤 **ubergarm** replied the **2025-06-29** at **14:58:06**:<br>
> @firecoperana very cool to hear :fire: !
> 
> As suggested by @0cc4m and some discussion by the author of those Vulkanised Conference PDF slides linked above, @jeffbolznv ,over on the [mainline vulkan benchmark discussion](https://github.com/ggml-org/llama.cpp/discussions/10879#discussioncomment-13606581) I might try to `pacman -Sy extra/nvidia-utils` and build the vulkan backend for my NVIDIA RTX 3090TI FE GPU and compare performance there as well.
> 
> Please update us here if you have a fork/branch/PR you'd like to test and if I still have access to the AMD RX 7900 XTX I can give it a go as I'd like to use ik's SOTA quants on that machine for a fun project...
> 
> 👤 **ikawrakow** replied the **2025-06-29** at **16:25:52**:<br>
> @firecoperana Great that you want to port the mainline Vulkan back-end to `ik_llama.cpp`, but are you also willing to maintain it?
> 
> 👤 **firecoperana** replied the **2025-06-29** at **19:30:13**:<br>
> PR is created. Welcome to test. I can maintain it if the vulkan code there hasn't been refactored too much. With this PR, the future update should be easier too. I don't use vulkan much so need someone to remind me if there is some major improvement in vulkan that is worth porting.
> 
> 👤 **ubergarm** replied the **2025-06-29** at **19:41:02**:<br>
> I'll give it a try, I just updated my home rig to latest greatest drivers (which I loathe to do but sometimes u gotta pay the piper...).
> 
> Interestingly on a `Qwen3-14B-Q4_0` the Vulkan FA=1 backend beats native CUDA implementation in token generation at sufficiently deep n_kv
> 
> https://github.com/ggml-org/llama.cpp/discussions/10879#discussioncomment-13611122
> 
> I'll take a look at the PR now, thanks! https://github.com/ikawrakow/ik_llama.cpp/pull/563
> 
> 👤 **firecoperana** replied the **2025-06-29** at **19:53:41**:<br>
> https://github.com/ggml-org/llama.cpp/pull/14366
> Vulkan also needs this one, but I couldn't port it in easily. The issue is vulkan does not have FUSED_RMS_NORM and FUSED_MUL_UNARY support, and when using RPC, it needs this. My current workaround is skip ggml_fused_rms_norm and ggml_fused_mul_unary when using vulkan.  @ikawrakow

---

👤 **ikawrakow** replied the **2025-07-01** at **13:50:50**:<br>

So, what is the "approved" way of installing the necessary dependencies for Vulkan development on Ubuntu? I ended up installing LunarG VulkanSDK, but the thing almost bricked my system because I hadn't run `sudo apt update && sudo apt upgrade` before importing their repository and attempting to install. Is there a better way, preferably with just Ubuntu packages and no 3rd party stuff?

Anyhow, at the end I got the mainline Vulkan build working, but performance is very far from CUDA on my RTX-4080

### Vulkan sweep-bench, LlaMA-3.1-8B

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    0.339 |  3024.00 |    2.808 |    91.16 |
|  1024 |    256 |   1024 |    0.337 |  3035.97 |    2.709 |    94.51 |
|  1024 |    256 |   2048 |    0.328 |  3121.27 |    2.657 |    96.36 |
|  1024 |    256 |   3072 |    0.336 |  3052.01 |    2.661 |    96.19 |
|  1024 |    256 |   4096 |    0.368 |  2781.06 |    2.704 |    94.67 |
|  1024 |    256 |   5120 |    0.405 |  2531.44 |    2.794 |    91.61 |
|  1024 |    256 |   6144 |    0.465 |  2202.62 |    2.917 |    87.75 |
|  1024 |    256 |   7168 |    0.542 |  1888.01 |    3.047 |    84.00 |
|  1024 |    256 |   8192 |    0.618 |  1656.82 |    3.196 |    80.10 |
|  1024 |    256 |   9216 |    0.657 |  1559.24 |    3.283 |    77.98 |
|  1024 |    256 |  10240 |    0.695 |  1473.46 |    3.365 |    76.08 |
|  1024 |    256 |  11264 |    0.720 |  1422.92 |    3.412 |    75.02 |
|  1024 |    256 |  12288 |    0.753 |  1359.30 |    3.464 |    73.89 |
|  1024 |    256 |  13312 |    0.792 |  1293.13 |    3.523 |    72.67 |
|  1024 |    256 |  14336 |    0.814 |  1257.77 |    3.588 |    71.35 |
|  1024 |    256 |  15360 |    0.858 |  1192.89 |    3.625 |    70.63 |

###  CUDA sweep-bench, LlaMA-3.1-8B

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  1024 |    256 |      0 |    0.134 |  7649.04 |    2.018 |   126.88 |
|  1024 |    256 |   1024 |    0.129 |  7921.34 |    2.105 |   121.63 |
|  1024 |    256 |   2048 |    0.135 |  7561.83 |    2.170 |   117.99 |
|  1024 |    256 |   3072 |    0.144 |  7121.15 |    2.226 |   114.99 |
|  1024 |    256 |   4096 |    0.151 |  6784.15 |    2.292 |   111.71 |
|  1024 |    256 |   5120 |    0.159 |  6460.57 |    2.354 |   108.75 |
|  1024 |    256 |   6144 |    0.164 |  6225.61 |    2.423 |   105.66 |
|  1024 |    256 |   7168 |    0.172 |  5961.15 |    2.484 |   103.05 |
|  1024 |    256 |   8192 |    0.183 |  5606.81 |    2.545 |   100.61 |
|  1024 |    256 |   9216 |    0.194 |  5289.56 |    2.604 |    98.31 |
|  1024 |    256 |  10240 |    0.195 |  5239.75 |    2.662 |    96.15 |
|  1024 |    256 |  11264 |    0.206 |  4962.13 |    2.731 |    93.72 |
|  1024 |    256 |  12288 |    0.214 |  4777.95 |    2.787 |    91.85 |
|  1024 |    256 |  13312 |    0.217 |  4725.71 |    2.845 |    89.97 |
|  1024 |    256 |  14336 |    0.230 |  4454.44 |    2.919 |    87.71 |
|  1024 |    256 |  15360 |    0.238 |  4311.56 |    2.966 |    86.30 |

So, PP is 3X lower, TG is 20-25% lower.

Given this, does it make sense to spend time on Vulkan? When I forked `llama.cpp` last year the Vulkan stuff was mostly a gimmick, with performance not much better than just running on a moderately fast CPU. They have done a lot of Vulkan development and performance improvements in mainline since then, but it still seems way too far behind.

> 👤 **jeffbolznv** replied the **2025-07-01** at **14:08:19**:<br>
> Installing the Vulkan SDK is the "right" way to get the dependencies. The pp scores shouldn't be that low, it suggests cooperative matrix isn't getting used. What driver version are you using? Can you share the beginning of the log where ggml-vulkan prints device info?
> 
> 👤 **ubergarm** replied the **2025-07-01** at **19:20:24**:<br>
> > Given this, does it make sense to spend time on Vulkan?
> 
> Personally, the two things I see Vulkan back-end support providing are:
> 1. A path allowing AMD GPUs to be used e.g. RX 7900 XTX 24GB VRAM
> 2. Potentially faster NVIDIA path for some situations/models (this was news to me).
> 
> This Qwen3-14B-Q4_0 dense sweep-bench I ran a couple days ago opened my eyes where the vulkan backend on mainline took the lead on TG after about ~8k depth. `NV_coopmat2` [is described in @jeffbolznv recent Vulkanised 2025 slides](https://vulkan.org/user/pages/09.events/vulkanised-2025/T47-Jeff-Bolz-NVIDIA.pdf). 
> 
> ![sweep-bench-llama-vs-ik-vulkan-qwen3-14b](https://github.com/user-attachments/assets/bc0d855e-5640-45df-bbb0-82e4d048c49c)
> 
> Otherwise ik CUDA is generally the fastest. I haven't tested other models/configs but likely vulkan takes the lead in other situations reading the benchmarks in the slides.
> 
> However, I also don't want to distract ik whatever optimizations and experiments are most interesting and intrinsically motivating. So nice to see a few folks from the community possibly providing some support. Big thanks @firecoperana for taking a stab at it on https://github.com/ikawrakow/ik_llama.cpp/pull/563
> 
> Thanks!

---

👤 **ikawrakow** replied the **2025-07-01** at **14:11:22**:<br>

```code
ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = NVIDIA GeForce RTX 4080 (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat
build: 5781 (ba3ef86c5) with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
llama_model_load_from_file_impl: using device Vulkan0 (NVIDIA GeForce RTX 4080) - 16376 MiB free
llama_model_loader: loaded meta data with 29 key-value pairs and 292 tensors from ../ncuda/junk.bin (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Meta Llama 3.1 8B Instruct
llama_model_loader: - kv   3:                           general.finetune str              = Instruct
llama_model_loader: - kv   4:                           general.basename str              = Meta-Llama-3.1
llama_model_loader: - kv   5:                         general.size_label str              = 8B
llama_model_loader: - kv   6:                            general.license str              = llama3.1
llama_model_loader: - kv   7:                               general.tags arr[str,6]       = ["facebook", "meta", "pytorch", "llam...
llama_model_loader: - kv   8:                          general.languages arr[str,8]       = ["en", "de", "fr", "it", "pt", "hi", ...
llama_model_loader: - kv   9:                          llama.block_count u32              = 32
llama_model_loader: - kv  10:                       llama.context_length u32              = 131072
llama_model_loader: - kv  11:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv  12:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv  13:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv  14:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv  15:                       llama.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv  16:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  17:                           llama.vocab_size u32              = 128256
llama_model_loader: - kv  18:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  19:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  20:                         tokenizer.ggml.pre str              = llama-bpe
llama_model_loader: - kv  21:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  22:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  23:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  24:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 128009
llama_model_loader: - kv  26:                    tokenizer.chat_template str              = {{- bos_token }}\n{%- if custom_tools ...
llama_model_loader: - kv  27:               general.quantization_version u32              = 2
llama_model_loader: - kv  28:                          general.file_type u32              = 2
llama_model_loader: - type  f32:   66 tensors
llama_model_loader: - type q4_0:  225 tensors
llama_model_loader: - type q6_K:    1 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_0
print_info: file size   = 4.33 GiB (4.64 BPW) 
load: special tokens cache size = 256
load: token to piece cache size = 0.7999 MB
print_info: arch             = llama
print_info: vocab_only       = 0
print_info: n_ctx_train      = 131072
print_info: n_embd           = 4096
print_info: n_layer          = 32
print_info: n_head           = 32
print_info: n_head_kv        = 8
print_info: n_rot            = 128
print_info: n_swa            = 0
print_info: is_swa_any       = 0
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 4
print_info: n_embd_k_gqa     = 1024
print_info: n_embd_v_gqa     = 1024
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-05
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 14336
print_info: n_expert         = 0
print_info: n_expert_used    = 0
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 0
print_info: rope scaling     = linear
print_info: freq_base_train  = 500000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 131072
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 8B
print_info: model params     = 8.03 B
print_info: general.name     = Meta Llama 3.1 8B Instruct
print_info: vocab type       = BPE
print_info: n_vocab          = 128256
print_info: n_merges         = 280147
print_info: BOS token        = 128000 '<|begin_of_text|>'
print_info: EOS token        = 128009 '<|eot_id|>'
print_info: EOT token        = 128009 '<|eot_id|>'
print_info: EOM token        = 128008 '<|eom_id|>'
print_info: LF token         = 198 'Ċ'
print_info: EOG token        = 128001 '<|end_of_text|>'
print_info: EOG token        = 128008 '<|eom_id|>'
print_info: EOG token        = 128009 '<|eot_id|>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 32 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 33/33 layers to GPU
load_tensors:      Vulkan0 model buffer size =  4155.99 MiB
load_tensors:   CPU_Mapped model buffer size =   281.81 MiB
......................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 16384
llama_context: n_ctx_per_seq = 16384
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 1024
llama_context: causal_attn   = 1
llama_context: flash_attn    = 1
llama_context: freq_base     = 500000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (16384) < n_ctx_train (131072) -- the full capacity of the model will not be utilized
llama_context: Vulkan_Host  output buffer size =     0.49 MiB
llama_kv_cache_unified:    Vulkan0 KV buffer size =  2048.00 MiB
llama_kv_cache_unified: size = 2048.00 MiB ( 16384 cells,  32 layers,  1 seqs), K (f16): 1024.00 MiB, V (f16): 1024.00 MiB
llama_context:    Vulkan0 compute buffer size =   613.01 MiB
llama_context: Vulkan_Host compute buffer size =    80.01 MiB
llama_context: graph nodes  = 999
llama_context: graph splits = 2
```

---

👤 **ikawrakow** replied the **2025-07-01** at **14:19:04**:<br>

@jeffbolznv  Thank you for chiming in. Above is the log. Is there something additional I need to do to improve performance? I did
```
mkdir vulkan && cd vulkan
cmake .. -DGGML_VULKAN=ON -DGGML_CUDA=OFF
make -j
```

> 👤 **jeffbolznv** replied the **2025-07-01** at **14:43:13**:<br>
> Is it a release build? I can't tell.
> 
> You'd probably get a boost from a newer driver (to enable coopmat2), but the pp numbers seem slow for coopmat1.
> 
> 👤 **ikawrakow** replied the **2025-07-01** at **14:54:36**:<br>
> Yes, this is a release build. @ubergarm is getting in the range of 3000 t/s for LlaMA-7B on his RX 7900 XTX, so same ball park.

---

👤 **jeffbolznv** replied the **2025-07-01** at **14:53:29**:<br>

What's the llama-bench equivalent of the `N_KV` column in that table? Is it `-d`? I see a big difference between coopmat1 and coopmat2 with large depth.

> 👤 **ikawrakow** replied the **2025-07-01** at **15:00:56**:<br>
> I haven't looked into mainline `llama.cpp`, but the `sweep-bench` here adds `N_KV` tokens to the KV cache, and then runs a batch of a given size (1024 tokens in the above example), and generates a given number of new tokens (256 in the example). Time is measured for both, and resulting tokens/second is printed. The KV cache is increased gradually in a sweep, which corresponds to a typical experience of a user interacting with an LLM.  I don't know what the `-d` option in mainline does (I think it is a relatively recent addition), that's why I have a port of `sweep-bench` to mainline `llama.cpp` to be able to run direct (and more meaningful) comparisons than `-p 512` or `-n 128`).
> 
> 👤 **jeffbolznv** replied the **2025-07-01** at **15:14:47**:<br>
> OK. I think these are basically the same parameter.
> 
> I see much better (>2x) performance for large KV with coopmat2, and I think this is because it's doing more rows at a time (64 vs 16). It might be possible to improve this for the coopmat1 path, but it may start running into register limits, hard to say. For an NV GPU, you should just update to a recent driver (r575) and you'll get the improved performance automatically.
> 
> 👤 **ikawrakow** replied the **2025-07-01** at **15:28:31**:<br>
> > you should just update to a recent driver (r575) and you'll get the improved performance automatically.
> 
> You mean the Nvidia driver?
> I'm on `560.35.03` and reluctant to update as the machine I'm working on is remote.
> 
> But IIRC, you have an RTX-4070. Can you post a comparison between CUDA and Vulkan on your GPU?
> 
> 👤 **jeffbolznv** replied the **2025-07-01** at **16:12:50**:<br>
> I recently got a 5090, so the 4070 is no longer in my system. Here's what I'm seeing for coopmat2, coopmat1, and CUDA.
> 
> ```
> Z:\github\jeffbolznv\llama.cpp\build\bin\RelWithDebInfo>llama-bench -m C:\models\meta-llama-3.1-8b-instruct-q4_0.gguf -fa 1 -n 0 -p 1024 --prio 1 -r 1 -d 1024-15360+1024
> ggml_vulkan: Found 1 Vulkan devices:
> ggml_vulkan: 0 = NVIDIA GeForce RTX 5090 (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
> | model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
> | ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d1024 |      10616.78 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d2048 |       9960.08 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d3072 |       9841.83 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d4096 |       9479.70 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d5120 |       9019.58 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d6144 |       8337.62 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d7168 |       8149.66 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d8192 |       7892.09 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d9216 |       7678.50 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 | pp1024 @ d10240 |       7396.89 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 | pp1024 @ d11264 |       7160.86 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 | pp1024 @ d12288 |       6865.95 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 | pp1024 @ d13312 |       6660.70 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 | pp1024 @ d14336 |       6481.23 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 | pp1024 @ d15360 |       6240.57 ± 0.00 |
> 
> Z:\github\jeffbolznv\llama.cpp\build\bin\RelWithDebInfo>llama-bench -m C:\models\meta-llama-3.1-8b-instruct-q4_0.gguf -fa 1 -n 0 -p 1024 --prio 1 -r 1 -d 1024-15360+1024
> ggml_vulkan: Found 1 Vulkan devices:
> ggml_vulkan: 0 = NVIDIA GeForce RTX 5090 (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat
> | model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
> | ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d1024 |       6484.20 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d2048 |       5791.34 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d3072 |       5398.55 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d4096 |       4879.42 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d5120 |       4477.92 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d6144 |       4112.65 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d7168 |       3902.24 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d8192 |       3651.50 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 |  pp1024 @ d9216 |       3420.07 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 | pp1024 @ d10240 |       3236.93 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 | pp1024 @ d11264 |       3061.68 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 | pp1024 @ d12288 |       2896.88 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 | pp1024 @ d13312 |       2734.89 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 | pp1024 @ d14336 |       2624.02 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | Vulkan     |  99 |  1 | pp1024 @ d15360 |       2496.16 ± 0.00 |
> 
> Z:\github\jeffbolznv\llama.cpp\buildcuda\bin\RelWithDebInfo>llama-bench -m C:\models\meta-llama-3.1-8b-instruct-q4_0.gguf -fa 1 -n 0 -p 1024 --prio 1 -r 1 -d 1024-15360+1024
> ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
> ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
> ggml_cuda_init: found 1 CUDA devices:
>   Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
> | model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
> | ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 |  pp1024 @ d1024 |      12854.24 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 |  pp1024 @ d2048 |      12101.30 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 |  pp1024 @ d3072 |      11831.37 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 |  pp1024 @ d4096 |      11467.68 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 |  pp1024 @ d5120 |      11072.99 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 |  pp1024 @ d6144 |      10646.26 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 |  pp1024 @ d7168 |      10287.17 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 |  pp1024 @ d8192 |       9873.84 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 |  pp1024 @ d9216 |       9688.37 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 | pp1024 @ d10240 |       9373.99 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 | pp1024 @ d11264 |       9117.66 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 | pp1024 @ d12288 |       8706.74 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 | pp1024 @ d13312 |       8635.61 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 | pp1024 @ d14336 |       8351.58 ± 0.00 |
> | llama 8B Q4_0                  |   5.61 GiB |     8.03 B | CUDA       |  99 |  1 | pp1024 @ d15360 |       8134.32 ± 0.00 |
> ```
> 
> > You mean the Nvidia driver?
> > I'm on 560.35.03 and reluctant to update as the machine I'm working on is remote.
> 
> Yeah, r575 has coopmat2 support.
> 
> 👤 **ikawrakow** replied the **2025-07-01** at **16:50:49**:<br>
> OK, thanks, this looks much better.
> 
> 👤 **ubergarm** replied the **2025-07-01** at **19:41:16**:<br>
> @jeffbolznv Thanks for the benchmarks. I'm curious how Vulkan coopmat2 is looking for TG. On the slightly larger Qwen3-14B-Q4_0 I mentioned above how it is actually faster than CUDA on my 3090TI FE for larger kv depths.
> 
> If you are interested, here is one way to use llama-sweep-bench on mainline llama.cpp for comparisons. I just updated my fork/branch to llama.cpp tip of master@de5694414
> 
> ```bash
> cd llama.cpp
> git remote add ubergarm git@github.com:ubergarm/llama.cpp.git
> git fetch ubergarm
> git checkout ug/port-sweep-bench
> # compile as usual for CUDA/Vulkan Release
> # it runs basically like llama-server with similar argument style
> # this might work on your windows box:
> llama-sweep-bench -m C:\models\meta-llama-3.1-8b-instruct-q4_0.gguf -fa -c 8192 -ngl 99 -t 1
> ```
> 
> 👤 **jeffbolznv** replied the **2025-07-01** at **19:53:41**:<br>
> coopmat2 mostly isn't used for tg, but if there's grouped query attention then it may be used for the flash attention shader. It's nice/surprising to see vulkan pull ahead for larger KV. I suspect the Vulkan driver still has some small launch overhead relative to CUDA that hurts at smaller sizes, but I'm not really sure.
> 
> 👤 **ikawrakow** replied the **2025-07-02** at **06:28:28**:<br>
> @jeffbolznv 
> 
> Once you are here, may I ask why flash attention for DeepSeek is not implemented in the `llama.cpp` Vulkan backend? Is it just that nobody has come around to do it, or are there principle issues?  The most efficient FA implementation requires k-head = 192, v-head = 128 for prompt processing, and k-head = 576, v-head = 512 for token generation.
> 
> 👤 **jeffbolznv** replied the **2025-07-02** at **12:51:50**:<br>
> Just nobody has done it yet. I don't think I've seen a version of the model that would even come close to fitting on my GPU. I suppose I could implement it just from the backend tests, but it would be nice to be able to perf test it.
> 
> 👤 **ikawrakow** replied the **2025-07-02** at **13:04:19**:<br>
> Here is a 16B parameter MoE model that easily fits in your 5090 with VRAM to spare that uses the exact same attention mechanism as DeepSeek-V3/R1: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite (except that it has 16 instead of 128 heads). I think this is what Johannes used for testing when he implemented the k-head-size != v-head-size FA in the `llama.cpp` CUDA backend. I did have a partial implementation here using this model quite a bit earlier than mainline (the part for k-head=192, v-head=128), but I was straggling to get a performant implementation for the k-head=576, v-head=512 case, so that's why I asked the question if there are principle issues with the Vulkan implementation.
> 
> 👤 **jeffbolznv** replied the **2025-07-02** at **13:10:56**:<br>
> I thought deepseek v2 was already accelerated and it was only deepseek R1 that uses the large/mixed head sizes?
> 
> 👤 **ikawrakow** replied the **2025-07-02** at **13:13:09**:<br>
> Well, get the model and see what happens.
> 
> 👤 **jeffbolznv** replied the **2025-07-02** at **13:52:10**:<br>
> OK, I do see FA falling back to CPU with it.
> 
> 👤 **jeffbolznv** replied the **2025-07-02** at **20:14:58**:<br>
> I added support for these head sizes in https://github.com/ggml-org/llama.cpp/pull/14509. Performance is tolerable with the coopmat2 shader but very slow for coopmat1/scalar. I'm sure there's some room for tuning.

---

👤 **ikawrakow** replied the **2025-07-02** at **06:16:07**:<br>

> Personally, the two things I see Vulkan back-end support providing are:
>
> A path allowing AMD GPUs to be used e.g. RX 7900 XTX 24GB VRAM

But a port of the mainline Vulkan back-end to `ik_llama.cpp` without the additions that make `ik_llama.cpp` faster for CUDA and CPU inference has zero benefits. People can simply use `llama.cpp` with their AMD GPUs.

> 👤 **firecoperana** replied the **2025-07-02** at **14:32:39**:<br>
> Another benefit is to people who have both nvidia and amd or even intel GPUs. They can use RPC to load different backends or just use vulkan to use non CUDA GPU to offload more weights to vram.
> 
> 👤 **ikawrakow** replied the **2025-07-02** at **14:43:52**:<br>
> > Another benefit is to people who have both nvidia and amd or even intel GPUs. They can use RPC to load different backends or just use vulkan to use non CUDA GPU to offload more weights to vram.
> 
> They already have this with `llama.cpp`. What does `ik_llama.cpp` without the additions implemented for Vulkan give them that they don't already have with `llama.cpp`?
> 
> 👤 **firecoperana** replied the **2025-07-02** at **15:38:13**:<br>
> One major thing I can think of is mla support for old quants of Deepseek V2.5 and V3 models. And if someone is already using ik_llama.cpp, adding AMD gpu that is not useable earlier can offer more speed boost.

---

👤 **ikawrakow** replied the **2025-07-06** at **13:41:44**:<br>

So, the Vulkan back-end is usable, and performance is better than `llama.cpp` (see, e.g., PR #584 that has a comparison for a MoE model). But compared to CUDA on the same GPU, performance is much lower, especially for MoE models (and most users appear to be using `ik_llama.cpp` exactly for one of the giant MoE models). I have mixed feelings how to proceed:
* There is much more performance optimization potential in the Vulkan back-end compared to CUDA or CPU. So, from that point of view it seems worthwhile to put some effort into optimizing the Vulkan back-end
* I know nothing about Vulkan programming in general or the `llama.cpp` Vulkan back-end in particular, hence, at least initially, it will be an uphill battle. Without a significant interest from the user base, I don't feel particularly motivated to do this to myself.

So, if you feel that Vulkan performance improvement in `ik_llama.cpp` is important, go to discussion #590 and vote!