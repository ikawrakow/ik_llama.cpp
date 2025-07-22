### 🔀 [#492](https://github.com/ikawrakow/ik_llama.cpp/pull/492) - CUDA implementation for IQ1_S_R4

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-04 |
| **Updated** | 2025-06-05 |

---

#### Description

Apparently there are people who would like to use `IQ1_S` or `IQ1_S_R4` quantized models. This PR adds CUDA implementation for `IQ1_S_R4`.

It seems there has been some confusion about which of these quants is supported where (see discussions in #477)

To clarify:
* `IQ1_S` and `IQ1_S_R4` have both fast GEMM and GEMV on the CPU, but `IQ1_S_R4` is faster for prompt processing due to row interleaving
* `IQ1_S` has GEMM and GEMV on CUDA. GEMM is quantized (a.k.a., MMQ)
* `IQ1_S_R4` **does not have** CUDA implementation at all on the main branch. This PR adds it. ~GEMM is implemented via dequantize+cuBLAS. Because of this, `cmake -DGGML_CUDA_IQK_FORCE_BF16 ...` may be required for DeepSeek models (and for some people with newer GPUs, this may be even faster)~. It is MMQ on Turing or newer, it will fall back to dequantize+cuBLAS on older cards. In that case, `cmake -DGGML_CUDA_IQK_FORCE_BF16 ...` may be required for DeepSeek models 
* `IQ1_S` **cannot be repacked** to `IQ1_S_R4`. This is because, unlike other quants where the exact same bits are simply rearranged to obtain the corresponding `_R4` or `_R8` quant, these two quants are not 100% equivalent. `IQ1_S` uses float scales per super-blocks of 256 weights, while `IQ1_S_R4` uses a single float scale for an entire tensor row (and is therefore slightly smaller with exactly 1.5 bpw, while `IQ1_S` is 1.5625 bpw). I broke the symmetry to be able to use `IQ1_S_R4` for models where some tensor row sizes are not a multiple of 256 (e.g., the 16B parameter DeepSeek-Lite model).

Here a quick performance comparison between `IQ1_S` and `IQ1_S_R4` for Qwen3-22B-A3B. Both are quantized with this recipe
```
 ./bin/llama-quantize --imatrix qwen3_imat_unsloth.dat --custom-q "token_embd\.weight=q4_K,attn=iq4_ks,ffn_down=iq2_k,ffn_.*_exps=iq1_s" ../models/qwen3moe/Qwen3-128x1.8B-BF16.gguf $mode iq1_s
```
(but in the `IQ1_S_R4` version all quantization types have `_r4` appended). GPU is RTX-4080, `sweep-bench` command is
```
./bin/llama-sweep-bench -m $model -c 16384 -b 4096 -ub 4096 -fmoe -fa -t 1 -ngl 100
```

### IQ1_S

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    0.748 |  5479.20 |    6.507 |   157.38 |
|  4096 |   1024 |   4096 |    0.865 |  4736.71 |    7.206 |   142.11 |
|  4096 |   1024 |   8192 |    0.999 |  4098.74 |    8.107 |   126.32 |
|  4096 |   1024 |  12288 |    1.140 |  3593.76 |    8.748 |   117.06 |

### IQ1_S_R4

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    0.778 |  5264.28 |    6.004 |   170.57 |
|  4096 |   1024 |   4096 |    0.936 |  4376.45 |    6.694 |   152.98 |
|  4096 |   1024 |   8192 |    1.033 |  3965.54 |    7.556 |   135.52 |
|  4096 |   1024 |  12288 |    1.169 |  3505.10 |    8.322 |   123.04 |


~As expected, IQ1_S has faster prompt processing due to MMQ. But, surprise, surprise, IQ1_S_R4 beats the IQ1_S implementation (which comes from Johannes) by about 10%.~

PP is (almost) on par with `IQ1_S`, but surprise, surprise, `IQ1_S_R4` beats the `IQ1_S` implementation (which comes from Johannes) by ~10%.

Here is the performance with dequantize+cuBLAS that I had originally:

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |    0.955 |  4290.21 |    5.938 |   172.44 |
|  4096 |   1024 |   4096 |    1.023 |  4001.99 |    6.637 |   154.28 |
|  4096 |   1024 |   8192 |    1.161 |  3529.12 |    7.432 |   137.78 |
|  4096 |   1024 |  12288 |    1.297 |  3157.94 |    8.135 |   125.87 |

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-06-04** at **22:53:11**:<br>

Well shucks, I tried this PR, but I'm not able to get the R1-0528-IQ1_S_R4 to run with GPU offload. I tried a few compilation options with and without `-DGGML_CUDA_IQK_FORCE_BF16=1` and the IQ1_S runs fine with the exact same llama-sweep-bench command.

This is on the 7965WX 256GB RAM + Dual RTX A6000 (96GB VRAM total) rig.

Watching `nvitop` the GPUs use low power even at 100% utilization as if it is just copying data perhaps and not actually running computations still like on main. I tried a single visible CUDA device as well but same behavior. I tried the earlier GEMV commit of `33ced81c` but same behavior.

## PR496@fb6a0d01 IQ1_S
`main: n_kv_max = 16384, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 99, n_threads = 24, n_threads_batch = 24`
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   10.083 |   406.25 |   65.816 |    15.56 |
|  4096 |   1024 |   4096 |   12.563 |   326.04 |   68.079 |    15.04 |
|  4096 |   1024 |   8192 |   15.014 |   272.81 |   71.013 |    14.42 |
|  4096 |   1024 |  12288 |   17.540 |   233.52 |   73.294 |    13.97 |

## PR496@fb6a0d01 IQ1_S_R4
`main: n_kv_max = 16384, n_batch = 512, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 24, n_threads_batch = 24`
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    6.579 |    77.82 |  148.734 |     0.86 |

I assume it should works with partial offload situation with some layers on CPU? Not sure what else to try in terms of compiler options etc, but maybe I'm doing something wrong?

Not sure if @Thireus or @randoentity have tried yet and found it working or not?

I found it odd that [line 174 of mmq.c in ggml_cuda_should_use_mmq()](https://github.com/ikawrakow/ik_llama.cpp/pull/492/commits/fb6a0d0184cf326a482e87bc741dc004402cf3f2#diff-b2fe862fcd5119199ae59ea13d1b6a46e0d23e41e727e39d90913f828a5ff66bR181-R183) 
```
    if (type == GGML_TYPE_IQ1_S_R4) {
        return false;
    }
```
So funzies I tried to compile with `-DGGML_CUDA_FORCE_MMQ` but still, no dice.

Anyway, the logs below if it is of any use. Thanks!

<details>

<summary>👈 Commands & Logs</summary>

#### Clean Build
```bash
# pull the PR branch
$ git branch | grep '*'
* ik/cuda_iq1_s_r4

$ git rev-parse --short HEAD
fb6a0d01

# clean build with no cache
$ rm -rf build
$ cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DGGML_CCACHE=OFF
$ cmake --build ./build --config Release -j $(nproc)
```

#### llama-sweep-bench
```bash
#model=DeepSeek-R1-0528-IQ1_S-00001-of-00003.gguf
model=DeepSeek-R1-0528-IQ1_S_R4-00001-of-00003.gguf

./build/bin/llama-sweep-bench \
  --model "$model" \
  -c 16384 \
  -ctk f16 \
  -mla 3 -fa \
  -amb 512 \
  -fmoe \
  -ngl 99 \
  -ot "blk\.(3|4|5|6|7|8|9|10|11|12|13|13|14|15|16|17|18|19)\.ffn_.*=CUDA0" \
  -ot "blk\.(20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36)\.ffn_.*=CUDA1" \
  -ot exps=CPU \
  -b 4096 -ub 4096 \
  --warmup-batch \
  --threads 24


ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
  Device 1: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 52 key-value pairs and 1147 tensors from /mnt/raid/hf/DeepSeek-R1-0528-GGUF/IQ1_S_R4/DeepSeek-R1-0528-IQ1_S_R4-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 0528
llama_model_loader: - kv   3:                            general.version str              = 0528
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek-R1
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   7:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   8:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   9:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  10:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  11:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  12:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  13: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  14:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  15:                          general.file_type u32              = 224
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  21:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  22:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  23:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  24:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  25:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  26:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  27:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  28:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  29:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  30:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  31: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  32: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  33:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  34:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  35:                      tokenizer.ggml.tokens arr[str,129280]  = ["
llama_model_loader: - kv  36:                  tokenizer.ggml.token_type arr[i32,129280]  = [3
llama_model_loader: - kv  37:                      tokenizer.ggml.merges arr[str,127741]  = ["
llama_model_loader: - kv  38:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  39:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  40:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  41:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  42:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  43:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  44:               general.quantization_version u32              = 2
llama_model_loader: - kv  45:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/DeepSeek-R1...
llama_model_loader: - kv  46:                   quantize.imatrix.dataset str              = ubergarm-imatrix-calibration-corpus-v...
llama_model_loader: - kv  47:             quantize.imatrix.entries_count i32              = 721
llama_model_loader: - kv  48:              quantize.imatrix.chunks_count i32              = 812
llama_model_loader: - kv  49:                                   split.no u16              = 0
llama_model_loader: - kv  50:                                split.count u16              = 3
llama_model_loader: - kv  51:                        split.tensors.count i32              = 1147
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q4_0:   61 tensors
llama_model_loader: - type iq4_ks:  551 tensors
llama_model_loader: - type iq1_s_r4:  116 tensors
llama_model_loader: - type iq1_m_r4:   58 tensors
llm_load_vocab: special tokens cache size = 818
llm_load_vocab: token to piece cache size = 0.8223 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = deepseek2
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 129280
llm_load_print_meta: n_merges         = 127741
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 163840
llm_load_print_meta: n_embd           = 7168
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 128
llm_load_print_meta: n_head_kv        = 128
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 192
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 24576
llm_load_print_meta: n_embd_v_gqa     = 16384
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 18432
llm_load_print_meta: n_expert         = 256
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = yarn
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 0.025
llm_load_print_meta: n_ctx_orig_yarn  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 671B
llm_load_print_meta: model ftype      = IQ1_S_R4 - 1.5 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 130.203 GiB (1.664 BPW) 
llm_load_print_meta: repeating layers = 129.285 GiB (1.657 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
llm_load_print_meta: BOS token        = 0 '<
llm_load_print_meta: EOS token        = 1 '<
llm_load_print_meta: PAD token        = 1 '<
llm_load_print_meta: LF token         = 131 '
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_layer_dense_lead   = 3
llm_load_print_meta: n_lora_q             = 1536
llm_load_print_meta: n_lora_kv            = 512
llm_load_print_meta: n_ff_exp             = 2048
llm_load_print_meta: n_expert_shared      = 1
llm_load_print_meta: expert_weights_scale = 2.5
llm_load_print_meta: expert_weights_norm  = 1
llm_load_print_meta: expert_gating_func   = sigmoid
llm_load_print_meta: rope_yarn_log_mul    = 0.1000
llm_load_tensors: ggml ctx size =    1.40 MiB
Tensor blk.3.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.3.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.4.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.5.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.6.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.7.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.8.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.9.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.10.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.11.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.12.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.13.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.14.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.15.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.16.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.17.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.18.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_norm.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_gate_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_down_shexp.weight buffer type overriden to CUDA0
Tensor blk.19.ffn_up_shexp.weight buffer type overriden to CUDA0
Tensor blk.20.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.20.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.21.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.22.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.23.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.24.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.25.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.26.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.27.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.28.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.29.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.30.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.30.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.30.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.30.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.30.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.31.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.31.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.31.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.31.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.31.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.32.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.33.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.33.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.33.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.33.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.33.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.34.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.34.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.34.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.34.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.34.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.35.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.35.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.35.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.35.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.35.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.36.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.36.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.36.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.36.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.36.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size = 10527.62 MiB
llm_load_tensors:        CPU buffer size = 44211.82 MiB
llm_load_tensors:        CPU buffer size =   469.99 MiB
llm_load_tensors:      CUDA0 buffer size = 40696.76 MiB
llm_load_tensors:      CUDA1 buffer size = 40957.25 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   576.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   522.00 MiB
llama_new_context_with_model: KV self size  = 1098.00 MiB, c^KV (f16): 1098.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  2094.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  2125.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   932.00 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 189

main: n_kv_max = 16384, n_batch = 512, n_ubatch = 512, flash_attn = 1, n_gpu_layers = 99, n_threads = 24, n_threads_batch = 24

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    6.579 |    77.82 |  148.734 |     0.86 |
^C
```

</details>

---

👤 **ubergarm** commented the **2025-06-05** at **04:19:52**:<br>

Okay, it works after removing the iq1_m_r4 layers! I rolled a new `IQ1_S_R4-smol` which is `iq1_s_r4` for all `exps` but I bumped up attn/token_embd/shexp to `iq5_ks`. 

![thud-sweep-R1-0528-IQ1_S_R4-smol](https://github.com/user-attachments/assets/2e7ef8c1-1fa9-4dfc-85da-12dddddc060a)

You can see how both GPUs are offloaded and with some utilization along with decent power usage:
![sweep-bench-screenshot-R1-0528-IQ1_S_R4-smol](https://github.com/user-attachments/assets/e3d7635a-8ca2-4f9f-834e-003cbc5f92a6)

I'll go test perplexity on this little guy and see how it looks. Thanks!