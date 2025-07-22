### 🗣️ [#393](https://github.com/ikawrakow/ik_llama.cpp/discussions/393) - Creating quantized models

| **Author** | `nux` |
| :--- | :--- |
| **Created** | 2025-05-07 |
| **Updated** | 2025-05-29 |

---

#### Description

Hello,  

I've been experimenting with the `ubergarm/DeepSeek-V3-0324-GGUF IQ4_K_R4` model on a system with a 3090 GPU and dual Epyc 9115 CPUs (768GB DDR5 5600). The model runs smoothly with ~11 TPS for short prompts and ~62 TPS eval / 8.65 TPS gen at 17k tokens. It uses ~22.9GB GPU and ~395GB RAM.  

I'm exploring whether higher-precision quantization (e.g., `q5_k_r4`, `q4_k_r4`, `q3_k_r4`) could improve quality while using more RAM. I modified the quantization script from ubergarm's Hugging Face page, but I'm not confident in my approach.  

What I did so far:
- Created a custom quantization config to use `q8_0` for most layers and lower-precision types for Routed Experts (e.g., `q5_k_r4`, `q4_k_r4`, `q3_k_r4`) in different layer ranges.  
- Ran the quantization command (see below).  

quantization script:
```bash
#!/usr/bin/env bash
custom="
# Token embedding (GPU)
token_embd\.weight=q8_0
# output tensors (GPU)
output\.weight=q8_0
output_norm\.weight=q8_0
# First 3 dense layers (0-3) (GPU)
blk\.[0-2]\..*=q8_0
# All attention, weights, and bias tensors (GPU)
blk\.[3-9]\.attn_.*=q8_0
blk\.[1-5][0-9]\.attn_.*=q8_0
blk\.60\.attn_.*=q8_0
blk\.[3-9]\.ffn_norm\.weight=q8_0
blk\.[1-5][0-9]\.ffn_norm\.weight=q8_0
blk\.60\.ffn_norm\.weight=q8_0
blk\.[3-9]\.exp_probs_b\.bias=q8_0
blk\.[1-5][0-9]\.exp_probs_b\.bias=q8_0
blk\.60\.exp_probs_b\.bias=q8_0
# Shared Experts (GPU)
blk\.[3-9]\.ffn_down_shexp\.weight=q8_0
blk\.[1-5][0-9]\.ffn_down_shexp\.weight=q8_0
blk\.60\.ffn_down_shexp\.weight=q8_0
blk\.[3-9]\.ffn_(gate|up)_shexp\.weight=q8_0
blk\.[1-5][0-9]\.ffn_(gate|up)_shexp\.weight=q8_0
blk\.60\.ffn_(gate|up)_shexp\.weight=q8_0
# Routed Experts - Early layers (3-20) (CPU)
blk\.[3-9]\.ffn_down_exps\.weight=q5_k_r4
blk\.1[0-9]\.ffn_down_exps\.weight=q5_k_r4
blk\.[3-9]\.ffn_(gate|up)_exps\.weight=q5_k_r4
blk\.1[0-9]\.ffn_(gate|up)_exps\.weight=q5_k_r4
# Routed Experts - Middle layers (21-40) (CPU)
blk\.2[0-9]\.ffn_down_exps\.weight=q4_k_r4
blk\.3[0-9]\.ffn_down_exps\.weight=q4_k_r4
blk\.2[0-9]\.ffn_(gate|up)_exps\.weight=q4_k_r4
blk\.3[0-9]\.ffn_(gate|up)_exps\.weight=q4_k_r4
# Routed Experts - Later layers (41-60) (CPU)
blk\.4[0-9]\.ffn_down_exps\.weight=q3_k_r4
blk\.5[0-9]\.ffn_down_exps\.weight=q3_k_r4
blk\.60\.ffn_down_exps\.weight=q3_k_r4
blk\.4[0-9]\.ffn_(gate|up)_exps\.weight=q3_k_r4
blk\.5[0-9]\.ffn_(gate|up)_exps\.weight=q3_k_r4
blk\.60\.ffn_(gate|up)_exps\.weight=q3_k_r4
"

custom=$(
echo "$custom" | grep -v '^#' | \
sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

/home/nux/dev/ik_llama.cpp/build/bin/llama-quantize \
--imatrix /mnt/nvme/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324.imatrix \
--token-embedding-type q8_0 \
--output-tensor-type q8_0 \
--custom-q "$custom" \
/mnt/amp/models/unsloth/DeepSeek-V3-0324-GGUF/BF16/DeepSeek-V3-0324-BF16-00001-of-00030.gguf \
/mnt/nvme/models/nux/DeepSeek-V3-0324/DeepSeek-V3-0324-OPTIMIZED.gguf \
Q5_K_M \
24
```

perplexity:

```bash
# ./build/bin/llama-perplexity     --model /mnt/nvme/models/nux/DeepSeek-V3-0324/DeepSeek-V3-0324-OPTIMIZED.gguf     -ctk q8_0     -mla 2 -fa     -amb 512     -fmoe     --ctx-size 512     --ubatch-size 512     -f wiki.test.raw     --seed 1337     --n-gpu-layers 63     --override-tensor exps=CPU     --threads 32 >> /mnt/nvme/models/nux/DeepSeek-V3-0324/perp.txt 2>&1

ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
main: build = 3668 (6c23618c)
main: built with cc (Debian 12.2.0-14) 12.2.0 for x86_64-linux-gnu
main: seed  = 1337
llama_model_loader: loaded meta data with 50 key-value pairs and 1025 tensors from /mnt/nvme/models/nux/DeepSeek-V3-0324/DeepSeek-V3-0324-OPTIMIZED.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324 BF16
llama_model_loader: - kv   3:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   4:                         general.size_label str              = 256x20B
llama_model_loader: - kv   5:                            general.license str              = mit
llama_model_loader: - kv   6:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   7:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   8:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   9:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  10:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  11:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  12:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  13:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  14: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  16:                          general.file_type u32              = 17
llama_model_loader: - kv  17:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  18:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  19:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  20:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  21:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  22:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  23:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  24:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  25:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  26:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  27:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  28:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  29:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  30:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  31:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  32: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  33: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<?...
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  45:               general.quantization_version u32              = 2
llama_model_loader: - kv  46:                      quantize.imatrix.file str              = /mnt/nvme/models/ubergarm/DeepSeek-V3...
llama_model_loader: - kv  47:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  48:             quantize.imatrix.entries_count i32              = 720
llama_model_loader: - kv  49:              quantize.imatrix.chunks_count i32              = 213
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  490 tensors
llama_model_loader: - type q3_k_r4:   63 tensors
llama_model_loader: - type q4_k_r4:   60 tensors
llama_model_loader: - type q5_k_r4:   51 tensors
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
llm_load_print_meta: model ftype      = Q5_K - Medium
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 352.822 GiB (4.517 BPW) 
llm_load_print_meta: repeating layers = 350.988 GiB (4.506 BPW, 669.173 B parameters)
llm_load_print_meta: general.name     = DeepSeek V3 0324 BF16
llm_load_print_meta: BOS token        = 0 '<｜begin▁of▁sentence｜>'
llm_load_print_meta: EOS token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: PAD token        = 1 '<｜end▁of▁sentence｜>'
llm_load_print_meta: LF token         = 131 'Ä'
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
llm_load_tensors: ggml ctx size =    0.85 MiB
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
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
llm_load_tensors:        CPU buffer size = 358335.85 MiB
llm_load_tensors:        CPU buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 16706.99 MiB
....................................................................................................
============ llm_load_tensors: need to compute 61 wk_b tensors
Computed blk.0.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.1.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.2.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.3.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.4.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.5.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.6.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.7.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.8.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.9.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.10.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.11.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.12.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.13.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.14.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.15.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.16.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.17.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.18.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.19.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.20.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.21.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.22.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.23.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.24.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.25.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.26.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.27.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.28.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.29.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.30.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.31.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.32.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.33.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.34.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.35.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.36.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.37.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.38.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.39.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.40.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.41.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.42.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.43.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.44.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.45.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.46.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.47.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.48.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.49.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.50.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.51.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.52.attn_v_b.weight as 128 x 512 x 128llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 2
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init: layer 0: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 1: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 2: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 3: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 4: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 5: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 6: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 7: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 8: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 9: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 10: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 11: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 12: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 13: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 14: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 15: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 16: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 17: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 18: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 19: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 20: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 21: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 22: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 23: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 24: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 25: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 26: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 27: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 28: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 29: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 30: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 31: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 32: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 33: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 34: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 35: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 36: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 37: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 38: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 39: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 40: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 41: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 42: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 43: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 44: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 45: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 46: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 47: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 48: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 49: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 50: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 51: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 52: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 53: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 54: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 55: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 56: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 57: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 58: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 59: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init: layer 60: n_embd_head_qk_rope = 64, kv_lora_rank = 512
llama_kv_cache_init:      CUDA0 KV buffer size =    72.94 MiB
llama_new_context_with_model: KV self size  =   72.91 MiB, c^KV (q8_0):   72.91 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.97 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   503.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   162.01 MiB
llama_new_context_with_model: graph nodes  = 3548
llama_new_context_with_model: graph splits = 118

system_info: n_threads = 32 / 64 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 697.458 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 23.76 seconds per pass - ETA 55.53 minutes
 and stored in buffer CUDA0
Computed blk.53.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.54.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.55.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.56.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.57.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.58.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.59.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
Computed blk.60.attn_v_b.weight as 128 x 512 x 128 and stored in buffer CUDA0
[1]2.4727,[2]3.1883,[3]2.3363,[4]1.9646,[5]1.7720,[6]1.6345,[7]1.5490,[8]1.4876,[9]1.4439,[10]1.4068,[11]1.3941,[12]1.4216,[13]1.4365,[14]1.5616,[15]1.6908,[16]1.7475,[17]1.9062,[18]2.0309,[19]1.9925,[20]1.9848,[21]2.0866,[22]2.0605,[23]2.0343,[24]2.0436,[25]2.0133,[26]1.9922,[27]2.0366,[28]2.0458,[29]2.0933,[30]2.1265,[31]2.1579,[32]2.1765,[33]2.2166,[34]2.2604,[35]2.3081,[36]2.3588,[37]2.3937,[38]2.4407,[39]2.4825,[40]2.5397,[41]2.5797,[42]2.5937,[43]2.6436,[44]2.6593,[45]2.7400,[46]2.7900,[47]2.7470,[48]2.7026,[49]2.6785,[50]2.6973,[51]2.7409,[52]2.7529,[53]2.8040,[54]2.8183,[55]2.8495,[56]2.8797,[57]2.8936,[58]2.9262,[59]2.9363,[60]2.9827,[61]3.0216,[62]3.0687,[63]3.1002,[64]3.1411,[65]3.1497,[66]3.1340,[67]3.1110,[68]3.1375,[69]3.1332,[70]3.1463,[71]3.1640,[72]3.1775,[73]3.1921,[74]3.2131,[75]3.1924,[76]3.1473,[77]3.1041,[78]3.0989,[79]3.0800,[80]3.0627,[81]3.0285,[82]3.0319,[83]3.0028,[84]2.9684,[85]2.9355,[86]2.9123,[87]2.9089,[88]2.8820,[89]2.8663,[90]2.8415,[91]2.8139,[92]2.7901,[93]2.7646,[94]2.7414,[95]2.7197,[96]2.7192,[97]2.7244,[98]2.7108,[99]2.6936,[100]2.6959,[101]2.6885,[102]2.7034,[103]2.7290,[104]2.7462,[105]2.7423,[106]2.7650,[107]2.7894,[108]2.8104,[109]2.8429,[110]2.8765,[111]2.8947,[112]2.8690,[113]2.8561,[114]2.8345,[115]2.8196,[116]2.8057,[117]2.7833,[118]2.7622,[119]2.7421,[120]2.7239,[121]2.7077,[122]2.6904,[123]2.6725,[124]2.6545,[125]2.6375,[126]2.6206,[127]2.6076,[128]2.6003,[129]2.5900,[130]2.5776,[131]2.5692,[132]2.5755,[133]2.5854,[134]2.5922,[135]2.6030,[136]2.6172,[137]2.6324,[138]2.6405,[139]2.6515,[140]2.6517,[141]2.6533,[142]2.6521,[143]2.6532,[144]2.6503,[145]2.6423,[146]2.6405,[147]2.6450,[148]2.6447,[149]2.6459,[150]2.6403,[151]2.6382,[152]2.6350,[153]2.6309,[154]2.6307,[155]2.6342,[156]2.6359,[157]2.6413,[158]2.6493,[159]2.6519,[160]2.6605,[161]2.6687,[162]2.6789,[163]2.6847,[164]2.7049,[165]2.7283,[166]2.7454,[167]2.7577,[168]2.7815,[169]2.8036,[170]2.8256,[171]2.8470,[172]2.8316,[173]2.8155,[174]2.8027,[175]2.7913,[176]2.7800,[177]2.7689,[178]2.7561,[179]2.7428,[180]2.7464,[181]2.7605,[182]2.7758,[183]2.7898,[184]2.8028,[185]2.8126,[186]2.8284,[187]2.8436,[188]2.8572,[189]2.8675,[190]2.8683,[191]2.8753,[192]2.8781,[193]2.8831,[194]2.9024,[195]2.9111,[196]2.9239,[197]2.9337,[198]2.9382,[199]2.9439,[200]2.9431,[201]2.9578,[202]2.9531,[203]2.9586,[204]2.9614,[205]2.9613,[206]2.9641,[207]2.9723,[208]2.9814,[209]2.9901,[210]2.9901,[211]2.9852,[212]2.9856,[213]2.9933,[214]2.9953,[215]3.0008,[216]3.0011,[217]2.9963,[218]2.9963,[219]2.9970,[220]2.9968,[221]2.9971,[222]2.9969,[223]2.9976,[224]3.0023,[225]3.0043,[226]2.9962,[227]2.9937,[228]2.9952,[229]2.9988,[230]3.0051,[231]3.0109,[232]3.0025,[233]2.9956,[234]2.9957,[235]2.9947,[236]3.0036,[237]3.0116,[238]3.0208,[239]3.0302,[240]3.0399,[241]3.0508,[242]3.0647,[243]3.0770,[244]3.0851,[245]3.0963,[246]3.1068,[247]3.1054,[248]3.1011,[249]3.0992,[250]3.0935,[251]3.0914,[252]3.0934,[253]3.0973,[254]3.1043,[255]3.1104,[256]3.1135,[257]3.1165,[258]3.1179,[259]3.1212,[260]3.1237,[261]3.1252,[262]3.1242,[263]3.1294,[264]3.1317,[265]3.1323,[266]3.1339,[267]3.1358,[268]3.1393,[269]3.1422,[270]3.1411,[271]3.1398,[272]3.1335,[273]3.1335,[274]3.1269,[275]3.1164,[276]3.1056,[277]3.1077,[278]3.1175,[279]3.1234,[280]3.1308,[281]3.1380,[282]3.1437,[283]3.1499,[284]3.1560,[285]3.1695,[286]3.1715,[287]3.1744,[288]3.1794,[289]3.1816,[290]3.1739,[291]3.1657,[292]3.1642,[293]3.1634,[294]3.1613,[295]3.1588,[296]3.1604,[297]3.1609,[298]3.1663,[299]3.1721,[300]3.1751,[301]3.1791,[302]3.1807,[303]3.1819,[304]3.1812,[305]3.1926,[306]3.1994,[307]3.2100,[308]3.1990,[309]3.1939,[310]3.1849,[311]3.1880,[312]3.1899,[313]3.1952,[314]3.1972,[315]3.2003,[316]3.2015,[317]3.2034,[318]3.2037,[319]3.2039,[320]3.2080,[321]3.2082,[322]3.2100,[323]3.2165,[324]3.2173,[325]3.2222,[326]3.2267,[327]3.2307,[328]3.2330,[329]3.2346,[330]3.2409,[331]3.2441,[332]3.2479,[333]3.2468,[334]3.2467,[335]3.2473,[336]3.2474,[337]3.2485,[338]3.2487,[339]3.2513,[340]3.2549,[341]3.2603,[342]3.2689,[343]3.2779,[344]3.2827,[345]3.2742,[346]3.2666,[347]3.2621,[348]3.2550,[349]3.2515,[350]3.2501,[351]3.2547,[352]3.2692,[353]3.2779,[354]3.2904,[355]3.2987,[356]3.3044,[357]3.3155,[358]3.3253,[359]3.3282,[360]3.3344,[361]3.3436,[362]3.3520,[363]3.3572,[364]3.3639,[365]3.3698,[366]3.3798,[367]3.3882,[368]3.3948,[369]3.4023,[370]3.4109,[371]3.4240,[372]3.4325,[373]3.4360,[374]3.4391,[375]3.4440,[376]3.4563,[377]3.4673,[378]3.4701,[379]3.4702,[380]3.4668,[381]3.4717,[382]3.4772,[383]3.4805,[384]3.4847,[385]3.4884,[386]3.4943,[387]3.5002,[388]3.5032,[389]3.4930,[390]3.4839,[391]3.4739,[392]3.4686,[393]3.4593,[394]3.4508,[395]3.4419,[396]3.4322,[397]3.4236,[398]3.4143,[399]3.4042,[400]3.3954,[401]3.3858,[402]3.3757,[403]3.3675,[404]3.3577,[405]3.3485,[406]3.3388,[407]3.3295,[408]3.3209,[409]3.3126,[410]3.3069,[411]3.3080,[412]3.3036,[413]3.3058,[414]3.3077,[415]3.3051,[416]3.3051,[417]3.3071,[418]3.3014,[419]3.3025,[420]3.2998,[421]3.2986,[422]3.2989,[423]3.2986,[424]3.3025,[425]3.3023,[426]3.3022,[427]3.3016,[428]3.3042,[429]3.3054,[430]3.3082,[431]3.3092,[432]3.3081,[433]3.3044,[434]3.3048,[435]3.2978,[436]3.2922,[437]3.2882,[438]3.2865,[439]3.2838,[440]3.2884,[441]3.2938,[442]3.3010,[443]3.2988,[444]3.2995,[445]3.3006,[446]3.3049,[447]3.3081,[448]3.3102,[449]3.3131,[450]3.3167,[451]3.3196,[452]3.3217,[453]3.3231,[454]3.3217,[455]3.3241,[456]3.3244,[457]3.3268,[458]3.3318,[459]3.3324,[460]3.3326,[461]3.3294,[462]3.3329,[463]3.3401,[464]3.3446,[465]3.3384,[466]3.3363,[467]3.3345,[468]3.3359,[469]3.3333,[470]3.3305,[471]3.3310,[472]3.3315,[473]3.3308,[474]3.3297,[475]3.3309,[476]3.3295,[477]3.3288,[478]3.3296,[479]3.3313,[480]3.3340,[481]3.3301,[482]3.3336,[483]3.3328,[484]3.3362,[485]3.3424,[486]3.3456,[487]3.3490,[488]3.3543,[489]3.3568,[490]3.3618,[491]3.3677,[492]3.3721,[493]3.3719,[494]3.3731,[495]3.3753,[496]3.3772,[497]3.3801,[498]3.3805,[499]3.3800,[500]3.3840,[501]3.3884,[502]3.3874,[503]3.3860,[504]3.3880,[505]3.3911,[506]3.3992,[507]3.4022,[508]3.4056,[509]3.3983,[510]3.3933,[511]3.3871,[512]3.3828,[513]3.3771,[514]3.3755,[515]3.3777,[516]3.3727,[517]3.3729,[518]3.3715,[519]3.3721,[520]3.3760,[521]3.3750,[522]3.3736,[523]3.3789,[524]3.3779,[525]3.3764,[526]3.3723,[527]3.3672,[528]3.3640,[529]3.3609,[530]3.3581,[531]3.3550,[532]3.3495,[533]3.3437,[534]3.3394,[535]3.3401,[536]3.3425,[537]3.3456,[538]3.3478,[539]3.3507,[540]3.3558,[541]3.3588,[542]3.3612,[543]3.3559,[544]3.3518,[545]3.3513,[546]3.3449,[547]3.3389,[548]3.3325,[549]3.3262,[550]3.3204,[551]3.3143,[552]3.3088,[553]3.3030,[554]3.3015,[555]3.2999,[556]3.3026,[557]3.3065,[558]3.3124,[559]3.3168,[560]3.3221,[561]3.3202,
llama_print_timings:        load time =   10030.15 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time = 3302906.56 ms / 287232 tokens (   11.50 ms per token,    86.96 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time = 3306566.09 ms / 287233 tokens

Final estimate: PPL = 3.3202 +/- 0.01831
```

Some notes:
- The resulting model is slightly smaller than the original (353GB vs 387GB)
- I initially used `q5_k_r4` instead of `q3_k_r4` in some layers (as per ubergarm's `iq3_k_r4`). Didn't notice the iq vs q until writing this up
- I ran a perplexity test, but lack proper benchmarks. The model seems comparable to the original for basic prompts.

Some questions:
- Does this quantization strategy make sense?  
- Are there obvious issues with the layer ranges or quantization types?  
- I meant to make it bigger/higher quality, but file size is smaller. Why is that?
- Considering trying again but modifying ubergarms $custom with: iq5_k_r4 in place of iq3_k_r4 and iq4_k_r4 in place of iq2_k_r4. Or should I just put all a to IQ6_K or something?

Any feedback or suggestions would be appreciated!