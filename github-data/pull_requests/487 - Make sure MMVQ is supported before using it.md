## 🔀 [Pull Request #487](https://github.com/ikawrakow/ik_llama.cpp/pull/487) - Make sure MMVQ is supported before using it

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `ik/mmvq_type_supported` |
| **Target Branch** | `main` |
| **Created** | 2025-06-03 |
| **Updated** | 2025-06-03 |

---

## 📄 Description

The new trellis quants do not support quantized matrix-vector multiplications (a.k.a., MMVQ), but the fused ffn_up+ffn_gate implementation does not check for that, which leads to an assert when the MMVQ is called for a trellis quant.

This PR attempts to fix it.

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-06-03** at **19:43:41**

Okay tested this PR which let's now run full DeepSeek-R1-0528 running a mix of all three new trellis quants with CUDA compiled.

1. This PR does fix my previous error `ik_llama.cpp/ggml/src/ggml-cuda/mmvq.cu:564: fatal error`.
2. I was able to offload onto two CUDA GPUs and do some very limited inferencing testing that looked okay.
3. Began running the usual `llama-perplexity` test but started getting `nan` after chunk 25.
4. If I compile with `-DGGML_CUDA_F16=ON` it seems to still inference okay, but perplexity throws `nan` immedeately on first chunk.
5. Compiling with `-DGGML_CUDA_IQK_FORCE_BF16=1` still throws nan after chunk 25.

Thanks, happy to try any other configurations or build flags etc. Otherwise might try CPU only to get this perplexity value for now haha...

<details>

<summary>👈 Details and Logs</summary>

## Testing PR487
#### Quant
DeepSeek-R1-0528-IQ2_KT 196.696 GiB (2.514 BPW)
- type  f32:  361 tensors
- type q5_0:   61 tensors `attn_k_b`
- type iq2_kt:  116 tensors `ffn_(gate|up)_exps`
- type iq3_kt:   58 tensors `ffn_down_exps`
- type iq4_kt:  551 tensors everything else

#### Rig
* CPU/RAM
    - AMD 7965WX 24x Core 256GB DDR5@4800 
* GPUs
    - Dual RTX A6000 48GB VRAM each total 96GB VRAM

#### Methodology and Logs
```bash
git checkout ik/mmvq_type_supported
git rev-parse --short HEAD
626f49ab

# also tested with -DGGML_CUDA_IQK_FORCE_BF16=1 with same results
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1
cmake --build ./build --config Release -j $(nproc)

model=/mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-IQ2_KT.gguf
./build/bin/llama-perplexity \
    --model "$model" \
    -f wiki.test.raw \
    --seed 1337 \
    --ctx-size 512 \
    -mla 3 -fa \
    -amb 512 \
    -fmoe \
    -ngl 99 \
    -ot "blk\.(3|4|5|6|7|8|9|10|11|12|13|13)\.ffn_.*=CUDA0" \
    -ot "blk\.(14|16|17|18|19|20|21|22|23|24|25)\.ffn_.*=CUDA1" \
    -ot exps=CPU \
    --threads 24

ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
  Device 1: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
main: build = 3724 (626f49ab)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: seed  = 1337
llama_model_loader: loaded meta data with 49 key-value pairs and 1147 tensors from /mnt/raid/models/ubergarm/DeepSeek-R1-0528-GGUF/DeepSeek-R1-0528-IQ2_KT.gguf (version GGUF V3 (latest))
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
llama_model_loader: - kv  15:                          general.file_type u32              = 151
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
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q5_0:   61 tensors
llama_model_loader: - type iq2_kt:  116 tensors
llama_model_loader: - type iq3_kt:   58 tensors
llama_model_loader: - type iq4_kt:  551 tensors
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
llm_load_print_meta: model ftype      = IQ2_KT - 2.125 bpw
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 196.696 GiB (2.514 BPW) 
llm_load_print_meta: repeating layers = 195.831 GiB (2.510 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1 0528
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
Tensor blk.14.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.14.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.14.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.14.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.14.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.16.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.17.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.18.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.19.ffn_up_shexp.weight buffer type overriden to CUDA1
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
llm_load_tensors:        CPU buffer size = 158670.43 MiB
llm_load_tensors:        CPU buffer size =   442.86 MiB
llm_load_tensors:      CUDA0 buffer size = 40719.56 MiB
llm_load_tensors:      CUDA1 buffer size = 40914.69 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 512
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =    72.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =    65.25 MiB
llama_new_context_with_model: KV self size  =  137.25 MiB, c^KV (f16):  137.25 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.97 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  2043.00 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =   476.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    18.01 MiB
llama_new_context_with_model: graph nodes  = 3487
llama_new_context_with_model: graph splits = 148

system_info: n_threads = 24 / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
perplexity: tokenizing the input ..
perplexity: tokenization took 604.513 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=2048, n_seq=4
perplexity: 49.29 seconds per pass - ETA 1 hours 55.22 minutes
[1]2.6769,[2]3.4157,[3]2.4584,[4]2.0552,[5]1.8836,[6]1.7454,[7]1.6643,[8]1.6095,[9]1.5704,[10]1.5282,[11]1.5253,[12]1.6034,[13]1.6284,[14]1.7546,[15]1.8981,[16]1.9518,[17]2.1167,[18]2.2438,[19]2.2098,[20]2.2026,[21]2.3055,[22]2.2735,[23]2.2438,[24]2.2625,[25]nan,[26]nan,[27]nan,[28]nan,[29]nan,[30]nan,[31]nan,[32]nan,[33]nan,[34]nan,[35]nan,[36]nan,
```

</details>