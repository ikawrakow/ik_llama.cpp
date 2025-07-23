### 🔀 [#291](https://github.com/ikawrakow/ik_llama.cpp/pull/291) - Disable Zen4 optimizations for Q8_0/Q8_0_R8

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-26 |
| **Updated** | 2025-03-27 |

---

#### Description

The purpose of this PR is to test if the NaNs observed for `Q8_0/Q8_0_R8` quantized DeepSeekV3/R1 will go away (#285)

My hypothesis is that we get an overflow in the block sum of `Q8_1/Q8_1_X4`, which is stored as `fp16`. `Q8_1/Q8_1_X4` is used for activation quantization on Zen4 for `Q8_0/Q8_0_R8` quants. See also #196 
  
The PR disables the Zen4 optimization and reverts to the vanilla `AVX2` implementation, which uses `Q8_0` (just like mainline `llama.cpp`).

Performance goes down quite a bit, but if we confirm that the change eliminates the NaNs, I will make a better PR that keeps the performance while avoiding the NaNs.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-03-26** at **18:52:28**:<br>

Finished successfully, just updated logs. Thanks!

---

👤 **ubergarm** commented the **2025-03-26** at **19:28:51**:<br>

Oh nice, seems like with this patch I'm also able to get an imatrix going with MLA tensors on the `V3-0324` `q8_0` gguf I recently made.  Letting that cook, here is partial outputs for now :point_down: 

<details>

<summary>llama-imatrix run on q8_0</summary>

```bash
$ git rev-parse --short HEAD
2089147a

$ numactl -N 1 -m 1 \
./build/bin/llama-imatrix \
    --verbosity 1 \
    -m /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0.gguf \
    -f calibration_data_v5_rc.txt \
    -o /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-$(git rev-parse --short HEAD).dat \
    --ctx-size 512 \
    --numa numactl \
    --threads 128 2>&1 | tee -a output.log

llama_model_loader: loaded meta data with 46 key-value pairs and 1147 tensors from /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-Q8_0.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 0324
llama_model_loader: - kv   3:                            general.version str              = V3-0324
llama_model_loader: - kv   4:                           general.basename str              = DeepSeek
llama_model_loader: - kv   5:                         general.size_label str              = 256x21B
llama_model_loader: - kv   6:                            general.license str              = mit
llama_model_loader: - kv   7:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   8:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   9:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  10:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  11:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  12:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  13:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  14: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  16:                          general.file_type u32              = 7
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
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  43:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  44:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  45:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q8_0:  786 tensors
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
llm_load_print_meta: model ftype      = Q8_0
llm_load_print_meta: model params     = 672.050 B
llm_load_print_meta: model size       = 665.308 GiB (8.504 BPW) 
llm_load_print_meta: repeating layers = 663.474 GiB (8.504 BPW, 670.196 B parameters)
llm_load_print_meta: general.name     = DeepSeek V3 0324
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
llm_load_tensors: ggml ctx size =    0.47 MiB
llm_load_tensors:        CPU buffer size = 681274.97 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:        CPU KV buffer size =  2440.00 MiB
llama_new_context_with_model: KV self size  = 2440.00 MiB, K (f16): 1464.00 MiB, V (f16):  976.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =   283.01 MiB
llama_new_context_with_model: graph nodes  = 3724
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 128 / 512 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
compute_imatrix: tokenizing the input ..
compute_imatrix: tokenization took 313.289 ms
compute_imatrix: computing over 213 chunks with batch_size 512
compute_imatrix: 41.77 seconds per pass - ETA 2 hours 28.28 minutes
[1]60.9029,[2]10.8011,[3]5.8709,[4]3.7872,[5]2.9688,[6]2.5088,[7]2.2214,[8]2.0224,[9]1.9110,
save_imatrix: entry '             blk.60.ffn_down_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '             blk.60.ffn_gate_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**
save_imatrix: entry '               blk.60.ffn_up_exps.weight' has partial data (99.61%) 1 out of 256 experts are missing data Storing **but be aware**

save_imatrix: stored collected data after 10 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat
[10]1.8230,[11]2.0314,[12]2.0866,[13]2.1000,[14]2.1455,[15]2.0412,[16]1.9535,[17]1.8827,[18]1.8197,[19]1.7778,
save_imatrix: stored collected data after 20 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat
[20]1.7349,[21]1.7018,[22]1.6640,[23]1.6347,[24]1.6222,[25]1.6104,[26]1.5849,[27]1.6838,[28]1.7577,[29]1.8237,
save_imatrix: stored collected data after 30 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat
[30]1.8219,[31]1.8354,[32]1.8351,[33]1.8125,[34]1.8489,[35]1.8250,[36]1.8245,[37]1.8131,[38]1.8239,[39]1.8108,
save_imatrix: stored collected data after 40 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat
[40]1.7876,[41]1.7643,[42]1.7444,[43]1.7325,[44]1.7193,[45]1.7059,[46]1.7016,[47]1.6954,[48]1.6846,[49]1.6741,
save_imatrix: stored collected data after 50 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat
[50]1.6684,[51]1.6656,[52]1.6657,[53]1.6704,[54]1.6844,[55]1.6811,[56]1.6712,[57]1.6794,[58]1.6833,[59]1.6943,
save_imatrix: stored collected data after 60 chunks in /mnt/ai/models/ubergarm/DeepSeek-V3-0324-GGUF/imatrix-ubergarm-DeepSeek-V3-0324-ik_llamacpp-2089147a.dat

*WIP - still cookin'*
```

</details>

---

👤 **ikawrakow** commented the **2025-03-27** at **04:49:39**:<br>

Close in favor of #292