## 🗣️ [Discussion #657](https://github.com/ikawrakow/ik_llama.cpp/discussions/657) - Intel Auto-round quants - how do they compare?

| **Author** | `Thireus` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-27 |
| **Updated** | 2025-07-27 |

---

## 📄 Description

Has anyone looked into https://github.com/intel/auto-round? I just saw it show up on my feed. Looks like they've been cooking for a little while. Has anyone tested their quants? I see they've recently added recipes for DeepSeek-R1-0528 - not too sure where to find those and how to evaluate them though - https://github.com/intel/auto-round/releases/tag/v0.6.0

Cc @ubergarm 

For ref: https://x.com/haihaoshen/status/1948610166573990236 - "Intel AutoRound v0.6 released, featuring blocking scale quantization and model export to mainstream formats including GGUF, AWQ, GPTQ etc."

---

## 💬 Discussion

👤 **saood06** commented on **2025-07-27** at **06:19:33**

>I see they've recently added recipes for DeepSeek-R1-0528 - not too sure where to find those

[Here](https://huggingface.co/Intel/DeepSeek-R1-0528-q2ks-mixed-AutoRound-inc-v1) and [bonus Qwen-235B](https://huggingface.co/Intel/Qwen3-235B-A22B-q2ks-mixed-AutoRound-inc-v1).

> and how to evaluate them though

PPL would work with these GGUF

---

👤 **ikawrakow** commented on **2025-07-27** at **06:30:00**

up/gate experts with q2_k, down experts with q4_k, shared experts with q4_k, attention with a mix of q4_k and q5_0. Expect it to be far away from the Pareto frontier.

> 👤 **saood06** replied on **2025-07-27** at **06:37:52**
> 
> >up/gate experts with q2_k, down experts with q4_k, shared experts with q4_k, attention with a mix of q4_k and q5_0. Expect it to be far away from the Pareto frontier.
> 
> The specialty isn't the recipe. It's the fact that they are exported from a finetuning process. It is a QAT quant.
> 
> >It leverages sign gradient descent to fine-tune both rounding values and min-max clipping thresholds in just 200 steps.
> 
> They have a paper (https://arxiv.org/pdf/2309.05516) haven't read yet.

> 👤 **ikawrakow** replied on **2025-07-27** at **06:39:03**
> 
> Oh, only the first few layers of ffn_down_exps are q4_k, the others are q2_k

> 👤 **ikawrakow** replied on **2025-07-27** at **06:59:37**
> 
> The whole point of better quantization approaches is to avoid fine tuning. They fine tune using The Pile, so who knows how useful that is for your use case. And, in the usual style of quantization papers, select only non SOTA methods for comparison.

> 👤 **saood06** replied on **2025-07-27** at **07:31:38**
> 
> Their approach to me sounds in the same ballpark as your optimization here:
> 
> >Typically this is just round-to-nearest from a super-block or tensor row scale. I decided to see what happens if I also check the nearest integer values for a block scale, and pick the integer value that minimizes RMSE (changing the block scales can change the quant values, which can sometimes result in a lower difference to the original model weights). This did give a small but non-negligible improvement for IQ2_KL. So, today I decided to see if the same trick can be applied to other quantization types, and the PR includes changes to those types where it helped.
> 
> They say
> >It leverages sign gradient descent to fine-tune both rounding values and min-max clipping thresholds in just 200 steps.
> 
> Am I wrong?
> 
> >They fine tune using The Pile, so who knows how useful that is for your use case. And, in the usual style of quantization papers, select only non SOTA methods for comparison.
> 
> I agree people should evaluate if they are good for their usecases. I'm not advocating for it, just saying what I think they are trying to present. People liked the Gemma QAT (but that was released by google) and PPL revealed they were very different then the original as they had lower PPL at Q4_0 then the BF16.

> 👤 **ikawrakow** replied on **2025-07-27** at **15:46:25**
> 
> > Am I wrong?
> 
> Yes, totally.
> 
> You don't see the difference between matching the model weights as closely as possible, and matching the activations that the model produces **on a given calibration set**, using all quantized model weights as free parameters and hence being totally prone to overfitting? It is a bit like comparing a linear fit on 200 data points (two free parameters, so overfitting virtually impossible), to a 200th order polynomial fit (so overfitting basically guaranteed), and saying, "Well, these two are basically the same as both use regression."

---

👤 **Thireus** commented on **2025-07-27** at **09:02:41**

Nothing out of the ordinary so far... I'll be computing the ppl next.
```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 ~/ik_llama-main-b3913-a6db9c4-bin-win-cuda-12.8-x64-avx512/llama-cli -m DeepSeek-R1-0528-hf-256x20B-Q2_K_S-00001-of-00005.gguf  -mla 3 -fa   -amb 1024   -fmoe   -ctk f16   -c 16384   -ngl 99   -ot "blk\.(3|4|5|6)\.ffn_.*=CUDA0" -ot "blk\.(7|8|9)\.ffn_.*=CUDA1" -ot "blk\.(10|11|12)\.ffn_.*=CUDA2"   -ot exps=CPU   -b 4096 -ub 4096   --warmup-batch   --no-mmap   --threads 36   --main-gpu 0   -p '<｜begin▁of▁sentence｜><｜User｜>What is the solution of x+5=-2?<｜Assistant｜><think>\n'
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 3 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
Log start
main: build = 1 (a6db9c4)
main: built with MSVC 19.44.35211.0 for
main: seed  = 1753606629
llama_model_loader: Max stdio successfully set to 2048
llama_model_loader: additional 4 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 49 key-value pairs and 1028 tensors from DeepSeek-R1-0528-hf-256x20B-Q2_K_S-00001-of-00005.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek-R1-0528-hf
llama_model_loader: - kv   3:                         general.size_label str              = 256x20B
llama_model_loader: - kv   4:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   5:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   6:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   7:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv   8:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv   9:          deepseek2.attention.head_count_kv u32              = 1
llama_model_loader: - kv  10:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  11: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  12:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  13:             deepseek2.attention.key_length u32              = 576
llama_model_loader: - kv  14:           deepseek2.attention.value_length u32              = 512
llama_model_loader: - kv  15:                          general.file_type u32              = 21
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:         deepseek2.attention.key_length_mla u32              = 192
llama_model_loader: - kv  21:       deepseek2.attention.value_length_mla u32              = 128
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
llama_model_loader: - kv  33:               general.quantization_version u32              = 2
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...  
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  43:               tokenizer.ggml.add_sep_token bool             = false
llama_model_loader: - kv  44:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  45:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...  
llama_model_loader: - kv  46:                                   split.no u16              = 0
llama_model_loader: - kv  47:                        split.tensors.count i32              = 1028
llama_model_loader: - kv  48:                                split.count u16              = 5
llama_model_loader: - type  f32:  303 tensors
llama_model_loader: - type q5_0:  122 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type q2_K:  174 tensors
llama_model_loader: - type q4_K:  427 tensors
==========================================================================
Detected incompatible DeepSeek model.
Will try to fix, but there are no guarantees

*** Your prompt processing speed will be crippled ***

Consider making your own ik_llama.cpp compatible model or
ask the model provider to make one for you,
==========================================================================
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
llm_load_print_meta: model ftype      = Q2_K - Small
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 210.122 GiB (2.690 BPW)
llm_load_print_meta: repeating layers = 208.288 GiB (2.674 BPW, 669.173 B parameters)
llm_load_print_meta: general.name     = DeepSeek-R1-0528-hf
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
llm_load_tensors: ggml ctx size =    1.70 MiB
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
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_up_shexp.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_up_shexp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_up_shexp.weight buffer type overriden to CUDA2
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
llm_load_tensors:        CPU buffer size = 169344.00 MiB
llm_load_tensors:  CUDA_Host buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 17800.56 MiB
llm_load_tensors:      CUDA1 buffer size = 13204.08 MiB
llm_load_tensors:      CUDA2 buffer size = 13876.97 MiB
....................................................................................................
============ llm_prepare_mla: need to compute 61 wkv_b tensors
Computed blk.0.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.1.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.2.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.3.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.4.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.5.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.6.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.7.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.8.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.9.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.10.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.11.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.12.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.13.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.14.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.15.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.16.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.17.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.18.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.19.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.20.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.21.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.22.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.23.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.24.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.25.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.26.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.27.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.28.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.29.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.30.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.31.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.32.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.33.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.34.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.35.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.36.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.37.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.38.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.39.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.40.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.41.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.42.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.43.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.44.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.45.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.46.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.47.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.48.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.49.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.50.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.51.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.52.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.53.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.54.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.55.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.56.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.57.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.58.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.59.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.60.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
llama_new_context_with_model: n_ctx      = 16384
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   450.00 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =   342.00 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =   306.00 MiB
llama_new_context_with_model: KV self size  = 1098.00 MiB, c^KV (f16): 1098.00 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.49 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  3768.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  3152.02 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  3152.03 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   368.05 MiB
llama_new_context_with_model: graph nodes  = 4100
llama_new_context_with_model: graph splits = 148

system_info: n_threads = 36 / 36 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
        xtc_probability = 0.000, xtc_threshold = 1.000, top_n_sigma = 0.000
sampling order:
CFG -> Penalties -> dry -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> xtc -> top_n_sigma -> temperature
generate: n_ctx = 16384, n_batch = 4096, n_predict = -1, n_keep = 1


<|begin?of?sentence|><|User|>What is the solution of x+5=-2?<|Assistant|><think>
We are given the equation: x + 5 = -2
 To solve for x, we need to isolate x on one side of the equation.
 Currently, 5 is added to x. So, we can subtract 5 from both sides to cancel it out.
 Let's do that:

 x + 5 - 5 = -2 - 5

 Simplifying both sides:
 x = -7

 Therefore, the solution is x = -7.
 We can check by substituting back: (-7) + 5 = -2, which is correct.
</think>
To solve the equation \(x + 5 = -2\), follow these steps:

1. **Isolate the variable \(x\)** by moving the constant term to the other side. Subtract 5 from both sides of the equation:
   \[
   x + 5 - 5 = -2 - 5
   \]

2. **Simplify** both sides:
   \[
   x = -7
   \]

**Verification**:  
Substitute \(x = -7\) back into the original equation:
\[
(-7) + 5 = -2
\]
\[
-2 = -2 \quad \text{(True)}

**Final Solution**:  
The solution is \(x = -7\). [end of text]

llama_print_timings:        load time =  164265.25 ms
llama_print_timings:      sample time =      49.54 ms /   269 runs   (    0.18 ms per token,  5430.39 tokens per second)
llama_print_timings: prompt eval time =   40783.65 ms /    32 tokens ( 1274.49 ms per token,     0.78 tokens per second)
llama_print_timings:        eval time =   45260.75 ms /   268 runs   (  168.88 ms per token,     5.92 tokens per second)
llama_print_timings:       total time =   86202.17 ms /   300 tokens
```

---

👤 **Thireus** commented on **2025-07-27** at **09:43:58**

Unable to calculate the ppl. Not sure what's going on here with these "nan"...

```
$ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1 ~/ik_llama-main-b3913-a6db9c4-bin-win-cuda-12.8-x64-avx512/llama-perplexity -m DeepSeek-R1-0528-hf-256x20B-Q2_K_S-00001-of-00005.gguf  -mla 3 -fa   -amb 1024   -fmoe   -ctk f16   -c 512   -ngl 99   -ot "blk\.(3|4|5|6)\.ffn_.*=CUDA0" -ot "blk\.(7|8|9)\.ffn_.*=CUDA1" -ot "blk\.(10|11|12)\.ffn_.*=CUDA2"   -ot exps=CPU   -b 4096 -ub 4096   --warmup-batch   --no-mmap   --threads 36   --main-gpu 0   --seed 1337   -f ../wiki.test.raw
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 3 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
  Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
  Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
main: build = 1 (a6db9c4)
main: built with MSVC 19.44.35211.0 for
main: seed  = 1337
llama_model_loader: Max stdio successfully set to 2048
llama_model_loader: additional 4 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 49 key-value pairs and 1028 tensors from DeepSeek-R1-0528-hf-256x20B-Q2_K_S-00001-of-00005.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek-R1-0528-hf
llama_model_loader: - kv   3:                         general.size_label str              = 256x20B
llama_model_loader: - kv   4:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   5:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   6:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   7:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv   8:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv   9:          deepseek2.attention.head_count_kv u32              = 1
llama_model_loader: - kv  10:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  11: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  12:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  13:             deepseek2.attention.key_length u32              = 576
llama_model_loader: - kv  14:           deepseek2.attention.value_length u32              = 512
llama_model_loader: - kv  15:                          general.file_type u32              = 21
llama_model_loader: - kv  16:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  17:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  18:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  19:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  20:         deepseek2.attention.key_length_mla u32              = 192
llama_model_loader: - kv  21:       deepseek2.attention.value_length_mla u32              = 128
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
llama_model_loader: - kv  33:               general.quantization_version u32              = 2
llama_model_loader: - kv  34:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  35:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  36:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  37:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...  
llama_model_loader: - kv  38:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  39:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  40:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  41:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  42:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  43:               tokenizer.ggml.add_sep_token bool             = false
llama_model_loader: - kv  44:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  45:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...  
llama_model_loader: - kv  46:                                   split.no u16              = 0
llama_model_loader: - kv  47:                        split.tensors.count i32              = 1028
llama_model_loader: - kv  48:                                split.count u16              = 5
llama_model_loader: - type  f32:  303 tensors
llama_model_loader: - type q5_0:  122 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type q2_K:  174 tensors
llama_model_loader: - type q4_K:  427 tensors
==========================================================================
Detected incompatible DeepSeek model.
Will try to fix, but there are no guarantees

*** Your prompt processing speed will be crippled ***

Consider making your own ik_llama.cpp compatible model or
ask the model provider to make one for you,
==========================================================================
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
llm_load_print_meta: model ftype      = Q2_K - Small
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 210.122 GiB (2.690 BPW)
llm_load_print_meta: repeating layers = 208.288 GiB (2.674 BPW, 669.173 B parameters)
llm_load_print_meta: general.name     = DeepSeek-R1-0528-hf
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
llm_load_tensors: ggml ctx size =    1.70 MiB
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
Tensor blk.7.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.7.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.8.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_norm.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_inp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_gate_shexp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_down_shexp.weight buffer type overriden to CUDA1
Tensor blk.9.ffn_up_shexp.weight buffer type overriden to CUDA1
Tensor blk.10.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.10.ffn_up_shexp.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.11.ffn_up_shexp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_norm.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_inp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_gate_shexp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_down_shexp.weight buffer type overriden to CUDA2
Tensor blk.12.ffn_up_shexp.weight buffer type overriden to CUDA2
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
llm_load_tensors:        CPU buffer size = 169344.00 MiB
llm_load_tensors:  CUDA_Host buffer size =   938.98 MiB
llm_load_tensors:      CUDA0 buffer size = 17800.56 MiB
llm_load_tensors:      CUDA1 buffer size = 13204.08 MiB
llm_load_tensors:      CUDA2 buffer size = 13876.97 MiB
....................................................................................................
============ llm_prepare_mla: need to compute 61 wkv_b tensors
Computed blk.0.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.1.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.2.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.3.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.4.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.5.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.6.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.7.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.8.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.9.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.10.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.11.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.12.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.13.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.14.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.15.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.16.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.17.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.18.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.19.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.20.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.21.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.22.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.23.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.24.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA0
Computed blk.25.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.26.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.27.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.28.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.29.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.30.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.31.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.32.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.33.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.34.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.35.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.36.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.37.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.38.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.39.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.40.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.41.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.42.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.43.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA1
Computed blk.44.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.45.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.46.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.47.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.48.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.49.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.50.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.51.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.52.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.53.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.54.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.55.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.56.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.57.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.58.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.59.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
Computed blk.60.attn_kv_b.weight as 512 x 32768 and stored in buffer CUDA2
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 3
llama_new_context_with_model: attn_max_b = 1024
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 0.025
llama_kv_cache_init:      CUDA0 KV buffer size =   112.50 MiB
llama_kv_cache_init:      CUDA1 KV buffer size =    85.50 MiB
llama_kv_cache_init:      CUDA2 KV buffer size =    76.50 MiB
llama_new_context_with_model: KV self size  =  274.50 MiB, c^KV (f16):  274.50 MiB, kv^T: not used
llama_new_context_with_model:  CUDA_Host  output buffer size =     3.95 MiB
llama_new_context_with_model: pipeline parallelism enabled (n_copies=1)
llama_new_context_with_model:      CUDA0 compute buffer size =  2928.02 MiB
llama_new_context_with_model:      CUDA1 compute buffer size =  1520.02 MiB
llama_new_context_with_model:      CUDA2 compute buffer size =  2132.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   176.05 MiB
llama_new_context_with_model: graph nodes  = 3429
llama_new_context_with_model: graph splits = 148

system_info: n_threads = 36 / 36 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
perplexity: tokenizing the input ..
perplexity: tokenization took 2345.47 ms
perplexity: calculating perplexity over 561 chunks, n_ctx=512, batch_size=4096, n_seq=8
perplexity: 31.28 seconds per pass - ETA 36.55 minutes
[1]2.8305,[2]3.5614,[3]2.5145,[4]2.1628,[5]1.9815,[6]1.8641,[7]1.7562,[8]1.7317,[9]1.6971,[10]1.6402,[11]1.6444,[12]1.7140,[13]1.7300,[14]1.8530,[15]1.9970,[16]2.0412,[17]2.2041,[18]2.3299,[19]2.2869,[20]2.2785,[21]2.3840,[22]2.3449,[23]2.3116,[24]2.3265,[25]2.2957,[26]2.2651,[27]2.3121,[28]2.3221,[29]2.3734,[30]2.4055,[31]2.4373,[32]2.4521,[33]2.4925,[34]2.5461,[35]2.6019,[36]2.6539,[37]2.6869,[38]2.7360,[39]-nan,[40]-nan,[41]-nan,[42]-nan,[43]-nan,[44]-nan,[45]-nan,[46]-nan,[47]-nan,[48]-nan,[49]-nan,[50]-nan,[51]-nan,[52]-nan,[53]-nan,[54]-nan,[55]-nan,[56]-nan,[57]-nan,[58]-nan,[59]-nan,[60]-nan,[61]-nan,[62]-nan,[63]-nan,[64]-nan,[65]-nan,[66]-nan,[67]-nan,[68]-nan,[69]-nan,[70]-nan,[71]-nan,[72]-nan,[73]-nan,[74]-nan,[75]-nan,[76]-nan,[77]-nan,[78]-nan,[79]-nan,[80]-nan,[81]-nan,[82]-nan,[83]-nan,[84]-nan,[85]-nan,[86]-nan,[87]-nan,[88]-nan,[89]-nan,[90]-nan,[91]-nan,[92]-nan,[93]-nan,[94]-nan,[95]-nan,[96]-nan,[97]-nan,[98]-nan,[99]-nan,[100]-nan,[101]-nan,[102]-nan,[103]-nan,[104]-nan,[105]-nan,[106]-nan,[107]-nan,[108]-nan,[109]-nan,[110]-nan,[111]-nan,[112]-nan,[113]-nan,[114]-nan,[115]-nan,[116]-nan,[117]-nan,[118]-nan,[119]-nan,[120]-nan,[121]-nan,[122]-nan,[123]-nan,[124]-nan,[125]-nan,[126]-nan,[127]-nan,[128]-nan,[129]-nan,[130]-nan,[131]-nan,[132]-nan,[133]-nan,[134]-nan,[135]-nan,[136]-nan,[137]-nan,[138]-nan,[139]-nan,[140]-nan,[141]-nan,[142]-nan,[143]-nan,[144]-nan,[145]-nan,[146]-nan,[147]-nan,[148]-nan,[149]-nan,[150]-nan,[151]-nan,[152]-nan,[153]-nan,[154]-nan,[155]-nan,[156]-nan,[157]-nan,[158]-nan,[159]-nan,[160]-nan,[161]-nan,[162]-nan,[163]-nan,[164]-nan,[165]-nan,[166]-nan,[167]-nan,[168]-nan,[169]-nan,[170]-nan,[171]-nan,[172]-nan,[173]-nan,[174]-nan,[175]-nan,[176]-nan,[177]-nan,[178]-nan,[179]-nan,[180]-nan,[181]-nan,[182]-nan,[183]-nan,[184]-nan,[185]-nan,[186]-nan,[187]-nan,[188]-nan,[189]-nan,[190]-nan,[191]-nan,[192]-nan,[193]-nan,[194]-nan,[195]-nan,[196]-nan,[197]-nan,[198]-nan,[199]-nan,[200]-nan,[201]-nan,[202]-nan,[203]-nan,[204]-nan,[205]-nan,[206]-nan,[207]-nan,[208]-nan,[209]-nan,[210]-nan,[211]-nan,[212]-nan,[213]-nan,[214]-nan,[215]-nan,[216]-nan,[217]-nan,[218]-nan,[219]-nan,[220]-nan,[221]-nan,[222]-nan,[223]-nan,[224]-nan,[225]-nan,[226]-nan,[227]-nan,[228]-nan,[229]-nan,[230]-nan,[231]-nan,[232]-nan,[233]-nan,[234]-nan,[235]-nan,[236]-nan,[237]-nan,[238]-nan,[239]-nan,[240]-nan,[241]-nan,[242]-nan,[243]-nan,[244]-nan,[245]-nan,[246]-nan,[247]-nan,[248]-nan,[249]-nan,[250]-nan,[251]-nan,[252]-nan,[253]-nan,[254]-nan,[255]-nan,[256]-nan,[257]-nan,[258]-nan,[259]-nan,[260]-nan,[261]-nan,[262]-nan,[263]-nan,[264]-nan,[265]-nan,[266]-nan,[267]-nan,[268]-nan,[269]-nan,[270]-nan,[271]-nan,[272]-nan,[273]-nan,[274]-nan,[275]-nan,[276]-nan,[277]-nan,[278]-nan,[279]-nan,[280]-nan,[281]-nan,[282]-nan,[283]-nan,[284]-nan,[285]-nan,[286]-nan,[287]-nan,[288]-nan,[289]-nan,[290]-nan,[291]-nan,[292]-nan,[293]-nan,[294]-nan,[295]-nan,[296]-nan,[297]-nan,[298]-nan,[299]-nan,[300]-nan,[301]-nan,[302]-nan,[303]-nan,[304]-nan,[305]-nan,[306]-nan,[307]-nan,[308]-nan,[309]-nan,[310]-nan,[311]-nan,[312]-nan,[313]-nan,[314]-nan,[315]-nan,[316]-nan,[317]-nan,[318]-nan,[319]-nan,[320]-nan,[321]-nan,[322]-nan,[323]-nan,[324]-nan,[325]-nan,[326]-nan,[327]-nan,[328]-nan,[329]-nan,[330]-nan,[331]-nan,[332]-nan,[333]-nan,[334]-nan,[335]-nan,[336]-nan,[337]-nan,[338]-nan,[339]-nan,[340]-nan,[341]-nan,[342]-nan,[343]-nan,[344]-nan,[345]-nan,[346]-nan,[347]-nan,[348]-nan,[349]-nan,[350]-nan,[351]-nan,[352]-nan,[353]-nan,[354]-nan,[355]-nan,[356]-nan,[357]-nan,[358]-nan,[359]-nan,[360]-nan,[361]-nan,[362]-nan,[363]-nan,[364]-nan,[365]-nan,[366]-nan,[367]-nan,[368]-nan,[369]-nan,[370]-nan,[371]-nan,[372]-nan,[373]-nan,[374]-nan,[375]-nan,[376]-nan,
```

> 👤 **ikawrakow** replied on **2025-07-27** at **15:48:21**
> 
> Does it run with `llama.cpp` without producing NaNs?