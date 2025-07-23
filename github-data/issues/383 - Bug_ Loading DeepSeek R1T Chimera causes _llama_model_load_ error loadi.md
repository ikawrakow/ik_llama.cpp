### 🐛 [#383](https://github.com/ikawrakow/ik_llama.cpp/issues/383) - Bug: Loading DeepSeek R1T Chimera causes \"llama_model_load: error loading model: check_tensor_dims: tensor 'blk.0.attn_q_b.weight' has wrong shape; expected  1536, 73728, got  1536, 24576,     1,     1\"

| **Author** | `Alexey-Akishin` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-06 |
| **Updated** | 2025-06-01 |

---

#### Description

### What happened?

I tried loading https://huggingface.co/bullerwins/DeepSeek-R1T-Chimera-GGUF/tree/main/DeepSeek-R1T-Chimera-Q4_K_M and get this error (the same model loads fine with llama.cpp):

```
llama_model_load: error loading model: check_tensor_dims: tensor 'blk.0.attn_q_b.weight' has wrong shape; expected  1536, 73728, got  1536, 24576,     1,     1
```

Original model: https://huggingface.co/tngtech/DeepSeek-R1T-Chimera

It is a merge of DeepSeek-R1 and DeepSeek-V3 (0324). It is quite well made too, bringing together good qualities of both models, not sure though why it fails in ik_llama.cpp. At first I tried to run repacked model with llama-quantize, but then I also tried to run the original quant, I also tried with or without -rtr and CPU-only without any cache quantization and without flash attention (just specifying ctx-size and model to load), with the same outcome unfortunately.

### Name and Version

version: 3667 (e3fec173)

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1T Chimera Bf16
llama_model_loader: - kv   3:                         general.size_label str              = 256x20B
llama_model_loader: - kv   4:                            general.license str              = mit
llama_model_loader: - kv   5:                   general.base_model.count u32              = 2
llama_model_loader: - kv   6:                  general.base_model.0.name str              = DeepSeek V3 0324
llama_model_loader: - kv   7:               general.base_model.0.version str              = V3-0324
llama_model_loader: - kv   8:          general.base_model.0.organization str              = Deepseek Ai
llama_model_loader: - kv   9:              general.base_model.0.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv  10:                  general.base_model.1.name str              = DeepSeek R1
llama_model_loader: - kv  11:          general.base_model.1.organization str              = Deepseek Ai
llama_model_loader: - kv  12:              general.base_model.1.repo_url str              = https://huggingface.co/deepseek-ai/De...
llama_model_loader: - kv  13:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv  14:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv  15:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv  16:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv  17:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv  18:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv  19:          deepseek2.attention.head_count_kv u32              = 1
llama_model_loader: - kv  20:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  21: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  22:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  23:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  24:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  25:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  26:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  27:             deepseek2.attention.key_length u32              = 576
llama_model_loader: - kv  28:           deepseek2.attention.value_length u32              = 512
llama_model_loader: - kv  29:         deepseek2.attention.key_length_mla u32              = 192
llama_model_loader: - kv  30:       deepseek2.attention.value_length_mla u32              = 128
llama_model_loader: - kv  31:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  32:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  33:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  34:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  35:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  36:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  37:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  38:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  39:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  40: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  41: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  42:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  43:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  44:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  45:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  46:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  47:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  48:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  49:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  50:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  51:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  52:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  53:               general.quantization_version u32              = 2
llama_model_loader: - kv  54:                          general.file_type u32              = 214
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q5_0:   61 tensors
llama_model_loader: - type q4_K:  467 tensors
llama_model_loader: - type q6_K:   31 tensors
llama_model_loader: - type q4_k_r4:  139 tensors
llama_model_loader: - type q6_k_r4:   27 tensors
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
llm_load_print_meta: n_head_kv        = 1
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 576
llm_load_print_meta: n_embd_head_v    = 512
llm_load_print_meta: n_gqa            = 128
llm_load_print_meta: n_embd_k_gqa     = 576
llm_load_print_meta: n_embd_v_gqa     = 512
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
llm_load_print_meta: model ftype      = Q4_K_R4
llm_load_print_meta: model params     = 671.026 B
llm_load_print_meta: model size       = 376.710 GiB (4.822 BPW) 
llm_load_print_meta: repeating layers = 375.516 GiB (4.820 BPW, 669.173 B parameters)
llm_load_print_meta: general.name     = DeepSeek R1T Chimera Bf16
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
llm_load_tensors: ggml ctx size =    2.23 MiB
llama_model_load: error loading model: check_tensor_dims: tensor 'blk.0.attn_q_b.weight' has wrong shape; expected  1536, 73728, got  1536, 24576,     1,     1
llama_load_model_from_file: failed to load model
```

---

#### 💬 Conversation

👤 **saood06** commented the **2025-05-06** at **00:29:56**:<br>

The reason you are seeing an error is the MLA implementation here and in mainline is no longer compatible, and the model linked is using the incompatible MLA implementation. We support creating the MLA tensors on the fly for models that existed before the MLA implementation or models that are converted using convert_hf_to_gguf.py from this repo, where it will add the MLA tensors used here.

If you want to use the model you can by directly converting from https://huggingface.co/tngtech/DeepSeek-R1T-Chimera (although using this https://huggingface.co/bullerwins/DeepSeek-R1T-Chimera-bf16/tree/main may be easier (at the cost of more download [file is double the size but using xet you might be able to transfer a similar amount of data]) as otherwise you will have to first convert to bf16.

---

👤 **saood06** commented the **2025-05-06** at **00:29:56**:<br>

The MLA implementation here and in mainline is no longer compatible. We support creating the MLA tensors on the fly for models that existed before the MLA implementation or models that are converted using convert_hf_to_gguf.py from this repo, where it will add the MLA tensors used here.

---

👤 **Alexey-Akishin** commented the **2025-05-06** at **00:59:06**:<br>

Oh, I see. Is there a way to somehow salvage https://huggingface.co/bullerwins/DeepSeek-R1T-Chimera-GGUF/tree/main/DeepSeek-R1T-Chimera-Q4_K_M quant, either remove or convert incompatible MLA tensors? Maybe there is a way to upconvert it to bf16 and then from there to the quant I need, or that wouldn't work? Unfortunately no other quants on huggingface exist.

https://huggingface.co/tngtech/DeepSeek-R1T-Chimera seems to have 163 files, mostly 4.3 GB in size, so about 700GB or half a month of downloading non-stop in my case, or maybe two months if I get speed limited for the rest of the month (since I already made multiple downloads this month and have 1TB traffic limit per month before speed is limited).

If nothing can be done and it is not a bug, I understand, but I suggest considering adding a clear error message, so it would be easier for users to understand that they are trying to run incompatible quant.

---

👤 **Alexey-Akishin** commented the **2025-05-06** at **00:59:06**:<br>

Oh, I see. https://huggingface.co/tngtech/DeepSeek-R1T-Chimera seems 163 files, mostly 4.3 GB in size, so about 700GB or half a month of downloading non-stop in my case, or maybe two months if I get speed limited for the rest of the month (since I already made multiple downloads this months and have only 1TB traffic limit before speed is limited).

Is there a way to somehow salvage https://huggingface.co/bullerwins/DeepSeek-R1T-Chimera-GGUF/tree/main/DeepSeek-R1T-Chimera-Q4_K_M quant, either remove or convert incompatible MLA tensors? Maybe upconvert it to bf16 and then from there to the quant I need, or that wouldn't work? Unfortunately no other quants on huggingface exist.

If nothing can be done and it is not a bug, I understand, but I suggest considering adding a clear error message, so it would be easier for users to understand that they are trying to run incompatible quant.

---

👤 **saood06** commented the **2025-05-06** at **01:14:23**:<br>

> Oh, I see. Is there a way to somehow salvage https://huggingface.co/bullerwins/DeepSeek-R1T-Chimera-GGUF/tree/main/DeepSeek-R1T-Chimera-Q4_K_M quant, either remove or convert incompatible MLA tensors? 

You probably could make a script that does that (I have been meaning to make one that merges my V3 and R1 GGUF in the same way chimera does to avoid downloading it since as you know these models are large).

>Maybe there is a way to upconvert it to bf16 and then from there to the quant I need, or that wouldn't work?

Converting to bf16 is needed before making a GGUF, what you are suggesting has been done (how the leaked quantized Miqu GGUF was turned back into a safetensor), but is not relevant to you.

>Unfortunately no other quants on huggingface exist.

I know I looked as well.

> If nothing can be done and it is not a bug, I understand, but I suggest considering adding a clear error message, so it would be easier for users to understand that they are trying to run incompatible quant.

We may end up doing that, I know for now the README for this repo mentions it saying:

>The new GGUFs for DeepSeek-V3/R1/Lite do not work in this repository. This is due to the backwards incompatibe change in mainline llama.cpp that https://github.com/ggml-org/llama.cpp/pull/12801 2.5 months after MLA was available here, and worked with the original DeepSeek GGUFs. Please use the original GGUF or, if you don't have one, convert the HF safetnosrs using the Python conversion scrip in this repository.

---

👤 **Lissanro** commented the **2025-05-06** at **05:08:04**:<br>

I downloaded the same not compatible quant few days ago, but seeing this bug report inspired me to create a request for the quant creator to consider create one that is compatible with ik_llama.cpp https://huggingface.co/bullerwins/DeepSeek-R1T-Chimera-GGUF/discussions/1 (I figured if it is not just me who needs it, maybe they will consider it). I am yet to download full version of it to make my own quant (I will not be able to upload it though, since I have less than 1 Mbps for upload but around 10-40 Mbps for download).

I ran some tests using llama.cpp and noticed that llama.cpp has a bug that makes it produce gibberish unless CUDA is disabled - https://github.com/ggml-org/llama.cpp/issues/13327 (my guess though it may not apply to ik_llama.cpp since probably caused by their MLA implementation). I thought I mention this just in case someone testing with llama.cpp and ik_llama.cpp.

Given how much ik_llama.cpp implementation is more mature and faster (by more than two times), it is surprising to me that people create so many quants specific to llama.cpp, but there are very few ones specific to ik_llama.cpp. But in the meantime, it looks like downloading full version and creating own GGUF quant is the only choice.

---

👤 **saood06** commented the **2025-05-06** at **05:37:18**:<br>

> I downloaded the same not compatible quant few days ago, but seeing this bug report inspired me to create a request for the quant creator to consider create one that is compatible with ik_llama.cpp https://huggingface.co/bullerwins/DeepSeek-R1T-Chimera-GGUF/discussions/1 (I figured if it is not just me who needs it, maybe they will consider it).

I still think a a script somewhat inspired by https://huggingface.co/stduhpf/google-gemma-3-27b-it-qat-q4_0-gguf-small/blob/main/swap_embeds.py could remove the incorrect tensors (and maybe even insert the correct ones if you happen to have a model with the correct ones since for on my machine the on the fly MLA tensors come with a small performance penalty).

I haven't downloaded the Chimera model because I have both V3 and R1, and don't want to waste bandwidth when I could just make a script that combines them, but I haven't been curious enough about the model to do it yet.

>I am yet to download full version of it to make my own quant (I will not be able to upload it though, since I have less than 1 Mbps for upload but around 10-40 Mbps for download).

I also have an asymmetric download/upload rate which is why I also can't really upload something Deepseek sized.

> Given how much ik_llama.cpp implementation is more mature and faster (by more than two times), it is surprising to me that people create so many quants specific to llama.cpp, but there are very few ones specific to ik_llama.cpp. But in the meantime, it looks like downloading full version and creating own GGUF quant is the only choice.

It is because less people know about and thus use ik_llama.cpp. It also doesn't help that model support here generally comes later (Deepseek's MLA implementation being an exception), and llama.cpp sometimes even gets 0-day support for models.

---

👤 **ikawrakow** commented the **2025-05-06** at **05:38:30**:<br>

> If nothing can be done and it is not a bug, I understand, but I suggest considering adding a clear error message, so it would be easier for users to understand that they are trying to run incompatible quant.

This is why I added the IMPORTANT note on the ik_llama.cpp main page, hoping to prevent at least some users wasting their time and traffic limits downloading a giant incompatible model.

I personally find the approach taken in mainline llama.cpp plain irresponsible. There was no reason to introduce the incompatibility. The tensors necessary for MLA can be created on-the-fly as done here.

---

👤 **saood06** commented the **2025-05-06** at **05:42:15**:<br>

> This is why I added the IMPORTANT note on the ik_llama.cpp main page, hoping to prevent at least some users wasting their time and traffic limits downloading a giant incompatible model.

Minor note, there are some typos in that note: "scrip" and "safetnosrs".

Edit: Thanks for fixing it.

---

👤 **saood06** commented the **2025-05-06** at **05:42:15**:<br>

> This is why I added the IMPORTANT note on the ik_llama.cpp main page, hoping to prevent at least some users wasting their time and traffic limits downloading a giant incompatible model.

Minor note, there are some typos in that note: "scrip" and "safetnosrs".

---

👤 **Lissanro** commented the **2025-05-06** at **07:37:41**:<br>

> I still think a a script somewhat inspired by https://huggingface.co/stduhpf/google-gemma-3-27b-it-qat-q4_0-gguf-small/blob/main/swap_embeds.py could remove the incorrect tensors

I tested it further, and I think a script will not help in this case. Even though https://huggingface.co/bullerwins/DeepSeek-R1T-Chimera-GGUF/tree/main/DeepSeek-R1T-Chimera-Q4_K_M is similar in size to old Q4_K_M quant of R1 by Unsloth (when they did not have UD XL version of it), the quality is much lower. It failed many tests, most of my tests are specific to my real world use cases, but some are generic public tests or common questions, for example easiest one to check and that reveals quantization degradation in reasoning models very well, is the [maze test](https://www.reddit.com/r/LocalLLaMA/comments/1j4lqe6/test_if_your_api_provider_is_quantizing_your/) - Chimera at OpenRouter passes it, and so does Q4_K_M quant of R1 from Unsloth, but https://huggingface.co/bullerwins/DeepSeek-R1T-Chimera-GGUF/tree/main/DeepSeek-R1T-Chimera-Q4_K_M consistently fails it.

The point is, even if it was possible to somehow recover this Q4 quant to make it work with ik_llama.cpp, its quality is very bad, so it still would be necessary to recreate it from scratch. I guess I just keep downloading the full version via my 4G connection and hope the provider will not limit my speed.

So far, I only created my own repacked quants for ik_llama.cpp, but not from scratch (last time I checked, on the fly conversion was disabling mmap, so I had to repack R1 and V3 quants to use them without performance loss). I know I will need to convert to bf16 first, but I am not yet sure how to create proper quant that would be comparable to UD-Q4_K_XL from Unsloth in quality. I plan to go through some articles Unsloth posted, maybe they shared how they did it.

It may take many days before I have the full Chimera, but if I will figure out a set of commands to convert to a good ik_llama.cpp quant, I will share here (if this discussion is closed by then, then I will just edit my existing message to add the info to avoid reopening it).

---

👤 **Lissanro** commented the **2025-05-06** at **07:37:41**:<br>

> I still think a a script somewhat inspired by https://huggingface.co/stduhpf/google-gemma-3-27b-it-qat-q4_0-gguf-small/blob/main/swap_embeds.py could remove the incorrect tensors

I tested it further, and I think a script will not help in this case. Even though https://huggingface.co/bullerwins/DeepSeek-R1T-Chimera-GGUF/tree/main/DeepSeek-R1T-Chimera-Q4_K_M is similar in size to old Q4_K_M quant of R1 by Unsloth (when they did not have UD or XL versions of it), the quality is much lower. It failed many tests, most of my tests are specific to my real world use cases, but some are generic public tests or common questions, for example easiest one to check and that reveals quantization degradation in reasoning models very well, is the [maze test](https://www.reddit.com/r/LocalLLaMA/comments/1j4lqe6/test_if_your_api_provider_is_quantizing_your/) - Chimera at OpenRouter passes it, and so does Q4_K_M quant of R1 from Unsloth, but https://huggingface.co/bullerwins/DeepSeek-R1T-Chimera-GGUF/tree/main/DeepSeek-R1T-Chimera-Q4_K_M consistently fails it.

The point is, even if it was possible to somehow recover this Q4 quant to make it work with ik_llama.cpp, its quality is very bad, so it still would be necessary to recreate it from scratch. I guess I just keep downloading the full version via my 4G connection and hope the provider will not limit my speed.

So far, I only created my own repacked quants for ik_llama.cpp, but not from scratch (last time I checked, on the fly conversion was disabling mmap, so I had to repack R1 and V3 quants to use them without performance loss). I know I will need to convert to bf16 first, but I am not yet sure how to create proper quant that would be comparable to UD-Q4_K_XL from Unsloth in quality. I plan to go through some articles Unsloth posted, maybe they shared how they did it.

It may take many days before I have the full Chimera, but if I will figure out a set of commands to convert to a good ik_llama.cpp quant, I will share here (if this discussion is closed by then, then I will just edit my existing message to add the info to avoid reopening it).

---

👤 **ikawrakow** commented the **2025-05-06** at **07:54:50**:<br>

>  is similar in size to old Q4_K_M quant of R1 by Unsloth (when they did not have UD XL version of it), the quality is much lower.

This is because the `llama.cpp` experts who decided that breaking backwards compatibility is OK did not consider (or perhaps did not understand?) the implications this breaking change has on quantized models. I'll not explain so that this time they can really independently discover what it is.

---

👤 **saood06** commented the **2025-05-06** at **10:05:33**:<br>

> but I am not yet sure how to create proper quant that would be comparable to UD-Q4_K_XL from Unsloth in quality. I plan to go through some articles Unsloth posted, maybe they shared how they did it.
> 
> It may take many days before I have the full Chimera, but if I will figure out a set of commands to convert to a good ik_llama.cpp quant, I will share here (if this discussion is closed by then, then I will just edit my existing message to add the info to avoid reopening it).

I would recommend actually looking into the quant types that are exclusive to this repo see https://github.com/ikawrakow/ik_llama.cpp/discussions/8 and there is also good discussion in this issue (after the realization that token_embd.weight should not use _r4 quants) about good mixes: https://github.com/ikawrakow/ik_llama.cpp/issues/296

---

👤 **Ph0rk0z** commented the **2025-05-06** at **12:27:36**:<br>

I too want to try this model with it's selective thinking. I rather download it than R1 or V3 alone.

There is also this pruned version: https://huggingface.co/DevQuasar/huihui-ai.DeepSeek-V3-0324-Pruned-Coder-411B-GGUF/tree/main

Deepseek v2.5 should be safe, right? https://huggingface.co/bartowski/DeepSeek-V2.5-1210-GGUF/tree/main and is a similar arch to V3 to benefit from speedups?

---

👤 **city96** commented the **2025-05-08** at **11:41:22**:<br>

@saood06 
> You probably could make a script that does that (I have been meaning to make one that merges my V3 and R1 GGUF in the same way chimera does to avoid downloading it since as you know these models are large).

Not 100% sure if it's correct but I made a script that attempts to do that - [GitHub Gist](https://gist.github.com/city96/a05cb7ec6664a5085efb007497f2049b).

It's based on the discussion on [HuggingFace](https://huggingface.co/tngtech/DeepSeek-R1T-Chimera/discussions/1) which had a reverse-engineered merge recipe for that model. At least for me, it produced a usable checkpoint with my original non-mla gguf files.

---

👤 **saood06** commented the **2025-05-08** at **22:12:47**:<br>

> Not 100% sure if it's correct but I made a script that attempts to do that - [GitHub Gist](https://gist.github.com/city96/a05cb7ec6664a5085efb007497f2049b).
> 
> It's based on the discussion on [HuggingFace](https://huggingface.co/tngtech/DeepSeek-R1T-Chimera/discussions/1) which had a reverse-engineered merge recipe for that model. At least for me, it produced a usable checkpoint with my original non-mla gguf files.

Thank you for this. I saw the beginning of that discussion but I hadn't checked back in to see your reply. At first I thought to use your script on the BF16 versions of the models I have, but I realized I don't see an imatrix of chimera that I would then be able to use, so I might just merge some already quantized (with imatrix) versions I have lying around.

---

👤 **Lissanro** commented the **2025-05-09** at **02:12:03**:<br>

I finally finished downloading unquantized Chimera, but cannot figure out how to convert it to BF16 in order to generate my own quants for ik_llama.cpp. I would greatly appreciate if anybody have any idea how to do it?

So far, I tried using DeepSeek fp8 to BF16 conversion script `fp8_cast_bf16.py`, but it fails with error `type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')`, here is the full log:

```
> cd ~/pkgs/ && git clone https://github.com/deepseek-ai/DeepSeek-V3.git
> python3 ~/pkgs/DeepSeek-V3/inference/fp8_cast_bf16.py --input-fp8-hf-path /mnt/secondary/neuro/DeepSeek-R1T-Chimera-163840seq --output-bf16-hf-path /mnt/secondary/neuro/DeepSeek-R1T-Chimera-BF16-163840seq
  0%|                                                                                                                                                                                                                | 0/163 [00:02<?, ?it/s]
Traceback (most recent call last):
  File "/home/lissanro/pkgs/DeepSeek-V3/inference/fp8_cast_bf16.py", line 111, in <module>
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lissanro/pkgs/DeepSeek-V3/inference/fp8_cast_bf16.py", line 80, in main
    new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                                  ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^
  File "/home/lissanro/pkgs/DeepSeek-V3/inference/kernel.py", line 104, in weight_dequant
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lissanro/.local/lib/python3.13/site-packages/triton/runtime/jit.py", line 330, in <lambda>
    return lambda *args, **kwargs: self.run(grid=grid, warmup=False, *args, **kwargs)
                                   ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lissanro/.local/lib/python3.13/site-packages/triton/runtime/jit.py", line 623, in run
    kernel = self.compile(
        src,
        target=target,
        options=options.__dict__,
    )
  File "/home/lissanro/.local/lib/python3.13/site-packages/triton/compiler/compiler.py", line 273, in compile
    module = src.make_ir(options, codegen_fns, module_map, context)
  File "/home/lissanro/.local/lib/python3.13/site-packages/triton/compiler/compiler.py", line 100, in make_ir
    return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                       module_map=module_map)
triton.compiler.errors.CompilationError: at 1:0:
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
^
ValueError("type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")
```

---

👤 **saood06** commented the **2025-05-09** at **02:20:14**:<br>

>I finally finished downloading unquantized Chimera, but cannot figure out how to convert it to BF16 in order to generate my own quants for ik_llama.cpp. I would greatly appreciate if anybody have any idea how to do it?

The solution I've given others and have used myself is to use [this](https://huggingface.co/daydream-org/DeepSeek-R1-GGUF-11446/discussions/1#67a327570051a98a96ded9e6) method 

I mentioned this before but I'll repeat since I think it still holds true, I've thought about porting that here but the triton dependence adds more complication than I think it is worth for most people, when more fp8 native models are released, I think something along the lines of https://github.com/ggml-org/llama.cpp/pull/10055 is the best path forward.

---

👤 **saood06** commented the **2025-05-09** at **02:20:14**:<br>

>I finally finished downloading unquantized Chimera, but cannot figure out how to convert it to BF16 in order to generate my own quants for ik_llama.cpp. I would greatly appreciate if anybody have any idea how to do it?

The solution I've given others and have used myself is to use this method https://huggingface.co/daydream-org/DeepSeek-R1-GGUF-11446/discussions/1#67a327570051a98a96ded9e6.

I mentioned this before but I'll repeat since I think it still holds true, I've thought about porting that here but the triton dependence adds more complication than I think it is worth for most people, when more fp8 native models are released, I think something along the lines of https://github.com/ggml-org/llama.cpp/pull/10055 is the best path forward.

---

👤 **Lissanro** commented the **2025-05-09** at **05:58:13**:<br>

It seems the tutorial is outdated. Just creating venv on the next step produces errors about not being able to satisfy dependencies, do you know by any chance what Python version was recommended at the time the tutorial was written? On Ubuntu 25.04, Python 3.13 is the default, but it did not work, failing to satisfy some dependencies. So I tried from scratch with older version of Python:

```
conda create -yn venv python=3.12
conda activate venv
```

Instead of these commands:

```
python3 -m venv venv
source venv/bin/activate
```

But then I am stuck at building triton-cpu:

```
> MAX_JOBS=32 pip3 install -e python
Obtaining file:///home/lissanro/pkgs/llama.cpp-fp8-to-bf16/triton-cpu/python
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Requirement already satisfied: setuptools>=40.8.0 in /home/lissanro/.local/lib/python3.12/site-packages (from triton==3.3.0+git0625715c) (75.1.0)
Building wheels for collected packages: triton
  Building editable for triton (pyproject.toml) ... \
```

The last line does not change after some hours and there is no CPU load. If I try to add -vvv, it gets stuck here:

```
 writing top-level names to /tmp/pip-wheel-zyz15gbv/.tmp-kvq7yn4o/triton.egg-info/top_level.txt
  writing manifest file '/tmp/pip-wheel-zyz15gbv/.tmp-kvq7yn4o/triton.egg-info/SOURCES.txt'
  reading manifest file '/tmp/pip-wheel-zyz15gbv/.tmp-kvq7yn4o/triton.egg-info/SOURCES.txt'
  reading manifest template 'MANIFEST.in'
  writing manifest file '/tmp/pip-wheel-zyz15gbv/.tmp-kvq7yn4o/triton.egg-info/SOURCES.txt'
  creating '/tmp/pip-wheel-zyz15gbv/.tmp-kvq7yn4o/triton-3.3.0+git0625715c.dist-info'
  creating /tmp/pip-wheel-zyz15gbv/.tmp-kvq7yn4o/triton-3.3.0+git0625715c.dist-info/WHEEL
  running build_py
  running build_ext
  <string>:304: DeprecationWarning: Python 3.14 will, by default, filter extracted tar archives and reject files or modify their metadata. Use the filter argument to control this behavior.
```

None of directories it mentions in /tmp exist, so I assume it processed them and removed. Warning seems to be harmless in Python 3.12, so I think it is not an issue either. It seems you were right about triton dependency adding complications... I tried to clean everything up, and start over, with the same outcome, or maybe I am doing something wrong, but I tried to follow tutorial steps precisely, except the necessary change to use older Python version.

I am still trying to find some way to convert, but to no avail yet. I tried looking into your second link, but it seems the patch wasn't updated in a while and no longer applies to llama.cpp, I tried few different old commits but could not find one yet where it applies successfully. Maybe I need to try even older llama.cpp commits, but not sure, if I go too far into the past, would it even support DeepSeek V3 architecture to convert to BF16? I also could not find any example command how to convert using https://github.com/ggml-org/llama.cpp/pull/10055 - maybe it is something obvious I missed, perhaps because I never created GGUF before.

I will keep trying to find a solution and if I find one, I will share here. If someone has any ideas or an advice, I would appreciate it greatly.

---

👤 **saood06** commented the **2025-05-09** at **06:34:18**:<br>

> It seems the tutorial is outdated. Just creating venv on the next step produces errors about not being able to satisfy dependencies, do you know by any chance what Python version was recommended at the time the tutorial was written? On Ubuntu 25.04, Python 3.13 is the default, but it did not work, failing to satisfy some dependencies.

I do not, but I know the system where I used triton for this is 3.13.

> It seems you were right about triton dependency adding complications...

I am not happy to be proven right. I ran into some complications myself (but was able to get past them), but up till now I've never had someone I recommended this solution not work for them (which is why I kept recommending it even if I don't think it is the ideal solution). I am really sorry if I wasted your time with something that didn't work for you.

Taking a look at my install `pip list` has:

`triton                        3.2.0+git4ce833eb [local path]`

(more specifically this commit hash 4ce833ebbce7b91564d7cc1f30573eb1129629f9)

Looking at the path it was installed and doing a git diff (since I remember having to change things in order to get it to compile, sorry I normally have full logs but the ones for this session is one of the ones I do not have)

```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index de6ed239..d8cadd8b 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -143,7 +143,7 @@ endfunction()

 # Disable warnings that show up in external code (gtest;pybind11)
 if(NOT MSVC)
-  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default -fvisibility=hidden")
+  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default -fvisibility=hidden")
 else()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4624 /wd4715 /wd4530")
 endif()
diff --git a/third_party/cpu/CMakeLists.txt b/third_party/cpu/CMakeLists.txt
index 25f5b017..615457d1 100644
--- a/third_party/cpu/CMakeLists.txt
+++ b/third_party/cpu/CMakeLists.txt
@@ -1,14 +1,14 @@
 # Find OneDNN ukernel library
-find_package(dnnl CONFIG)
-if (dnnl_FOUND)
-  message(STATUS "Found OneDNN/DNNL")
-  add_compile_definitions(ONEDNN_AVAILABLE)
-  get_target_property(dnnl_include DNNL::dnnl INTERFACE_INCLUDE_DIRECTORIES)
-  # currently used only in triton_cpu.cc and in ConvertDotToOneDNN
-  include_directories(${dnnl_include})
-else ()
-  message(STATUS "Could NOT find OneDNN/DNNL")
-endif()
+#find_package(dnnl CONFIG)
+#if (dnnl_FOUND)
+#  message(STATUS "Found OneDNN/DNNL")
+#  add_compile_definitions(ONEDNN_AVAILABLE)
+#  get_target_property(dnnl_include DNNL::dnnl INTERFACE_INCLUDE_DIRECTORIES)
+#  # currently used only in triton_cpu.cc and in ConvertDotToOneDNN
+#  include_directories(${dnnl_include})
+#else ()
+#  message(STATUS "Could NOT find OneDNN/DNNL")
+#endif()

 # Find XSMM ukernel library
 find_library(LIBXSMM xsmm

```

>I tried looking into your second link, but it seems the patch wasn't updated in a while and no longer applies to llama.cpp, I tried few different old commits but could not find one yet where it applies successfully. Maybe I need to try even older llama.cpp commits, but not sure, if I go too far into the past, would it even support DeepSeek V3 architecture to convert to BF16? I also could not find any example command how to convert using [ggml-org/llama.cpp#10055](https://github.com/ggml-org/llama.cpp/pull/10055) - maybe it is something obvious I missed, perhaps because I never created GGUF before.

I am sorry, I did not link that for you to use, just as a reference to what I see as a better long term solution to the greater issue of handling fp8 native models would be.

> I will keep trying to find a solution and if I find one, I will share here. If someone has any ideas or an advice, I would appreciate it greatly.

If you feel like trying one more time with triton (and no guarantees that it will work), you can try building the commit I was on (with my changes) on 3.13 and see if that works for you?

---

👤 **saood06** commented the **2025-05-09** at **06:34:18**:<br>

> It seems the tutorial is outdated. Just creating venv on the next step produces errors about not being able to satisfy dependencies, do you know by any chance what Python version was recommended at the time the tutorial was written? On Ubuntu 25.04, Python 3.13 is the default, but it did not work, failing to satisfy some dependencies.

I do not, but I know the system where I used triton for this is 3.13.

> It seems you were right about triton dependency adding complications...

I am not happy to be proven right. I ran into some complications myself (but was able to get past them), but up till now I've never had someone I recommended this solution not work for them (which is why I kept recommending it even if I don't think it is the ideal solution). I am really sorry if I wasted your time with something that didn't work for you.

Taking a look at my install `pip list` has:

`triton                        3.2.0+git4ce833eb [local path]`

(more specifically this commit hash 4ce833ebbce7b91564d7cc1f30573eb1129629f9)

Looking at the path it was installed and doing a git diff (since I remember having to change things in order to get it to compile, sorry I normally have full logs of what I do but the ones for this session is one of the ones I do not have)

```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index de6ed239..d8cadd8b 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -143,7 +143,7 @@ endfunction()

 # Disable warnings that show up in external code (gtest;pybind11)
 if(NOT MSVC)
-  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default -fvisibility=hidden")
+  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default -fvisibility=hidden")
 else()
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4624 /wd4715 /wd4530")
 endif()
diff --git a/third_party/cpu/CMakeLists.txt b/third_party/cpu/CMakeLists.txt
index 25f5b017..615457d1 100644
--- a/third_party/cpu/CMakeLists.txt
+++ b/third_party/cpu/CMakeLists.txt
@@ -1,14 +1,14 @@
 # Find OneDNN ukernel library
-find_package(dnnl CONFIG)
-if (dnnl_FOUND)
-  message(STATUS "Found OneDNN/DNNL")
-  add_compile_definitions(ONEDNN_AVAILABLE)
-  get_target_property(dnnl_include DNNL::dnnl INTERFACE_INCLUDE_DIRECTORIES)
-  # currently used only in triton_cpu.cc and in ConvertDotToOneDNN
-  include_directories(${dnnl_include})
-else ()
-  message(STATUS "Could NOT find OneDNN/DNNL")
-endif()
+#find_package(dnnl CONFIG)
+#if (dnnl_FOUND)
+#  message(STATUS "Found OneDNN/DNNL")
+#  add_compile_definitions(ONEDNN_AVAILABLE)
+#  get_target_property(dnnl_include DNNL::dnnl INTERFACE_INCLUDE_DIRECTORIES)
+#  # currently used only in triton_cpu.cc and in ConvertDotToOneDNN
+#  include_directories(${dnnl_include})
+#else ()
+#  message(STATUS "Could NOT find OneDNN/DNNL")
+#endif()

 # Find XSMM ukernel library
 find_library(LIBXSMM xsmm

```

>I tried looking into your second link, but it seems the patch wasn't updated in a while and no longer applies to llama.cpp, I tried few different old commits but could not find one yet where it applies successfully. Maybe I need to try even older llama.cpp commits, but not sure, if I go too far into the past, would it even support DeepSeek V3 architecture to convert to BF16? I also could not find any example command how to convert using [ggml-org/llama.cpp#10055](https://github.com/ggml-org/llama.cpp/pull/10055) - maybe it is something obvious I missed, perhaps because I never created GGUF before.

I am sorry, I did not link that for you to use, just as a reference to what I see as a better long term solution to the greater issue of handling fp8 native models would be.

> I will keep trying to find a solution and if I find one, I will share here. If someone has any ideas or an advice, I would appreciate it greatly.

If you feel like trying one more time with triton (and no guarantees that it will work), you can try building the commit I was on (with my changes) on 3.13 and see if that works for you?

---

👤 **Panchovix** commented the **2025-05-09** at **19:19:58**:<br>

Issue should be fixed now on https://github.com/ikawrakow/ik_llama.cpp/commit/43a154d8b8b0e9217114577442cecb224a488d45

Can confirm you can load deepseek MLA quants with that commit.

EDIT: Can confirm Chimera works fine as well.

---

👤 **Lissanro** commented the **2025-05-11** at **07:05:01**:<br>

@saood06 Thank you, I was able to create BF16 quant after all. I switched to the system version of Python 3.13 without venv, I have applied the patch you shared and also had to bump up torch version in requirements/requirements-convert_hf_to_gguf.txt to torch~=2.5.0, otherwise it refused to proceed on my system. Without venv, I also was able to build triton-cpu. I am not sure exactly what helped out of these steps, so some of them may be unneccary. I finally was able to create BF16 command using this command:

    python3 llama.cpp/convert_hf_to_gguf.py --outtype bf16 --split-max-size 50G /mnt/neuro/DeepSeek-R1T-Chimera-163840seq

...where llama.cpp is the special version from [the tutorial](https://huggingface.co/daydream-org/DeepSeek-R1-GGUF-11446/discussions/1#67a327570051a98a96ded9e6) you have shared earlier.

Then, using ik_llama.cpp, I created my first GGUF quant, using Q8_0 format:

```
~/pkgs/ik_llama.cpp/build/bin/llama-quantize \
/mnt/neuro/DeepSeek-R1T-Chimera-256x21B-163840seq-BF16-00001-of-00030.gguf \
/mnt/neuro/DeepSeek-R1T-Chimera-256x21B-Q8_0-163840seq.gguf \
Q8_0
```

This is usable quant, but it is slow (I get about 2 tokens/s instead of 8 tokens/s like with Q4_K_M or UD-Q4_K_XL). However, I had to consider different solution, given I already know that Q4_K_M breaks the Chimera model (since Q4_K_M from huggingface fails the [maze test](https://www.reddit.com/r/LocalLLaMA/comments/1j4lqe6/test_if_your_api_provider_is_quantizing_your/), while Q8_0 and Q6_K Chimera quant succeed, and R1 Q4 quants from Unsloth also succeed).

It turned out that creation of Dynamic Quants [is not documented yet and active work in progress](https://www.reddit.com/r/LocalLLaMA/comments/1kjshnd/comment/mrpacfb/), so I decided to go with creating IQ and imatrix based quants in the hope they work better than normal Q4_K_M from the huggingface.

This is the command I used to create imatrix.dat:

```
numactl --cpunodebind=0 --interleave=all~/pkgs/ik_llama.cpp/build/bin/llama-imatrix \
--model /mnt/neuro/DeepSeek-R1T-Chimera-256x21B-Q8_0-163840seq.gguf \
--ctx-size 102400 --n-gpu-layers 62 --tensor-split 15,25,30,30 -mla 3 -fa -ctk q8_0 -amb 1024 -fmoe -b 4096 -ub 4096 \
-ot "blk\.3\.ffn_up_exps=CUDA0, blk\.3\.ffn_gate_exps=CUDA0, blk\.3\.ffn_down_exps=CUDA0" \
-ot "blk\.4\.ffn_up_exps=CUDA1, blk\.4\.ffn_gate_exps=CUDA1, blk\.4\.ffn_down_exps=CUDA1" \
-ot "blk\.5\.ffn_up_exps=CUDA2, blk\.5\.ffn_gate_exps=CUDA2, blk\.5\.ffn_down_exps=CUDA2" \
-ot "blk\.6\.ffn_up_exps=CUDA3, blk\.6\.ffn_gate_exps=CUDA3, blk\.6\.ffn_down_exps=CUDA3" \
-ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" \
--threads 64 --host 0.0.0.0 --port 5000
-f ~/pkgs/imatrix/all.txt \
--ctx-size 512
```

Context length optional, but it was mentioned [here](https://github.com/ggml-org/llama.cpp/pull/13199#issuecomment-2849293461) that Unsloth may be setting it to something higher than default 512, "possibly using 6144 - 12288" (later testing demonstrated that making imatrix with non-default context length does not help with long context performance, so if unsure better stick with the default 512 length).

More information about dynamic quant creation is here in comments:
https://www.reddit.com/r/LocalLLaMA/comments/1kjshnd/is_it_possible_to_generate_my_own_dynamic_quant/
But I decided to create a normal quant with imatrix for now (UPDATE: later I tested some custom receipts, but results were worse than default settings, and Unsloth quants, even though are good, also did not prove to be better than the default, or difference was too small to measure in my limited testing).

The all.txt file is a merge of these (I had to convert parquet to txt first):
https://huggingface.co/datasets/eaddario/imatrix-calibration/resolve/main/calibration_all_large.parquet
https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/2c64bb691316d32915b188e495754ef34931ae71/calibration_datav3.txt
https://gist.github.com/bartowski1182/f003237f2e8612278a6d01622af1cb6f/raw/6cf9d7538b3a234952d927459d0ce42cb3d3ea6e/qwen_calibration_with_chat.txt
(also, some personal data, but probably will have little compared to the three datasets above).

I probably could have just used calibration_datav3.txt and nothing else, but calibration_all_large contained many languages that are not well represented in calibration_datav3.txt or qwen_calibration_with_chat.txt, and I happen to need support for multiple languages since I often do translation work.

By the way, I remember a post where someone tested creating imatrix.dat file from BF16, Q8, Q6 and some lower quants, and then creating imatrix quant from BF16 with it, and the conclusion was the result was practically identical, especially if higher quants are used to create the imatrix. I did not save the link to it at the time (it was long before now), but I thought I mention it. This means if you are short on memory, you can use Q6 or even non-imatrix Q4 if you must, but using Q8 is recommended if possible to build the imatrix.dat.

My imatrix: https://dragon.studio/2025/05/DeepSeek-R1T-Chimera-imatrix-8192seq.dat (I renamed it from imatrix.dat for clarity), it took about 12 hours to generate on EPYC 7763 64-core at 3.25 GHz.

Also, here is another imatrix file for recent R1 0528 version: https://dragon.studio/2025/06/imatrix-DeepSeek-R1-0528.dat

Now, we can create the final quant:

```
~/pkgs/ik_llama.cpp/build/bin/llama-quantize \
--imatrix imatrix.dat \
/mnt/neuro/DeepSeek-R1T-Chimera-256x21B-163840seq-BF16-00001-of-00030.gguf \
/mnt/neuro/DeepSeek-R1T-Chimera-256x21B-IQ4_K-163840seq.gguf \
IQ4_K
```

Note: repacking with R4 seems to be no longer needed, and may even reduce performance.

Due to my upload speed being around 1Mbps on average, I will not be able to share any of my quants, but I hope documenting the process will help others who may want to create their own quant. Even once this issue is closed, I still will be able to link here in case I want to share my steps elsewhere, since there was a lot of useful discussion and valuable information shared in this thread.

Performance is good, still getting 8 tokens/s just like with Unsloth's UD-Q4_K_XL quant for R1. 

By the way, I also confirm that loading the existing quant from huggingface works now - so it seems the original issue that was reported is fixed. It is amazing that we can now use new MLA-enabled quants created by llama.cpp, but creating own quant may help to achieve better quality and performance, especially for models with very limited selection of quants like in this case. However, figuring out how to do it was really big challenge, and I wouldn't be able to do it without help. Big thanks to @saood06 and @ikawrakow!

Note: this comment was updated more recently then following messages below. So, if unsure, prefer commands and information shared in this comment, since it is more likely to be recent.

---

👤 **Lissanro** commented the **2025-05-11** at **07:05:01**:<br>

@saood06 Thank you, I was able to create BF16 quant after all. I switched to the system version of Python 3.13 without venv, I have applied the patch you shared and also had to bump up torch version in requirements/requirements-convert_hf_to_gguf.txt to torch~=2.5.0, otherwise it refused to proceed on my system. Without venv, I also was able to build triton-cpu. I am not sure exactly what helped out of these steps, so some of them may be unneccary. I finally was able to create BF16 command using this command:

    python3 llama.cpp/convert_hf_to_gguf.py --outtype bf16 --split-max-size 50G /mnt/secondary/neuro/DeepSeek-R1T-Chimera-163840seq

...where llama.cpp is the special version from [the tutorial](https://huggingface.co/daydream-org/DeepSeek-R1-GGUF-11446/discussions/1#67a327570051a98a96ded9e6) you have shared earlier.

Then, using ik_llama.cpp, I created my first GGUF quant, using Q6_K_R4 format:

```
~/pkgs/ik_llama.cpp/build/bin/llama-quantize \
/mnt/secondary/neuro/DeepSeek-R1T-Chimera-163840seq/DeepSeek-R1T-Chimera-256x21B-163840seq-BF16-00001-of-00030.gguf \
/mnt/secondary/neuro/DeepSeek-R1T-Chimera-163840seq/DeepSeek-R1T-Chimera-256x21B-Q6_K_R4-163840seq.gguf \
Q6_K_R4
```

This is usable quant, but it is slow (I get about 2 tokens/s instead of 8 tokens/s like with Q4_K_M or UD-Q4_K_XL). However, I had to consider different solution, given I already know that Q4_K_M breaks the Chimera model (since Q4_K_M from huggingface fails the [maze test](https://www.reddit.com/r/LocalLLaMA/comments/1j4lqe6/test_if_your_api_provider_is_quantizing_your/), while Q6_K Chimera quant succeeds, and R1 Q4 quants from Unsloth also succeed).

It turned out that creation of Dynamic Quants [is not documented yet and active work in progress](https://www.reddit.com/r/LocalLLaMA/comments/1kjshnd/comment/mrpacfb/), so I decided to go with creating IQ and imatrix based quants in the hope they work better than normal Q4_K_M from the huggingface.

This is the command I used to create imatrix.dat:

```
~/pkgs/ik_llama.cpp/build/bin/llama-imatrix \
-m /mnt/neuro/text-generation-webui/models/DeepSeek-R1T-Chimera-256x21B-Q6_K_R4-163840seq/DeepSeek-R1T-Chimera-256x21B-Q6_K_R4-163840seq.gguf \
-f ~/pkgs/imatrix/all.txt \
--n-gpu-layers 62 --tensor-split 25,23,26,26 -mla 2 -fa -ctk q8_0 -amb 1024 -fmoe \
-ot "blk\.3\.ffn_up_exps=CUDA0, blk\.3\.ffn_gate_exps=CUDA0" \
-ot "blk\.4\.ffn_up_exps=CUDA1, blk\.4\.ffn_gate_exps=CUDA1" \
-ot "blk\.5\.ffn_up_exps=CUDA2, blk\.5\.ffn_gate_exps=CUDA2" \
-ot "blk\.6\.ffn_up_exps=CUDA3, blk\.6\.ffn_gate_exps=CUDA3" \
-ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" \
--threads 64
```

The all.txt file is a merge of these (I had to conver parquet to txt first):
https://huggingface.co/datasets/eaddario/imatrix-calibration/resolve/main/calibration_all_large.parquet
https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8/raw/2c64bb691316d32915b188e495754ef34931ae71/calibration_datav3.txt
https://gist.github.com/bartowski1182/f003237f2e8612278a6d01622af1cb6f/raw/6cf9d7538b3a234952d927459d0ce42cb3d3ea6e/qwen_calibration_with_chat.txt
(also, some personal data, but probably will have little compared to the three datasets above).

I probably could have just used calibration_datav3.txt and nothing else, but calibration_all_large contained many languages that are not well represented in calibration_datav3.txt or qwen_calibration_with_chat.txt, and I happen to need support for multiple languages since I often do translation work.

By the way, I remember a post where someone tested creating imatrix.dat file from BF16, Q8, Q6 and some lower quants, and then creating imatrix quant from BF16 with it, and the conclusion was the result was practically identical, especially if higher quants are used to create the imatrix. I did not save the link to it at the time (it was long before now), but I thought I mention it, to explain why I used Q6_K for this purpose.

Estimated time to generate imatrix.dat was 16 hours, and I am still waiting for it to finish. Once I complete generating the imatrix.dat, I plan to run this command to create a final quant:

```
~/pkgs/ik_llama.cpp/build/bin/llama-quantize \
--imatrix imatrix.dat \
/mnt/secondary/neuro/DeepSeek-R1T-Chimera-163840seq/DeepSeek-R1T-Chimera-256x21B-163840seq-BF16-00001-of-00030.gguf \
/mnt/secondary/neuro/DeepSeek-R1T-Chimera-163840seq/DeepSeek-R1T-Chimera-256x21B-IQ4_K_R4-163840seq.gguf \
IQ4_K_R4
```

I also plan to try other methods besides IQ4_K_R4, like IQ4_NL_R4 - to see if I will get better performance on my rig with CPU+GPU inference.

Due to my upload speed being around 1Mbps on average, I will not be able to share any of my quants, but I hope documenting the process will help others who may want to create their own quant. Even once this issue is closed, I still will be able to link here in case I want to share my steps elsewhere, since there was a lot of useful discussion and valuable information shared in this thread.

By the way, I also confirm that loading the existing quant from huggingface works now - so it seems the original issue that was reported is fixed. It is amazing that we can now use new MLA-enabled quants created by llama.cpp, but creating own quant may help to achieve better quality and performance, especially for models with very limited selection of quants like in this case. However, figuring out how to do it was really big challenge, and I wouldn't be able to do it without help. Big thanks to @saood06 and @ikawrakow!

---

👤 **saood06** commented the **2025-05-11** at **08:19:17**:<br>

>Due to my upload speed being around 1Mbps on average, I will not be able to share any of my quants, but I hope documenting the process will help others who may want to create their own quant.

It is understandable, I am in the same position as are many others. I would be very grateful if you could upload the imatrix file generated (it is only 1 GB).

>>I also plan to try other methods besides IQ4_K_R4, like IQ4_NL_R4 - to see if I will get better performance on my rig with CPU+GPU inference.

`_R4` tensors should only be on the CPU. You should use non` _R4` tensors for the GPU's. Also on Ampere or newer GPU's this is relevant : https://github.com/ikawrakow/ik_llama.cpp/pull/386.

---

👤 **ikawrakow** commented the **2025-05-12** at **05:41:16**:<br>

I think this is solved now.

---

👤 **Alexey-Akishin** commented the **2025-05-12** at **15:33:55**:<br>

I just tested and solved indeed, thank you so much!

I understand from the discussion that the pre-made quant from HG is not perfect, but it is all I got and I can't download the full model or even another quant this month due to bandwidth limits, so I am very grateful for being able to use the one I already have! Thanks again for fixing this.

---

👤 **Lissanro** commented the **2025-05-12** at **15:57:49**:<br>

@saood06 I have updated my previous comment based on your feedback: added imatrix link (it turned out to be 130MB) and also fixed commands to properly generate and repack quant using R4 only where needed as you have suggested (the repack pattern for CPU may need to be adjusted for a specific configuration, unless it happen to match with mine). Hope the experience I shared will be useful to those who decide to generate their own quant.

---

👤 **saood06** commented the **2025-05-13** at **00:31:52**:<br>

>added imatrix link (it turned out to be 130MB)

I'm not sure why it is smaller. All the imatrix files I've seen for that architecture are 987 MB. I have no idea why yours is smaller, but I really do appreciate you sharing it.

>Hope the experience I shared will be useful to those who decide to generate their own quant.

Thank you for documenting this to help others.

---

👤 **ubergarm** commented the **2025-05-13** at **20:37:25**:<br>

@Lissanro great job jumping through all the hoops and finding the breadcrumbs spread around github, reddit, etc!

> My imatrix: https://dragon.studio/2025/05/DeepSeek-R1T-Chimera-imatrix-8192seq.dat

Just curious, given the date on this is ~3 days ago, I'm guessing it wasn't created with this https://github.com/ikawrakow/ik_llama.cpp/pull/411 ? Not sure how much it will effect you if you're using mostly >~4bpw quants.

If you're looking for speed, a recent PR improved CUDA performance on `iq4_ks`. I'm toying with maybe making a new quant something like this, just playing around for now though given I don't have enough VRAM to really make use of https://github.com/ikawrakow/ik_llama.cpp/pull/374 with DeepSeek...

<details>

<summary>Possible quant recipe</summary>

```
#!/usr/bin/env bash

# Notes:
# https://github.com/ikawrakow/ik_llama.cpp/issues/296#issuecomment-2765210993
# https://github.com/ikawrakow/ik_llama.cpp/issues/296#issuecomment-2768567062
custom="
# Token embedding and output tensors (GPU)
# Remember only use _r4 for CPU *only* or offline repack later
# Remember all attention and shexp isn't so big so could go all q8_0 and still fit under 24GB VRAM w/ 32k MLA context
# note token_embd cannot be repacked quant type
token_embd\.weight=iq6_k
output\.weight=iq6_k
output_norm\.weight=iq6_k

# First 3 dense layers (0-3) (GPU)
blk\.[0-2]\.attn_k_b.*=q6_0
blk\.[0-2]\.attn_.*=iq6_k
blk\.[0-2]\..*=iq6_k

# All attention, norm weights, and bias tensors for MoE layers (3-60) (GPU)
# Except blk.*.attn_k_b.weight is not divisible by 256 and no iq6_k so go with q6_0
blk\.[3-9]\.attn_k_b.*=q6_0
blk\.[1-5][0-9]\.attn_k_b.*=q6_0
blk\.60\.attn_k_b.*=q6_0

blk\.[3-9]\.attn_.*=iq6_k
blk\.[1-5][0-9]\.attn_.*=iq6_k
blk\.60\.attn_.*=iq6_k

blk\.[3-9]\.ffn_norm\.weight=iq6_k
blk\.[1-5][0-9]\.ffn_norm\.weight=iq6_k
blk\.60\.ffn_norm\.weight=iq6_k

blk\.[3-9]\.exp_probs_b\.bias=iq6_k
blk\.[1-5][0-9]\.exp_probs_b\.bias=iq6_k
blk\.60\.exp_probs_b\.bias=iq6_k

# Shared Experts (3-60) (GPU)
blk\.[3-9]\.ffn_down_shexp\.weight=iq6_k
blk\.[1-5][0-9]\.ffn_down_shexp\.weight=iq6_k
blk\.60\.ffn_down_shexp\.weight=iq6_k

blk\.[3-9]\.ffn_(gate|up)_shexp\.weight=iq6_k
blk\.[1-5][0-9]\.ffn_(gate|up)_shexp\.weight=iq6_k
blk\.60\.ffn_(gate|up)_shexp\.weight=iq6_k

# Most of the model size is below
# Routed Experts (3-60) (CPU)
# usually ffn_down is made a bit bigger than ffn_(gate|up) but you do you
blk\.[3-9]\.ffn_down_exps\.weight=iq4_ks
blk\.[1-5][0-9]\.ffn_down_exps\.weight=iq4_ks
blk\.60\.ffn_down_exps\.weight=iq4_ks

blk\.[3-9]\.ffn_(gate|up)_exps\.weight=iq4_ks
blk\.[1-5][0-9]\.ffn_(gate|up)_exps\.weight=iq4_ks
blk\.60\.ffn_(gate|up)_exps\.weight=iq4_ks
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

./build/bin/llama-quantize \
    --imatrix /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324.imatrix \
    --custom-q "$custom" \
    /mnt/raid/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/DeepSeek-256x21B-V3-0324-BF16-00001-of-00030.gguf \
    /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_KS.gguf \
    IQ4_KS \
    24
```

</details>

---

👤 **ubergarm** commented the **2025-05-13** at **20:37:25**:<br>

@Lissanro great job jumping through all the hoops and finding the breadcrumbs spread around github, reddit, etc!

> My imatrix: https://dragon.studio/2025/05/DeepSeek-R1T-Chimera-imatrix-8192seq.dat

Just curious, given the date on this is ~3 days ago, I'm guessing it wasn't created with this https://github.com/ikawrakow/ik_llama.cpp/pull/411 ? Not sure how much it will effect you if you're using mostly >~4bpw quants.

If you're looking for speed, a recent PR improved CUDA performance on `iq4_ks`. I'm toying with maybe making a new quant something like this, just playing around for now though.

<details>

<summary>Possible quant recipe</summary>

```
#!/usr/bin/env bash

# Notes:
# https://github.com/ikawrakow/ik_llama.cpp/issues/296#issuecomment-2765210993
# https://github.com/ikawrakow/ik_llama.cpp/issues/296#issuecomment-2768567062
custom="
# Token embedding and output tensors (GPU)
# Remember only use _r4 for CPU *only* or offline repack later
# Remember all attention and shexp isn't so big so could go all q8_0 and still fit under 24GB VRAM w/ 32k MLA context
# note token_embd cannot be repacked quant type
token_embd\.weight=iq6_k
output\.weight=iq6_k
output_norm\.weight=iq6_k

# First 3 dense layers (0-3) (GPU)
blk\.[0-2]\.attn_k_b.*=q6_0
blk\.[0-2]\.attn_.*=iq6_k
blk\.[0-2]\..*=iq6_k

# All attention, norm weights, and bias tensors for MoE layers (3-60) (GPU)
# Except blk.*.attn_k_b.weight is not divisible by 256 and no iq6_k so go with q6_0
blk\.[3-9]\.attn_k_b.*=q6_0
blk\.[1-5][0-9]\.attn_k_b.*=q6_0
blk\.60\.attn_k_b.*=q6_0

blk\.[3-9]\.attn_.*=iq6_k
blk\.[1-5][0-9]\.attn_.*=iq6_k
blk\.60\.attn_.*=iq6_k

blk\.[3-9]\.ffn_norm\.weight=iq6_k
blk\.[1-5][0-9]\.ffn_norm\.weight=iq6_k
blk\.60\.ffn_norm\.weight=iq6_k

blk\.[3-9]\.exp_probs_b\.bias=iq6_k
blk\.[1-5][0-9]\.exp_probs_b\.bias=iq6_k
blk\.60\.exp_probs_b\.bias=iq6_k

# Shared Experts (3-60) (GPU)
blk\.[3-9]\.ffn_down_shexp\.weight=iq6_k
blk\.[1-5][0-9]\.ffn_down_shexp\.weight=iq6_k
blk\.60\.ffn_down_shexp\.weight=iq6_k

blk\.[3-9]\.ffn_(gate|up)_shexp\.weight=iq6_k
blk\.[1-5][0-9]\.ffn_(gate|up)_shexp\.weight=iq6_k
blk\.60\.ffn_(gate|up)_shexp\.weight=iq6_k

# Most of the model size is below
# Routed Experts (3-60) (CPU)
# usually ffn_down is made a bit bigger than ffn_(gate|up) but you do you
blk\.[3-9]\.ffn_down_exps\.weight=iq4_ks
blk\.[1-5][0-9]\.ffn_down_exps\.weight=iq4_ks
blk\.60\.ffn_down_exps\.weight=iq4_ks

blk\.[3-9]\.ffn_(gate|up)_exps\.weight=iq4_ks
blk\.[1-5][0-9]\.ffn_(gate|up)_exps\.weight=iq4_ks
blk\.60\.ffn_(gate|up)_exps\.weight=iq4_ks
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

./build/bin/llama-quantize \
    --imatrix /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324.imatrix \
    --custom-q "$custom" \
    /mnt/raid/models/deepseek-ai/DeepSeek-V3-0324-bf16-GGUF/DeepSeek-256x21B-V3-0324-BF16-00001-of-00030.gguf \
    /mnt/raid/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ4_KS.gguf \
    IQ4_KS \
    24
```

</details>

---

👤 **Lissanro** commented the **2025-05-13** at **22:16:19**:<br>

@ubergarm 
Thank you for sharing the recipe, I will give it a try, every bit of speed up will make a difference for me. I may have to wait until I get new 8TB SSD, I should get it within 1-2 days (since I ran out of space on my SSDs, and trying to load models from 16TB HDD takes hours instead of minutes like on SSD, making hard to experiment).

As of #411, it says "This PR fixes imatrix calculation for llama.cpp-style MLA GGUFs", but I generated my imatrix from a normal GGUF derived from BF16 (using ik_llama.cpp's tools), which in turn was derived from the original fp8 model. So most likely it will not have effect on my imatrix, but please correct me if I am wrong and if it worth regenarating.

@saood06 Not sure then why my imatrix is smaller, but I created it using ik_llama.cpp's llama-imatrix, maybe the larger versions were created by some other tool, or used some special settings?

I tried creating another imatrix with default 512 context length, and then compare perplexity of quants generated from it, and this is the result (in R4 quants, only tensors that I run on CPU were repacked as R4):

```
IQ4_K_R4 from imatrix generated using n_ctx=512:
Final estimate: PPL = 3.2911 +/- 0.01817 (perplexity tested with n_ctx=512)
Final estimate: PPL = 3.0219 +/- 0.01568 (perplexity tested with n_ctx=8192)
```

```
IQ4_K_R4 from imatrix generated using n_ctx=8192
Final estimate: PPL = 3.2911 +/- 0.01816 (perplexity tested with n_ctx=512)
Final estimate: PPL = 3.0230 +/- 0.01569 (perplexity tested with n_ctx=8192)
```

```
Q6_K reference quant:
Final estimate: PPL = 3.2611 +/- 0.01791 (perplexity tested with n_ctx=512)
Final estimate: PPL = 3.0039 +/- 0.01554 (perplexity tested with n_ctx=8192)
```

The conclusion it seems that generating imatrix with longer context either does not make a difference or makes quality very slightly worse (but within margin of error, so hard to tell). So generating imatrix with the default n_ctx=512 should be sufficient (it was suggested by someone in the discussions I linked in my earlier post that Unsloth may have been using context length within 6144 - 12288 range to generate imatrix, so I wanted to see if it actually makes a difference, but apparently not).

For reference, this is the command I used to test perplexity:

```
numactl --cpunodebind=0 --interleave=all ~/pkgs/ik_llama.cpp/build/bin/llama-perplexity \
--model /path/to/model.gguf --n-gpu-layers 62 --tensor-split 25,23,26,26 \
-mla 3 -fa -ctk q8_0 -amb 1024 -fmoe \
-ot "blk\.3\.ffn_up_exps=CUDA0, blk\.3\.ffn_gate_exps=CUDA0" \
-ot "blk\.4\.ffn_up_exps=CUDA1, blk\.4\.ffn_gate_exps=CUDA1" \
-ot "blk\.5\.ffn_up_exps=CUDA2, blk\.5\.ffn_gate_exps=CUDA2" \
-ot "blk\.6\.ffn_up_exps=CUDA3, blk\.6\.ffn_gate_exps=CUDA3" \
-ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" \
--threads 64 -f /home/lissanro/pkgs/ik_llama.cpp/wikitext-2-raw/wiki.test.ra \
--ctx-size 512
```

In case someone else decides to test their quants, the command needs to be adjusted for a specific configuration, for non-repacked quants -rtr option may be needed, and ctx-size is 512 by default but can be changed if needed. And to get wiki.test.ra, I had to run the following command:

`~/pkgs/ik_llama.cpp/scripts/get-wikitext-2.sh`

---

👤 **Lissanro** commented the **2025-05-13** at **22:16:19**:<br>

@ubergarm 
Thank you for sharing the recipe, I will give it a try, every bit of speed up will make a difference for me. I may have to wait until I get new 8TB SSD, I should get it within 1-2 days (since I ran out of space on my SSDs, and trying to load models from 16TB HDD takes hours instead of minutes like on SSD, making hard to experiment).

As of #411, it says "This PR fixes imatrix calculation for llama.cpp-style MLA GGUFs", but I generated my imatrix from a normal GGUF derived from BF16 (using ik_llama.cpp's tools), which in turn was derived from the original fp8 model. So most likely it will not have effect on my imatrix, but please correct me if I am wrong and if it worth regenarating.

@saood06 Not sure then why my imatrix is smaller, but I created it using ik_llama.cpp's llama-imatrix, maybe the larger versions were created by some other tool, or used some special settings?

I tried creating another imatrix with default 512 context length, and then compare perplexity of quants generated from it, and this is the result (in R4 quants, only tensors that I run on CPU were repacked as R4):

```
IQ4_K_R4 from imatrix generated using n_ctx=512:
Final estimate: PPL = 3.2911 +/- 0.01817 (perplexity tested with n_ctx=512)
Final estimate: PPL = 3.0219 +/- 0.01568 (perplexity tested with n_ctx=8192)
```

```
IQ4_K_R4 from imatrix generated using n_ctx=512
Final estimate: PPL = 3.2911 +/- 0.01816 (perplexity tested with n_ctx=512)
Final estimate: PPL = 3.0230 +/- 0.01569 (perplexity tested with n_ctx=8192)
```

```
Q6_K reference quant:
Final estimate: PPL = 3.2611 +/- 0.01791 (perplexity tested with n_ctx=512)
Final estimate: PPL = 3.0039 +/- 0.01554 (perplexity tested with n_ctx=8192)
```

The conclusion it seems that generating imatrix with longer context either does not make a difference or makes quality very slightly worse (but within margin of error, so hard to tell). So generating imatrix with the default n_ctx=512 should be sufficient (it was suggested by someone in the discussions I linked in my earlier post that Unsloth may have been using context length within 6144 - 12288 range to generate imatrix, so I wanted to see if it actually makes a difference, but apparently not).

For reference, this is the command I used to test perplexity:

```
numactl --cpunodebind=0 --interleave=all ~/pkgs/ik_llama.cpp/build/bin/llama-perplexity \
--model /path/to/model.gguf --n-gpu-layers 62 --tensor-split 25,23,26,26 \
-mla 3 -fa -ctk q8_0 -amb 1024 -fmoe \
-ot "blk\.3\.ffn_up_exps=CUDA0, blk\.3\.ffn_gate_exps=CUDA0" \
-ot "blk\.4\.ffn_up_exps=CUDA1, blk\.4\.ffn_gate_exps=CUDA1" \
-ot "blk\.5\.ffn_up_exps=CUDA2, blk\.5\.ffn_gate_exps=CUDA2" \
-ot "blk\.6\.ffn_up_exps=CUDA3, blk\.6\.ffn_gate_exps=CUDA3" \
-ot "ffn_down_exps=CPU, ffn_up_exps=CPU, gate_exps=CPU" \
--threads 64 -f /home/lissanro/pkgs/ik_llama.cpp/wikitext-2-raw/wiki.test.ra \
--ctx-size 512
```

In case someone else decides to test their quants, the command needs to be adjusted for a specific configuration, for non-repacked quants -rtr option may be needed, and ctx-size is 512 by default but can be changed if needed. And to get wiki.test.ra, I had to run the following command:

`~/pkgs/ik_llama.cpp/scripts/get-wikitext-2.sh`

---

👤 **saood06** commented the **2025-05-13** at **22:29:01**:<br>

> trying to load models from 16TB HDD takes hours instead of minutes like on SSD, making hard to experiment).

I've only ever used HDDs for these quants, and yes it is quite the pain.
 
>but please correct me if I am wrong and if it worth regenarating.

I think your understanding is correct. It may be worth regenerating if there turns out to be an issue leading to your smaller than expected `imatrix.dat` size but that would be a separate issue.

>Not sure then why my imatrix is smaller

I'm sorry, I'm not sure either.

> The conclusion it seems that generating imatrix with longer context either does not make a difference or makes quality very slightly worse (but within margin of error, so hard to tell). So generating imatrix with the default n_ctx=512 should be sufficient (it was suggested by someone in the discussions I linked in my earlier post that Unsloth may have been using context length within 6144 - 12288 range to generate imatrix, so I wanted to see if it actually makes a difference, but apparently not).

Thank you for your testing and sharing of the results.

---

👤 **ubergarm** commented the **2025-05-14** at **01:37:52**:<br>

@Lissanro 

> if it's worth regenerating.

tbh I'm not sure myself. if you're using all > ~4bpw quants it might not make a huge deal. 

> Not sure then why my imatrix is smaller

I just converted [tngtech/DeepSeek-R1T-Chimera](https://huggingface.co/tngtech/DeepSeek-R1T-Chimera) fp8 to bf16 GGUF with evshiron's llama.cpp fork and triton-cpu. I can't run the full bf16 easily with enough RAM in a single NUMA node so just made a full q8_0 version without imatrix first. Then using the q8_0 as my baseline I kept it simple and old school with

```bash
numactl -N 0 -m 0 \
./build/bin/llama-imatrix \
    --verbosity 1 \
    -m /media/b/data2/models/ubergarm/DeepSeek-R1T-Chimera-GGUF/DeepSeek-R1T-Chimera-Q8_0.gguf \
    -f calibration_data_v5_rc.txt \
    -o DeepSeek-R1T-Chimera.imatrix \
    --ctx-size 512 \
    --numa numactl \
    --threads 40
```
Resulting imatrix size is 942MiB and when using it to quantize it prints out: `720 importance matrix entries ... on 213 chunks`.

Also here is a snippet of all of of `blk.18.*` logs showing the various tensor names in this one:

<details>

<summary>👈 Snippet of ik_llama.cpp llama-quantize showing tensors</summary>

```
[ 329/1147]              blk.18.exp_probs_b.bias - [  256,     1,     1,     1], type =    f32, size =    0.001 MB
[ 330/1147]           blk.18.ffn_gate_inp.weight - [ 7168,   256,     1,     1], type =    f32, size =    7.000 MB
[ 331/1147]         blk.18.ffn_down_shexp.weight - [ 2048,  7168,     1,     1], type =   bf16, Using custom type iq6_k for tensor blk.18.ffn_down_shexp.weight
converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 332/1147]         blk.18.ffn_gate_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, Using custom type iq6_k for tensor blk.18.ffn_gate_shexp.weight
converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 333/1147]           blk.18.ffn_up_shexp.weight - [ 7168,  2048,     1,     1], type =   bf16, Using custom type iq6_k for tensor blk.18.ffn_up_shexp.weight
converting to iq6_k .. size =    28.00 MiB ->    11.59 MiB
[ 334/1147]         blk.18.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
[ 335/1147]          blk.18.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =   bf16, Using custom type iq6_k for tensor blk.18.attn_kv_a_mqa.weight
converting to iq6_k .. size =     7.88 MiB ->     3.26 MiB
[ 336/1147]              blk.18.attn_kv_b.weight - [  512, 32768,     1,     1], type =   bf16, Using custom type iq6_k for tensor blk.18.attn_kv_b.weight
converting to iq6_k .. size =    32.00 MiB ->    13.25 MiB
[ 337/1147]               blk.18.attn_k_b.weight - [  128, 65536,     1,     1], type =   bf16, Using custom type q6_0 for tensor blk.18.attn_k_b.weight
====== llama_model_quantize_internal: did not find weights for blk.18.attn_k_b.weight
converting to q6_0 .. size =    16.00 MiB ->     6.50 MiB
[ 338/1147]               blk.18.attn_v_b.weight - [  512, 16384,     1,     1], type =   bf16, Using custom type iq6_k for tensor blk.18.attn_v_b.weight
converting to iq6_k .. size =    16.00 MiB ->     6.62 MiB
[ 339/1147]            blk.18.attn_output.weight - [16384,  7168,     1,     1], type =   bf16, Using custom type iq6_k for tensor blk.18.attn_output.weight
converting to iq6_k .. size =   224.00 MiB ->    92.75 MiB
[ 340/1147]          blk.18.attn_q_a_norm.weight - [ 1536,     1,     1,     1], type =    f32, size =    0.006 MB
[ 341/1147]               blk.18.attn_q_a.weight - [ 7168,  1536,     1,     1], type =   bf16, Using custom type iq6_k for tensor blk.18.attn_q_a.weight
converting to iq6_k .. size =    21.00 MiB ->     8.70 MiB
[ 342/1147]               blk.18.attn_q_b.weight - [ 1536, 24576,     1,     1], type =   bf16, Using custom type iq6_k for tensor blk.18.attn_q_b.weight
converting to iq6_k .. size =    72.00 MiB ->    29.81 MiB
[ 343/1147]              blk.18.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
[ 344/1147]          blk.18.ffn_down_exps.weight - [ 2048,  7168,   256,     1], type =   bf16, Using custom type iq4_ks for tensor blk.18.ffn_down_exps.weight
converting to iq4_ks .. size =  7168.00 MiB ->  1911.00 MiB
[ 345/1147]          blk.18.ffn_gate_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type iq4_ks for tensor blk.18.ffn_gate_exps.weight
converting to iq4_ks .. size =  7168.00 MiB ->  1906.00 MiB
[ 346/1147]            blk.18.ffn_up_exps.weight - [ 7168,  2048,   256,     1], type =   bf16, Using custom type iq4_ks for tensor blk.18.ffn_up_exps.weight
converting to iq4_ks .. size =  7168.00 MiB ->  1906.00 MiB
[ 347/1147]               blk.18.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
```

</details>

> The conclusion it seems that generating imatrix with longer context either does not make a difference or makes quality very slightly worse (but within margin of error, so hard to tell). So generating imatrix with the default n_ctx=512 should be sufficient

Hey appreciate the additional data points with your practical empirical approach. If you follow along there is already [much interesting old discussions still available](https://github.com/ggml-org/llama.cpp/discussions/5263) which suggests the same.  Apparently [unsloth is using longer context length at least for some GGUF imatrix files now](https://huggingface.co/unsloth/Qwen3-30B-A3B-128K-GGUF/discussions/8#6821262ba2ff408c1deccba6) but to be honest I don't follow their logic nor yet see any clear evidence. (I'm not saying its wrong, it might be so, but I don't know.)

With luck I'll have some updated perplexity values using the latest method for generating imatrix and update you. Thanks for sharing your research!

---

👤 **ubergarm** commented the **2025-05-14** at **01:37:52**:<br>

> if it's worth regenerating.

tbh I'm not sure myself. if you're using all > ~4bpw quants it might not make a huge deal. 

> Not sure then why my imatrix is smaller

I just converted [tngtech/DeepSeek-R1T-Chimera](https://huggingface.co/tngtech/DeepSeek-R1T-Chimera) fp8 to bf16 GGUF with evshiron's llama.cpp fork and triton-cpu. I can't run the full bf16 easily with enough RAM in a single NUMA node so just made a full q8_0 version without imatrix first. Then using the q8_0 as my baseline I kept it simple and old school with

```bash
numactl -N 0 -m 0 \
./build/bin/llama-imatrix \
    --verbosity 1 \
    -m /media/b/data2/models/ubergarm/DeepSeek-R1T-Chimera-GGUF/DeepSeek-R1T-Chimera-Q8_0.gguf \
    -f calibration_data_v5_rc.txt \
    -o DeepSeek-R1T-Chimera.imatrix \
    --ctx-size 512 \
    --numa numactl \
    --threads 40
```
Resulting imatrix size is 942MiB and when using it to quantize it prints out: `720 importance matrix entries ... on 213 chunks`.

> The conclusion it seems that generating imatrix with longer context either does not make a difference or makes quality very slightly worse (but within margin of error, so hard to tell). So generating imatrix with the default n_ctx=512 should be sufficient

Hey appreciate the additional data points with your practical empirical approach. If you follow along there is already [much interesting old discussions still available](https://github.com/ggml-org/llama.cpp/discussions/5263) which suggests the same.  Apparently [unsloth is using longer context length at least for some GGUF imatrix files now](https://huggingface.co/unsloth/Qwen3-30B-A3B-128K-GGUF/discussions/8#6821262ba2ff408c1deccba6) but to be honest I don't follow their logic nor yet see any clear evidence. (I'm not saying its wrong, it might be so, but I don't know.)

With luck I'll have some updated perplexity values using the latest method for generating imatrix and update you. Thanks for sharing your research!

---

👤 **Lissanro** commented the **2025-05-15** at **05:49:40**:<br>

> Resulting imatrix size is 942MiB and when using it to quantize it prints out: 720 importance matrix entries ... on 213 chunks.

For me it shows "load_imatrix: loaded 543 importance matrix entries from DeepSeek-R1T-Chimera-imatrix.dat computed on 3660 chunks" (probably because I am using large input file) and resulting size is 130 MB. I wonder what makes mine smaller, maybe because I am creating it from Q6_K instead of Q8_0? However, my imatrix file seems to work as expected as far as I can tell.

> Possible quant recipe

I have tested the recipe for the IQ4_KS quant and based on perplexity it seems to be quite good, the size is slightly smaller, perplexity remained almost exactly the same as for IQ4_K and performance remained similar (slightly more than 8 tokens/s for both IQ4_K and IQ4_KS quants, with only necessary for CPU tensors converted to R4, on EPYC 7763 + 1 TB 3200MHz RAM + 4x3090 GPUs):

```
IQ4_KS_R4 (339G)
Final estimate: PPL = 3.2876 +/- 0.01807
Final estimate: PPL = 3.0262 +/- 0.01568
```

```
IQ4_K_R4 (356G):
Final estimate: PPL = 3.2911 +/- 0.01817 (perplexity tested with n_ctx=512)
Final estimate: PPL = 3.0219 +/- 0.01568 (perplexity tested with n_ctx=8192)
```

```
Q6_K reference quant (515G):
Final estimate: PPL = 3.2611 +/- 0.01791 (perplexity tested with n_ctx=512)
Final estimate: PPL = 3.0039 +/- 0.01554 (perplexity tested with n_ctx=8192)
```

UPDATE: Further testing revealed Q4_KS quant quality dropped significantly in reasoning tasks, most noticeable in the [maze test](https://www.reddit.com/r/LocalLLaMA/comments/1j4lqe6/test_if_your_api_provider_is_quantizing_your/):

IQ4_K_R4: 10 / 10 (100% success rate)
IQ4_KS_R4: 1 / 10 (10% success rate)

Since performance and size are similar, normal imatrix IQ4_K_R4 quant seem to be the best option.

---

👤 **Lissanro** commented the **2025-05-15** at **05:49:40**:<br>

> Resulting imatrix size is 942MiB and when using it to quantize it prints out: 720 importance matrix entries ... on 213 chunks.

For me it shows "load_imatrix: loaded 543 importance matrix entries from DeepSeek-R1T-Chimera-imatrix.dat computed on 3660 chunks" (probably because I am using large input file) and resulting size is 130 MB. I wonder what makes mine smaller, maybe because I am creating it from Q6_K instead of Q8_0? However, my imatrix file seems to work as expected as far as I can tell.

> Possible quant recipe

I have tested the recipe for the IQ4_KS quant and based on perplexity it seems to be quite good, the size is slightly smaller, perplexity remained almost exactly the same as for IQ4_K and performance remained similar (slightly more than 8 tokens/s for both IQ4_K and IQ4_KS quants, with only necessary for CPU tensors converted to R4, on EPYC 7763 + 1 TB 3200MHz RAM + 4x3090 GPUs):

```
IQ4_KS_R (339G)
Final estimate: PPL = 3.2876 +/- 0.01807
Final estimate: PPL = 3.0262 +/- 0.01568
```

```
IQ4_K_R4 (356G):
Final estimate: PPL = 3.2911 +/- 0.01817 (perplexity tested with n_ctx=512)
Final estimate: PPL = 3.0219 +/- 0.01568 (perplexity tested with n_ctx=8192)
```

```
Q6_K reference quant (515G):
Final estimate: PPL = 3.2611 +/- 0.01791 (perplexity tested with n_ctx=512)
Final estimate: PPL = 3.0039 +/- 0.01554 (perplexity tested with n_ctx=8192)
```

---

👤 **ubergarm** commented the **2025-05-15** at **14:49:15**:<br>

@Lissanro 

> However, my imatrix file seems to work as expected as far as I can tell.

Yeah it seems like just having almost any imatrix is generally better than not.

I just got my first numbers on this [DeepSeek-R1T-Chimera-IQ4_KS](https://huggingface.co/ubergarm/DeepSeek-R1T-Chimera-GGUF#deepseek-r1t-chimera-iq4_ks)*:

```
IQ4_KS - 338.456 GiB - 4.326 BPW
Final estimate: PPL = 3.4082 +/- 0.01892
```

*EDIT*: the q8_0 came back with `Final estimate: PPL = 3.3793 +/- 0.01873` so this KS seems really good from a PPL only perspective, want to try that maze test too though if it ever finishes upload!

*it is super slow to upload, not sure it will ever finish lol... The new imatrix is at least there computed with the latest fixes from PR411

My PPL is higher than yours, could be using iq6_k for all attention, but you have the longer imatrix corpus as well. Too many variables to know for sure but at least another data point.

<details>

<summary>perplexity command</summary>

```
# running on single 4090 GPU with plenty of RAM
$ wget https://github.com/user-attachments/files/19090237/wiki.test.raw.gz
$ gunzip wiki.test.raw.gz
$ numactl -N 0 -m 0 \
./build/bin/llama-perplexity \
    -m /models/ubergarm/DeepSeek-R1T-Chimera-GGUF/DeepSeek-R1T-Chimera-IQ4_KS.gguf \
    -f wiki.test.raw \
    --ctx-size 512 \
    --ubatch-size 512 \
    --seed 1337 \
    -ctk f16 \
    -fa \
    -mla 3 \
    -amb 512 \
    -fmoe \
    -ngl 99 \
    --override-tensor exps=CPU \
    -rtr \
    --numa numactl \
    --threads 40
```

</details>

> Further testing revealed Q4_KS quant quality dropped significantly in reasoning tasks, most noticeable in the [maze test](https://www.reddit.com/r/LocalLLaMA/comments/1j4lqe6/test_if_your_api_provider_is_quantizing_your/):

Huh fascinating, I wonder what is going on there. Both the K and KS have similar perplexities. I haven't looked into the maze test but will maybe try it out on some smaller models locally soon just to see. Is it failing in terms of getting the output directions correct with you as a human looking at the result? Or is it some syntactical errors with it messing up the `<|up|>` formatting resulting in a "failed" run as computed by some python script? I assume sampling may effect the output somewhat. But if it works reliably it could be a useful test, thanks for sharing!

*EDIT2*:

I went back and looked at that maze test and was wondering why the formatting looked like token strings e.g. `<|0-0|><|up_down_left_wall|><|blank|>` type stuff, and looking at the [alphamaze paper referenced](https://arxiv.org/html/2502.14669v3) they are training their model to use specific token representations of the maze.

Given most models are not trained on those specific tokens I have some questions like:
1. Wouldn't one just use some other maze representation rather than these "token like" strings?
2. Is there a better more generalized representation that would improve a model not trained specifically on alphamaze tokens' performance?
3. Is there really a sudden "break point" in quantization quality where there is a repeatable meaningful difference e.g. `iq4_k` can solve the maze with 95% chance but slightly smaller `iq4_ks` can only solve it say 15% chance, and if so, does this generalize to indicate similar sudden gap in performance in other tasks or not?

I guess I'm not sure it applies to use this tokenized style maze test on models not trained to recognize those tokens? This is from the paper:

> (SFT) approach on the DeepSeek-R1-Distill-Qwen-1.5B architecture. This model was trained to directly predict the complete sequence of movement tokens representing the solution path through a given maze

This specific alpha maze test seems to be used on SFTd models to compare the underlying model architectures ability to solve spatial tasks, not to compare quantizations of a model not SFTd with these tokens.

But I dunno, maybe it is useful?

---

👤 **ubergarm** commented the **2025-05-15** at **14:49:15**:<br>

@Lissanro 

> However, my imatrix file seems to work as expected as far as I can tell.

Yeah it seems like just having almost any imatrix is generally better than not.

I just got my first numbers on this [DeepSeek-R1T-Chimera-IQ4_KS](https://huggingface.co/ubergarm/DeepSeek-R1T-Chimera-GGUF#deepseek-r1t-chimera-iq4_ks)*:

```
IQ4_KS - 338.456 GiB - 4.326 BPW
Final estimate: PPL = 3.4082 +/- 0.01892
```

*it is super slow to upload, not sure it will ever finish lol... The new imatrix is at least there computed with the latest fixes from PR411

Need to run one on the Q8_0 for comparison but its kinda slow as I haven't optimized the command on this remote rig.

My PPL is higher than yours, could be using iq6_k for all attention, but you have the longer imatrix corpus as well. Too many variables to know for sure but at least another data point.

<details>

<summary>perplexity command</summary>

```
# running on single 4090 GPU with plenty of RAM
$ wget https://github.com/user-attachments/files/19090237/wiki.test.raw.gz
$ gunzip wiki.test.raw.gz
$ numactl -N 0 -m 0 \
./build/bin/llama-perplexity \
    -m /models/ubergarm/DeepSeek-R1T-Chimera-GGUF/DeepSeek-R1T-Chimera-IQ4_KS.gguf \
    -f wiki.test.raw \
    --ctx-size 512 \
    --ubatch-size 512 \
    --seed 1337 \
    -ctk f16 \
    -fa \
    -mla 3 \
    -amb 512 \
    -fmoe \
    -ngl 99 \
    --override-tensor exps=CPU \
    -rtr \
    --numa numactl \
    --threads 40
```

</details>

> Further testing revealed Q4_KS quant quality dropped significantly in reasoning tasks, most noticeable in the [maze test](https://www.reddit.com/r/LocalLLaMA/comments/1j4lqe6/test_if_your_api_provider_is_quantizing_your/):

Huh fascinating, I wonder what is going on there. Both the K and KS have similar perplexities. I haven't looked into the maze test but will maybe try it out on some smaller models locally soon just to see. Is it failing in terms of getting the output directions correct with you as a human looking at the result? Or is it some syntactical errors with it messing up the `<|up|>` formatting resulting in a "failed" run as computed by some python script? I assume sampling may effect the output somewhat. But if it works reliably it could be a useful test, thanks for sharing!