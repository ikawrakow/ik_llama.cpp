### [Discussion #242](https://github.com/ikawrakow/ik_llama.cpp/discussions/242) - Switching from llama.cpp/ktransformers, seeking advice/guidance

| **Author** | `ThomasBaruzier` |
| :--- | :--- |
| **Created** | 2025-03-05 |
| **Updated** | 2025-03-15 |

---

#### Description

Hello,

I discovered this repo today, and I'm very excited to try all the new features and optimizations made here.

I am currently downloading R1 BF16 (can't convert using 3090, lack of fp8 support), and in the meantime, I am trying to learn as much as possible.

The goal is to run R1 with a reasonable PPL using 72GB VRAM and 128 GB RAM. Looking at the PRs and comments, the new IQ1_S_R4 (https://github.com/ikawrakow/ik_llama.cpp/pull/185) and IQ1_M_R4 (https://github.com/ikawrakow/ik_llama.cpp/pull/187) quants look really promising, as well as all the fancy stuff related to MLA and context cache (https://github.com/ikawrakow/ik_llama.cpp/pull/208, https://github.com/ikawrakow/ik_llama.cpp/pull/240, https://github.com/ikawrakow/ik_llama.cpp/pull/241, ...), but it's a bit overwhelming at first glance.

I guess that the best option right now is to run one of these R4 quants, writing rules that are equivalent to a Ktransformers config for partial offload of critical sections of the model (https://github.com/ikawrakow/ik_llama.cpp/pull/232), and try poking around with `--mla` values. For cache, I guess I can play with the new Q8_KV if applicable. Regarding CUDA, MLA and/or FA, I am sure what is compatible for CPU / GPU / multi GPU, what combinations of parameters could work.

Do you have any advice regarding this type of setup? Is there a way to use more VRAM by selectively offloading individual experts/layers? If I read it right, R4 quants do not support offloading yet. Are there other tweaks or resources I can learn from to try and use your work as efficiently as possible?

I'd be happy to share my benchmarks and params when I am done quanting the model.

Thank you very much

---

#### 🗣️ Discussion

👤 **ikawrakow** commented on **2025-03-06** at **06:01:05**

Is the 72 GB VRAM from 3 x 24 GB GPUs?

You setup is somewhat unusual as you "only" have 128 GB of RAM. If you want to use a ready model your only option would be the `IQ1_S` or `IQ1_M` models from Unsloth. The next step up is already too big for the 200 GB you have available.

If you are willing to do your custom quantization, it will require a manual setup as there isn't an out-of-the-box mix to best take advantage of your amount of RAM+VRAM. I guess, I should add a similar functionality as the tensor overrides from #232 also to `llama-quantize` so people don't need to go and change the code to get the quantization mix they want.

Once you have a model that you want to use, I think the best way to distribute the model weights between CPU RAM and GPU VRAM will be to use several `-ot` command line arguments. But to determine the regular expressions required one needs to know the quantization types (and hence sizes) of all tensors.

What is the CPU in this system?

---

👤 **ikawrakow** commented on **2025-03-07** at **12:00:58**

PR #244 has been merged, so hopefully this will help you with making your custom DeepSeekR1 quantization.

The `-b 31 -ub 31` option is a clever hack, but I expect prompt processing performance to be unacceptably low. So will be TG with any significant context (more than a few hundred tokens). Or not?

---

👤 **ikawrakow** commented on **2025-03-07** at **15:16:11**

Could the following work in your 3x24 GiB VRAM + 128 GiB RAM:

* The first 3 dense layers + `output.weight` + all attention tensors + all shared experts on GPU0. If you quantize of of these with `Q6_K` or `Q5_K`, this will use 12.2 GiB or 10.2 GiB of VRAM. This will allow you to use longer contexts. If you don't need the longer context, you can add 2-3 MoE experts layers to GPU0.
* Let's assume you decide to put 2 extra layers on GPU0. The first MoE layers are very important, so I would use `IQ4_XS` for `ffn_down_exps` and `IQ2_XXS` for `ffn_up/gate_exps`. This uses 3.664 GiB per layer, so with the 10.24 GiB from above using `Q5_K` you have used up 17.57 GiB on GPU0. 6.5 remaining GiB is still plenty for KV cache and compute buffer if you use `mla = 2` for attention. 
* 7 MoE layers (layers 5-11) on GPU1 where `ffn_down_exps` is quantized with `IQ3_XXS`, and `ffn_gate_exps` and `ffn_up_exps` with `IQ2_XXS`.  This uses 22.3 GiB of VRAM, so ~1.5 GiB are left for compute buffers so you don't need `-b 31 -ub 31`
* Another 7 MoE layers (layers 12-18) done the same way on GPU2 (not 100% sure about that, it might be that it is better to put the last 7 layers on GPU2. From past experience using more bits on the last few layers improved some models).
* You are now left with 42 layers for the 128 GiB of RAM to be processed by the CPU. If you use `IQ2_K` for `ffn_down_exps` and `IQ2_XXS` for `ffn_up/gate_exps`, this is 2.844 GiB per layer, so 119.44 GiB in total. 

Oh, forgot. The tensors that go on the CPU should be quantized to the corresponding `_R4`  variant. You can decide to not quantize to `*_R4` and then use run time repacking (`-rtr`) to repack to `_R4`, but this adds quite a bit of extra loading time (2-3 minutes on a 32-core EPYC).

---

👤 **ikawrakow** commented on **2025-03-07** at **17:57:23**

The NaNs are concerning. If we got NaN probabilities (logits) out of the forward pass, the imatrix will be useless (will likely have NaNs). Another way to get a NaN in the perplexity is if the predicted probability for the observed token is zero. You maybe better of getting an imatrix from somewhere else. Have you tried running the same calculation with mainline `llama.cpp`? Btw, if you want to create imatrix data yourself and have enough disk space, you can quantize to `Q8_0` (no imatrix required for that), and then use the quantized model for the imatrix calculation. You will fit 2X more layers on the GPUs, so it may be somewhat faster. 

The messages about partial data are to be expected. Only 8 out of 256 experts get activated per token, so if the batch was short, it is likely to have some experts that never were activated, so the imatrix for those contains just zeros. If one tries to use such an imatrix to quantize a model, this can lead to bad results (including NaNs in the model). That's why in mainline `llama.cpp` they wouldn't let you save the data for **the entire experts tensor**, even if just one expert is missing data. I have changed that to allow the imatrix to be saved (and fill the missing experts with 1s to avoid issues during quantization), but only if the number of missing experts is greater than some fraction of the total experts in the tensor. That's why initially you see for some tensors "storing but be aware", and for others you see "skipping". As you collect more data eventually all experts have seen at least one token, so the messages go away.

Concerning offloading specific experts: I haven't gathered statistics myself, so I don't know how useful that could be. I have seen claims around the Internet that one can gain that way (by offloading often used experts). On the other hand, this is such an obvious thing to do but has not become widely used, so my guess is that this may not be really true. The term "expert" is kind of misleading in the sense that it kind of implies that a given set of experts will be active when dealing with a given kind of context. But this is absolutely not true. If you process a paragraph of, say, 500 tokens on some specific topic, you will observe that basically all "experts" were active at least once.

> 👤 **ThomasBaruzier** replied on **2025-03-09** at **14:28:25**
> 
> >  You maybe better of getting an imatrix from somewhere else.
> 
> I tried using one from [Bartowski's repo](https://huggingface.co/bartowski/DeepSeek-R1-GGUF/blob/main/DeepSeek-R1.imatrix) and [mradermacher's repo](https://huggingface.co/mradermacher/DeepSeek-R1-i1-GGUF/blob/main/imatrix.dat)
> 
> Unfortunately, I get this error with the following command:
> 
> `CMD | '/home/user/files/ai/llama/ik_llama.cpp/llama-quantize' --imatrix '/home/user/nvme/gguf/DeepSeek-R1/imatrix.dat' '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf' '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-iq1_s_r4.gguf' 'iq1_s_r4' '32'`
> 
> ```
> Missing importance matrix for tensor blk.0.attn_v_b.weight in a very low-bit quantization
> ```
> 
> <details>
> <summary>Full logs</summary>
> 
> ```
> Skipping F16 as it already exists: /home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf
> Skipping imatrix as it already exists: /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat
> CMD | '/home/user/files/ai/llama/ik_llama.cpp/llama-quantize' --imatrix '/home/user/nvme/gguf/DeepSeek-R1/imatrix.dat' '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf' '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-iq1_s_r4.gguf' 'iq1_s_r4' '32'
> load_imatrix: imatrix dataset='/training_data/calibration_datav3.txt'
> load_imatrix: loaded 720 importance matrix entries from /home/user/nvme/gguf/DeepSeek-R1/imatrix.dat computed on 124 chunks
> prepare_imatrix: have 720 importance matrix entries
> main: build = 1 (7bdbf99)
> main: built with cc (GCC) 14.2.1 20250207 for x86_64-pc-linux-gnu
> main: quantizing '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf' to '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-iq1_s_r4.gguf' as IQ1_S_R4 using 32 threads
> llama_model_loader: loaded meta data with 44 key-value pairs and 1147 tensors from /home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf (version GGUF V3 (latest))
> llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
> llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
> llama_model_loader: - kv   1:                               general.type str              = model
> llama_model_loader: - kv   2:                               general.name str              = DeepSeek R1 Bf16
> llama_model_loader: - kv   3:                         general.size_label str              = 256x21B
> llama_model_loader: - kv   4:                               general.tags arr[str,1]       = ["text-generation"]
> llama_model_loader: - kv   5:                      deepseek2.block_count u32              = 61
> llama_model_loader: - kv   6:                   deepseek2.context_length u32              = 163840
> llama_model_loader: - kv   7:                 deepseek2.embedding_length u32              = 7168
> llama_model_loader: - kv   8:              deepseek2.feed_forward_length u32              = 18432
> llama_model_loader: - kv   9:             deepseek2.attention.head_count u32              = 128
> llama_model_loader: - kv  10:          deepseek2.attention.head_count_kv u32              = 128
> llama_model_loader: - kv  11:                   deepseek2.rope.freq_base f32              = 10000.000000
> llama_model_loader: - kv  12: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
> llama_model_loader: - kv  13:                deepseek2.expert_used_count u32              = 8
> llama_model_loader: - kv  14:                          general.file_type u32              = 1
> llama_model_loader: - kv  15:        deepseek2.leading_dense_block_count u32              = 3
> llama_model_loader: - kv  16:                       deepseek2.vocab_size u32              = 129280
> llama_model_loader: - kv  17:            deepseek2.attention.q_lora_rank u32              = 1536
> llama_model_loader: - kv  18:           deepseek2.attention.kv_lora_rank u32              = 512
> llama_model_loader: - kv  19:             deepseek2.attention.key_length u32              = 192
> llama_model_loader: - kv  20:           deepseek2.attention.value_length u32              = 128
> llama_model_loader: - kv  21:       deepseek2.expert_feed_forward_length u32              = 2048
> llama_model_loader: - kv  22:                     deepseek2.expert_count u32              = 256
> llama_model_loader: - kv  23:              deepseek2.expert_shared_count u32              = 1
> llama_model_loader: - kv  24:             deepseek2.expert_weights_scale f32              = 2.500000
> llama_model_loader: - kv  25:              deepseek2.expert_weights_norm bool             = true
> llama_model_loader: - kv  26:               deepseek2.expert_gating_func u32              = 2
> llama_model_loader: - kv  27:             deepseek2.rope.dimension_count u32              = 64
> llama_model_loader: - kv  28:                deepseek2.rope.scaling.type str              = yarn
> llama_model_loader: - kv  29:              deepseek2.rope.scaling.factor f32              = 40.000000
> llama_model_loader: - kv  30: deepseek2.rope.scaling.original_context_length u32              = 4096
> llama_model_loader: - kv  31: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
> llama_model_loader: - kv  32:                       tokenizer.ggml.model str              = gpt2
> llama_model_loader: - kv  33:                         tokenizer.ggml.pre str              = deepseek-v3
> llama_model_loader: - kv  34:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
> llama_model_loader: - kv  35:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
> llama_model_loader: - kv  36:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
> llama_model_loader: - kv  37:                tokenizer.ggml.bos_token_id u32              = 0
> llama_model_loader: - kv  38:                tokenizer.ggml.eos_token_id u32              = 1
> llama_model_loader: - kv  39:            tokenizer.ggml.padding_token_id u32              = 1
> llama_model_loader: - kv  40:               tokenizer.ggml.add_bos_token bool             = true
> llama_model_loader: - kv  41:               tokenizer.ggml.add_eos_token bool             = false
> llama_model_loader: - kv  42:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
> llama_model_loader: - kv  43:               general.quantization_version u32              = 2
> llama_model_loader: - type  f32:  361 tensors
> llama_model_loader: - type  f16:  786 tensors
> ================================ Have weights data with 720 entries
> [   1/1147]                    token_embd.weight - [ 7168, 129280,     1,     1], type =    f16, 
> ====== llama_model_quantize_internal: did not find weights for token_embd.weight
> converting to q8_0 .. size =  1767.50 MiB ->   938.98 MiB
> [   2/1147]               blk.0.attn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
> [   3/1147]                blk.0.ffn_down.weight - [18432,  7168,     1,     1], type =    f16, converting to iq3_k_r4 .. size =   252.00 MiB ->    54.14 MiB
> [   4/1147]                blk.0.ffn_gate.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq3_k_r4 .. size =   252.00 MiB ->    54.14 MiB
> [   5/1147]                  blk.0.ffn_up.weight - [ 7168, 18432,     1,     1], type =    f16, converting to iq3_k_r4 .. size =   252.00 MiB ->    54.14 MiB
> [   6/1147]                blk.0.ffn_norm.weight - [ 7168,     1,     1,     1], type =    f32, size =    0.027 MB
> [   7/1147]          blk.0.attn_kv_a_norm.weight - [  512,     1,     1,     1], type =    f32, size =    0.002 MB
> [   8/1147]           blk.0.attn_kv_a_mqa.weight - [ 7168,   576,     1,     1], type =    f16, converting to q4_k_r4 .. size =     7.88 MiB ->     2.21 MiB
> [   9/1147]               blk.0.attn_kv_b.weight - [  512, 32768,     1,     1], type =    f16, converting to q4_k_r4 .. size =    32.00 MiB ->     9.00 MiB
> [  10/1147]                blk.0.attn_k_b.weight - [  128, 65536,     1,     1], type =    f16, 
> 
> llama_tensor_get_type : tensor cols 128 x 65536 are not divisible by 256, required for q4_k_r4 - using fallback quantization q5_0
> 
> ====== llama_model_quantize_internal: did not find weights for blk.0.attn_k_b.weight
> converting to q5_0 .. size =    16.00 MiB ->     5.50 MiB
> [  11/1147]                blk.0.attn_v_b.weight - [  512, 16384,     1,     1], type =    f16, 
> ====== llama_model_quantize_internal: did not find weights for blk.0.attn_v_b.weight
> 
> 
> ============================================================
> Missing importance matrix for tensor blk.0.attn_v_b.weight in a very low-bit quantization
> The result will be garbage, so bailing out
> ============================================================
> 
> llama_model_quantize: failed to quantize: Missing importance matrix for tensor blk.0.attn_v_b.weight in a very low-bit quantization
> main: failed to quantize model from '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf'
> ```
> </details>
> 
> But it's not your repo, llama.cpp faces the exact same issue for some reason, with the equivalent command:
> 
> `CMD | '/home/user/files/ai/llama/llama.cpp/llama-quantize' --imatrix '/home/user/nvme/gguf/DeepSeek-R1/imatrix.dat' '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-F16.gguf' '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-iq1_s.gguf' 'iq1_s' '32'`
> 
> For completeness, I used `arcee-ai/DeepSeek-R1-bf16` to create the F16 GGUF using the following command:
> 
> `CMD | python '/home/user/files/ai/llama/ik_llama.cpp/convert_hf_to_gguf.py' '/home/user/nvme/models/DeepSeek-R1-bf16' --outfile '/home/user/storage/quants/gguf/DeepSeek-R1-bf16/DeepSeek-R1-bf16-F16.gguf' --outtype f16`
> 
> ```
> INFO:hf-to-gguf:Model successfully exported to /home/user/storage/quants/gguf/DeepSeek-R1-bf16/DeepSeek-R1-bf16-F16.gguf
> ```
> 
> I'm having a hard time figuring out what I did wrong to end up having these issues. By any chance, would you have an idea about what is going on?
> 
> ---
> 
> > On the other hand, this is such an obvious thing to do but has not become widely used, so my guess is that this may not be really true.
> 
> I guess I could try making stats about experts usage and see what happens. Even so the distribution of tokens accross experts is supposed to be even, nothing said that some experts could be used a little bit more than others, just like what happens when creating an imatrix for the model?
> 
> ---
> 
> Finally, thanks for all the other precious explanations. I just started making the imatrix for R1 using mainline llama.cpp, brb.

> 👤 **ikawrakow** replied on **2025-03-09** at **14:32:32**
> 
> Try adding `--ignore-imatrix-rules` to your `quantize` command.

> 👤 **ThomasBaruzier** replied on **2025-03-09** at **14:46:11**
> 
> So far so good, but the errors `did not find weights for blk.0.attn_k_b.weight` and `did not find weights for blk.0.attn_v_b.weight` are persisting across every layer quantized so far (0 though 7 for now). I don't know enough to tell, but wouldn't that mean that this is going to be equal to a non-imatrix quant?

> 👤 **ikawrakow** replied on **2025-03-09** at **14:47:20**
> 
> Explanation: the imatrix you use has been computed with standard attention. For MLA one adds two additional tensors (` attn_v_b` and `attn_k_b`). As these were not present during the imatrix calculation, they never got data. In mainline you cannot quantize a low-bit model with such imatrix. Here you can do it by adding `--ignore-imatrix-rules` to the command.

> 👤 **ikawrakow** replied on **2025-03-09** at **14:49:44**
> 
> > but wouldn't that mean that this is going to be equal to a non-imatrix quant
> 
> Only these two tensors (in each layer) will be quantized without imatrix. I see in the log they are quantized with `Q5_0`. This is not ideal (`Q5_K` would have been better), but at 5 bits the gain from having an imatrix is quite modest.

> 👤 **ikawrakow** replied on **2025-03-09** at **14:52:42**
> 
> If you are using the latest `ik_llama.cpp`, you can overwrite the `Q5_0` choice for these tensors by using
> ```
> --custom-q "\.attn_k_b\.weight=Q5_K,\.attn_v_b\.weight=Q5_K"
> ```

> 👤 **ThomasBaruzier** replied on **2025-03-09** at **14:53:50**
> 
> Wouldn't that mean I should be better off trying again making the imatrix myself with this repo for a higher quality result? Or, maybe, do these tensors not having any imatrix data have a negligible impact on the conversion?
> 
> Edit: I guess negligible looking at your latest answers

> 👤 **ThomasBaruzier** replied on **2025-03-09** at **15:27:39**
> 
> There is an issue when adding the `custom-q` argument:
> 
> `'./ik_llama.cpp/llama-quantize' --imatrix 'imatrix.dat' --token-embedding-type q8_0 --custom-q '\.attn_k_b\.weight=Q5_K,\.attn_v_b\.weight=Q5_K' --ignore-imatrix-rules 'DeepSeek-R1-F16.gguf' 'DeepSeek-R1-IQ1_S_R4.gguf' 'IQ1_S_R4' '32'`
> 
> ```
> Invalid quantization type 'Q5_K' in custom quantization input \.attn_k_b\.weight=Q5_K
> ```
> 
> Simplifying to commands like `--custom-q "\.attn_v_b\.weight=17"` or `--custom-q "test=Q4_0"` does not help. The error is thrown in .04s, before the model had a chance to be read.

> 👤 **ikawrakow** replied on **2025-03-09** at **16:15:56**
> 
> Sorry, it is `q5_K`, to `Q5_K`. It needs to match the quantization name in `ggml.c`.

> 👤 **ThomasBaruzier** replied on **2025-03-09** at **16:37:29**
> 
> Seems to work, thanks!

---

👤 **ikawrakow** commented on **2025-03-09** at **08:05:31**

> Slightly offtopic but, how does the imatrix command here handle the 3 attention tensors?

You calculate the imatrix with MLA enabled (and no FA, because this skips one of the activations). This gives you imatrix data for `wk_b` and `wv_b`. As `wv_b` is just the low half of `wkv_b`, the imatrix data for these two is the same. It is very easy to add this to the quantization function. I haven't done that because I don't have the concept of many MLA imatrix data files to be floating around the Internet. But if I'm wrong, let me know, and I'll put that in.

For imatrix data computed with standard attention, imatrix data for `wkv_b` apply to `wv_b` (see above). So, the only tensor left that does not have imatrix data is `wk_b`, which is the transposed version of the upper half of `wkv_b`. I don't think this is a big issue because one shouldn't be using low-bit quantization for `wk_b`, and once you go to `Q5_K` or above, there is barely any difference between quantization quality with and without imatrix.

---

👤 **ThomasBaruzier** commented on **2025-03-10** at **18:19:24**

So here's what I came up with following your instructions:

`custom.sh`:
```sh
#!/bin/bash

cd /home/user/nvme/gguf/DeepSeek-R1
rm -f DeepSeek-R1-custom.gguf

custom="
# Token embedding and output tensors
token_embd\.weight=q8_0
output\.weight=q6_K
output_norm\.weight=q5_K

# First 3 dense layers (GPU0)
blk\.[0-2]\..*=q5_K

# Layers 3-4 (GPU0) - MoE experts
blk\.[3-4]\.ffn_down_exps\.weight=iq4_xs
blk\.[3-4]\.ffn_gate_exps\.weight=iq2_xxs
blk\.[3-4]\.ffn_up_exps\.weight=iq2_xxs

# Layers 5-11 (GPU1) - MoE experts
blk\.[5-9]\.ffn_down_exps\.weight=iq3_xxs
blk\.[5-9]\.ffn_gate_exps\.weight=iq2_xxs
blk\.[5-9]\.ffn_up_exps\.weight=iq2_xxs
blk\.1[0-1]\.ffn_down_exps\.weight=iq3_xxs
blk\.1[0-1]\.ffn_gate_exps\.weight=iq2_xxs
blk\.1[0-1]\.ffn_up_exps\.weight=iq2_xxs

# Layers 12-18 (GPU2) - MoE experts
blk\.1[2-8]\.ffn_down_exps\.weight=iq3_xxs
blk\.1[2-8]\.ffn_gate_exps\.weight=iq2_xxs
blk\.1[2-8]\.ffn_up_exps\.weight=iq2_xxs

# Layers 19-60 (CPU) - MoE experts
blk\.19\.ffn_down_exps\.weight=iq2_k_r4
blk\.[2-5][0-9]\.ffn_down_exps\.weight=iq2_k_r4
blk\.60\.ffn_down_exps\.weight=iq2_k_r4
blk\.19\.ffn_gate_exps\.weight=iq2_xxs_r4
blk\.[2-5][0-9]\.ffn_gate_exps\.weight=iq2_xxs_r4
blk\.60\.ffn_gate_exps\.weight=iq2_xxs_r4
blk\.19\.ffn_up_exps\.weight=iq2_xxs_r4
blk\.[2-5][0-9]\.ffn_up_exps\.weight=iq2_xxs_r4
blk\.60\.ffn_up_exps\.weight=iq2_xxs_r4

# All attention tensors for MoE layers (3-60)
blk\.[3-9]\.attn_.*=q5_K
blk\.[1-5][0-9]\.attn_.*=q5_K
blk\.60\.attn_.*=q5_K

# Norm weights and bias for MoE layers (3-60)
blk\.[3-9]\.ffn_norm\.weight=q5_K
blk\.[1-5][0-9]\.ffn_norm\.weight=q5_K
blk\.60\.ffn_norm\.weight=q5_K
blk\.[3-9]\.exp_probs_b\.bias=q5_K
blk\.[1-5][0-9]\.exp_probs_b\.bias=q5_K
blk\.60\.exp_probs_b\.bias=q5_K

# Shared experts weights for MoE layers (3-60)
blk\.3\.ffn_.*shexp\.weight=q5_K
blk\.[4-9]\.ffn_.*shexp\.weight=q5_K
blk\.[1-5][0-9]\.ffn_.*shexp\.weight=q5_K
blk\.60\.ffn_.*shexp\.weight=q5_K
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

/home/user/files/ai/llama/ik_llama.cpp/llama-quantize \
  --imatrix imatrix.dat \
  --token-embedding-type q8_0 \
  --output-tensor-type q6_K \
  --ignore-imatrix-rules \
  --custom-q "$custom" \
  DeepSeek-R1-F16.gguf DeepSeek-R1-custom.gguf Q6_K 32
```

`server.sh` (CUDA0 and CUDA1 switched because of PCIe speeds):
```sh
#!/bin/bash

/home/user/files/ai/llama/ik_llama.cpp/llama-server \
  -m /home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-custom.gguf \
  --api-key "$LOCAL_API_KEY" \
  --host 0.0.0.0 \
  --port 5000 \
  -c 8192 \
  -t 16 \
  -sm layer \
  -mg 1 \
  -mla 2 \
  -fmoe \
  -ot "output\.weight=CUDA1" \
  -ot "output_norm\.weight=CUDA1" \
  -ot "token_embd\.weight=CUDA1" \
  -ot "blk\.[0-4]\..*=CUDA1" \
  -ot "blk\.[3-9]\.attn_.*=CUDA1" \
  -ot "blk\.[1-5][0-9]\.attn_.*=CUDA1" \
  -ot "blk\.60\.attn_.*=CUDA1" \
  -ot "blk\.[3-9]\.ffn_norm\.weight=CUDA1" \
  -ot "blk\.[1-5][0-9]\.ffn_norm\.weight=CUDA1" \
  -ot "blk\.60\.ffn_norm\.weight=CUDA1" \
  -ot "blk\.[3-9]\.ffn_.*shexp\.weight=CUDA1" \
  -ot "blk\.[1-5][0-9]\.ffn_.*shexp\.weight=CUDA1" \
  -ot "blk\.60\.ffn_.*shexp\.weight=CUDA1" \
  -ot "blk\.[5-9]\.ffn_down_exps\.weight=CUDA0" \
  -ot "blk\.[5-9]\.ffn_gate_exps\.weight=CUDA0" \
  -ot "blk\.[5-9]\.ffn_up_exps\.weight=CUDA0" \
  -ot "blk\.1[0-1]\.ffn_down_exps\.weight=CUDA0" \
  -ot "blk\.1[0-1]\.ffn_gate_exps\.weight=CUDA0" \
  -ot "blk\.1[0-1]\.ffn_up_exps\.weight=CUDA0" \
  -ot "blk\.1[2-8]\.ffn_down_exps\.weight=CUDA2" \
  -ot "blk\.1[2-8]\.ffn_gate_exps\.weight=CUDA2" \
  -ot "blk\.1[2-8]\.ffn_up_exps\.weight=CUDA2" \
```

Even though I haven't spent much time playing with the settings, the speed is already at 7.1-7.3 tok/s with very short prompt and generation, 6.6-6.8tok/s with a few hundred tokens and 6.2-6.4tok/s for 1k. Also, a ~1k token ingestion goes at 35-40tok/s. I don't really know if those numbers make sense given the setup, but I am already very happy with these speeds.

VRAM use is 23.59GB on the main GPU and 23.00GB on the other two. So 2.3/2.4GB is free to play with for longer context.

Next steps:
- play with kv cache quants and optimizations (would you have any recommendations?)
- run `llama-bench` and `llama-perplexity`

Also, it seems that I can't use `-ot` with llama-perplexity (haven't tried with `llama-bench`)

Edit: Main GPU usage is at 25% and other cards are at 0% when generating. Is it because of the RAM speed limitations?

> 👤 **ikawrakow** replied on **2025-03-11** at **06:43:37**
> 
> If you remove the `-fmoe`, does it still run everything on the main GPU?

> 👤 **ThomasBaruzier** replied on **2025-03-11** at **16:30:22**
> 
> Great! Thank you for all the advice, once again.
> 
> It seems that I forgot a backslash, `llama-bench` and `llama-perplexity` correctly uses the `-ot` argument, oops.
> 
> `llama-perplexity` works well, but I still have some issues with llama-bench, and the error is not very descriptive:
> ```
> ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
> ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
> ggml_cuda_init: found 3 CUDA devices:
>   Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 1: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
>   Device 2: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes
> | model                          |       size |     params | backend    | ngl |   main_gpu | mla | fmoe |          test |              t/s |
> | ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --: | ---: | ------------: | ---------------: |
> main: error: failed to load model '/home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-custom.gguf'
> ```
> 
> <details>
> <summary>Full command</summary>
> 
> ```sh
> #!/bin/bash
> 
> /home/user/files/ai/llama/ik_llama.cpp/llama-bench \
>   -m /home/user/nvme/gguf/DeepSeek-R1/DeepSeek-R1-custom.gguf \
>   -p 1024 \
>   -n 128 \
>   -t 16 \
>   -sm layer \
>   -mg 1 \
>   -mla 2 \
>   -fmoe 1 \
>   -ot "output\.weight=CUDA1" \
>   -ot "output_norm\.weight=CUDA1" \
>   -ot "token_embd\.weight=CUDA1" \
>   -ot "blk\.[0-4]\..*=CUDA1" \
>   -ot "blk\.[3-9]\.attn_.*=CUDA1" \
>   -ot "blk\.[1-5][0-9]\.attn_.*=CUDA1" \
>   -ot "blk\.60\.attn_.*=CUDA1" \
>   -ot "blk\.[3-9]\.ffn_norm\.weight=CUDA1" \
>   -ot "blk\.[1-5][0-9]\.ffn_norm\.weight=CUDA1" \
>   -ot "blk\.60\.ffn_norm\.weight=CUDA1" \
>   -ot "blk\.[3-9]\.ffn_.*shexp\.weight=CUDA1" \
>   -ot "blk\.[1-5][0-9]\.ffn_.*shexp\.weight=CUDA1" \
>   -ot "blk\.60\.ffn_.*shexp\.weight=CUDA1" \
>   -ot "blk\.[5-9]\.ffn_down_exps\.weight=CUDA0" \
>   -ot "blk\.[5-9]\.ffn_gate_exps\.weight=CUDA0" \
>   -ot "blk\.[5-9]\.ffn_up_exps\.weight=CUDA0" \
>   -ot "blk\.1[0-1]\.ffn_down_exps\.weight=CUDA0" \
>   -ot "blk\.1[0-1]\.ffn_gate_exps\.weight=CUDA0" \
>   -ot "blk\.1[0-1]\.ffn_up_exps\.weight=CUDA0" \
>   -ot "blk\.1[2-8]\.ffn_down_exps\.weight=CUDA2" \
>   -ot "blk\.1[2-8]\.ffn_gate_exps\.weight=CUDA2" \
>   -ot "blk\.1[2-8]\.ffn_up_exps\.weight=CUDA2" \
> ```
> </details>
> 
> Edit: using `--verbose`, I get: `llama_model_load: error loading model: failed to allocate buffer`. Is it allocating more context than it should? There is no `-c` equivalent (other than values in `-p` and `-n`), it seems.
> 
> When removing `-fmoe`, the GPU usage is still centralized on the main GPU, with 20-25% usage at 130-140w, while the other cards stay at 0% at ~100w.
> 
> Finally, using `-fa` slows down the prompt ingestion speeds to 28tok/s. Generation seems to not be affected. I've already seen this behavior on mainline when using `fa` with CPU offloading.

> 👤 **ikawrakow** replied on **2025-03-11** at **16:36:21**
> 
> You can add `-v` to `llama-bench` to see why it fails to load the model.

> 👤 **ThomasBaruzier** replied on **2025-03-11** at **16:57:45**
> 
> I get: `llama_model_load: error loading model: failed to allocate buffer`. Is it trying to allocate the full 128k context? There is no `-c` equivalent (other than values in `-p` and `-n`), it seems.

> 👤 **ikawrakow** replied on **2025-03-11** at **18:04:04**
> 
> No, it should use a context given by the sum of `-p` and `-n`.

---

👤 **ThomasBaruzier** commented on **2025-03-13** at **14:22:08**

Here are some early results for wiki.test:
IQ1_S unsloth (1.67 BPW): 5.5749 +/- 0.03545
IQ1_M unsloth (2.01 BPW): 4.7238 +/- 0.02859
IQ2_XXS custom (2.34 BPW): 4.1059 +/- 0.02411

PPL for IQ2_XXS unsloth (size equivalent with your custom quant) and IQ1_S_R4/IQ1_M_R4 are still running.

In the meantime, is there any reason why you didn't recommend your new SOTA quant types like IQ2_K, or IQ4_KSS?
Or, are these not quant types but rather full quants consisting of an improved mixture of already existing quants types? (Edit: seems like new quant types that  are fast on CPU as well, wow https://github.com/ikawrakow/ik_llama.cpp/discussions/8)

I see you added Q8 KV cache for MLA2. Nice! I will test perfs after the PPL tests.

Finally, I stumbled upon this paper I thought you might find interesting: https://arxiv.org/pdf/2503.05840
TLDR no more V cache as it can be retrieved from K cache with full accuracy, supposedly compatible with quantization and FA, with nice speed benefits.
Edit: I don't think it could apply here: "Slim attention is somewhat similar to DeepSeek’s multi-head latent attention"

---

👤 **ikawrakow** commented on **2025-03-13** at **15:15:04**

> In the meantime, is there any reason why you didn't recommend your new SOTA quant types like IQ2_K, or IQ4_KSS?

Someone else was observing issues (NaNs) with `IQ4_KSS` and `IQ4_K` and I wasn't sure where the problem is. In the meantime I know that the problem is with using those on CUDA for the experts weights. These quants do not have quantized matrix multiplication kernels (a.k.a. MMQ), so for them on CUDA matrix multiplications are done by first dequantizing to `fp16` and then using cuBLAS `fp16` GEMM. It turns out, for DeepSeek-R1 this does not work, the `fp16` range is not sufficient to accommodate the result.  Hence, these quants cannot be used on CUDA for the DeepSeek models. But if you want to use them for experts that are computed on the CPU, this is perfectly fine. `IQ4_K` in particular is much better than any other 4-bit quantization type for the models I have tested (all LLaMA-3 models apart from the 405B one, Gemma2, Qwen-2.5, Mistral-Nemo, etc.). `IQ4_KSS` does not have an `_r4` variant. The bit packing is very awkward to achieve exactly 4 bpw, so implemnting the `_r4` version will be a bit of a nightmare, so  I keep postponing to do it). `IQ4_KS` (same size as `IQ4_XS`) is a bit of hit-or-miss. For some models it is quite a bit better than `IQ4_XS`, but for some models it is only on par (and it has a slightly lower inference performance than `IQ4_XS`). `IQ3_K` is slighty better than `IQ3_S` with the same bpw, but it is much faster on the CPU. `IQ2_K` is about in the middle between `IQ2_XS` and `IQ2_S` in terms of size and quality, but should also be much faster. If you feel like experimenting with these, I would be curious to learn about their performance for DeepSeekR1.

> Finally, I stumbled upon this paper I thought you might find interesting: https://arxiv.org/pdf/2503.05840

Yes, I know about this paper. MLA=2 does the same thing, there is only K cache and the `V` tensor gets computed from that (in different ways, depending on context). The only difference is that with MLA one does not need to compute $W_K^{-1}$ matrix, the equivalent is provided by the DeepSeek $W_{KV}$ tensor. It sounds nice in theory, but there is the theory and than there is the practice. In practice one needs to also consider compute buffers as intermediate results need to go somewhere, and the fact that counting multiply-adds is just a very rough estimate of actual performance, which also depends on memory access patterns, matrix shapes and sizes, etc. IIRC, the main factor that made me reluctant to spend the time implementing something along these lines is the fact that the benefit mostly goes away for GQA, which most models use these days.

> 👤 **ikawrakow** replied on **2025-03-13** at **17:17:46**
> 
> > So what's the difference between MLA=2 and "something along these lines"?
> 
> MLA=2 is specific to the DeepSeek attention mechanism. "Something along these lines" would be a generic implementation for any MHA model.

---

👤 **ikawrakow** commented on **2025-03-15** at **09:31:42**

> PPL for IQ2_XXS unsloth (size equivalent with your custom quant) and IQ1_S_R4/IQ1_M_R4 are still running.

Do you have the results now? I'm curious to know.