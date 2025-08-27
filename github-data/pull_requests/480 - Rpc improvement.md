## 🔀 [Pull Request #480](https://github.com/ikawrakow/ik_llama.cpp/pull/480) - Rpc improvement

| **Author** | `firecoperana` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `rpc_improvement` |
| **Target Branch** | `main` |
| **Created** | 2025-06-01 |
| **Updated** | 2025-06-25 |
| **Merged** | 2025-06-08 |

---

## 📄 Description

Include various improvement of rpc from mainline including: 
1. adding rpc backend to override tensor option
2. add argument of number of threads in the cpu rpc backend
3. cache model locally in rpc 
4. no delay for sending tcp
5. various bugs fix. 

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [ ] Medium
  - [x] High

---

## 💬 Conversation

👤 **saood06** commented on **2025-06-01** at **02:58:58**

Has this been tested? If so with what models and backends and what configurations. I attempted a similar PR a while ago, see [#193](https://github.com/ikawrakow/ik_llama.cpp/issues/193) and when tested it did not work with Qwen2.5 72B since on mainline the PR that added "non-512 aligned tensors" was created to add support for that model. I also found that using KV cache quantization still did not work with RPC with or without [#193](https://github.com/ikawrakow/ik_llama.cpp/issues/193).

---

👤 **ikawrakow** commented on **2025-06-01** at **05:43:47**

I don't use RPC, so need other people to confirm that this works.

---

👤 **saood06** commented on **2025-06-01** at **06:20:06**

> I don't use RPC, so need other people to confirm that this works.

I don't mind testing and reviewing this but before I do, I want to know what new models/configurations support this PR @firecoperana tested and saw benefit from. I deleted pretty much all the models I previously used when RPC testing when trying to bring support parity up to mainline.

---

👤 **firecoperana** commented on **2025-06-01** at **12:45:44**

I tested various quants of Deepseek v2.5, v3, v3 0324 models and it works. V3 0324 is the one with MLA support from mainline. Didn't test other models as I don't use them on this repo.

---

👤 **saood06** commented on **2025-06-01** at **12:53:03**

> I tested various quants of Deepseek v2.5, v3, v3 0324 models and it works. V3 0324 is the one with MLA support from mainline. Didn't test other models as I don't use them on this repo.

Did you test with `-ot` or cache quantization? Do you mind sharing performance and what hardware you used?

---

👤 **firecoperana** commented on **2025-06-01** at **13:08:24**

My main machine is 3090 with 128GB ddr4. I did -ot to override individual expert tensors to my other machines with ddr4 3000mhz ram and 3060, and with --cache-type-k q8_0 and batch size of 512, in which case I can load the whole model into either vram and ram. I use cpu RPC backend to use ram from remote machines. For Deepseek V3 Q2_K_XL, I can get 10 it/s for pp and 3 it/s for inferencing. Deepseek V2.5 Q4 is about 6-7 it/s for inferencing.

---

👤 **saood06** commented on **2025-06-01** at **13:16:15**

> My main machine is 3090 with 128GB ddr4. I did -ot to override individual expert tensors to my other machines with ddr4 3000mhz ram and 3060, and with --cache-type-k q8_0 and batch size of 512, in which case I can load the whole model into either vram and ram. I use cpu RPC backend to use ram from remote machines. For Deepseek V3 Q2_K_XL, I can get 10 it/s for pp and 3 it/s for inferencing. Deepseek V2.5 Q4 is about 6-7 it/s for inferencing.

Thank you for the details. For now I'll do some testing on Deepseek, with an RPC backend on my 3090 and `-ot`, with the rest of the model in RAM on the DDR4 server I usually use for inference.

For reference with my pure IQ4_K_R4 I get similar speeds you get with RPC for both PP and TG so hopefully with RPC it can improve (and since those quants are now supported on CUDA, I won't need to make a new quant).

---

👤 **firecoperana** commented on **2025-06-01** at **13:24:34**

Be sure to set -t n -c in cpu backend, where n is the number of threads you want the tensors in ram to use. -c is to load tensors from local files next time. This is useful if you have slow LAN transfer speed.

---

👤 **ikawrakow** commented on **2025-06-02** at **09:25:04**

No user feedback here, so new strategy: I'll merge this tomorrow. If we don't get bug reports, all is good. If we do get bug reports, all is good too because we know that it needs further work.

---

👤 **saood06** commented on **2025-06-02** at **10:15:59**

> No user feedback here, so new strategy: I'll merge this tomorrow. If we don't get bug reports, all is good. If we do get bug reports, all is good too because we know that it needs further work.

I haven't found the time to test this, but I do plan to, in the next few days. (I've already downloaded a few of the models I plan to to use to test this alongside Deepseek). Either way I'll give some feedback even if it has already been merged by then.

---

👤 **ikawrakow** commented on **2025-06-08** at **11:52:02**

I get build errors after merging this PR, so reverted. Please fix and resubmit.

---

👤 **firecoperana** commented on **2025-06-08** at **13:42:29**

> I get build errors after merging this PR, so reverted. Please fix and resubmit.

What's the error? Does the error happen only when you set DGGML_RPC=OFF?

---

👤 **ikawrakow** commented on **2025-06-08** at **13:48:08**

```
/home/iwan/other/ik_llama.cpp/common/common.cpp: In function ‘bool gpt_params_find_arg(int, char**, const std::string&, gpt_params&, int&, bool&)’:
/home/iwan/other/ik_llama.cpp/common/common.cpp:1013:13: error: ‘ggml_backend_rpc_buffer_type’ was not declared in this scope; did you mean ‘ggml_backend_cpu_buffer_type’?
 1013 |             ggml_backend_rpc_buffer_type(server.c_str());
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |             ggml_backend_cpu_buffer_type
/home/iwan/other/ik_llama.cpp/common/common.cpp:1016:9: error: ‘ggml_backend_rpc_buffer_type’ was not declared in this scope; did you mean ‘ggml_backend_cpu_buffer_type’?
 1016 |         ggml_backend_rpc_buffer_type(servers.c_str());
      |         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |         ggml_backend_cpu_buffer_type
```

This is with `-DGGML_RPC=OFF`

---

👤 **firecoperana** commented on **2025-06-08** at **14:23:46**

Fixed

---

👤 **saood06** commented on **2025-06-22** at **20:52:33**

Finally got around to testing this. It seems functional (sweep-bench testing only), but I couldn't get any performance advantage from offloading Deepseek-V3 based models via RPC to my 3090. I know when I tested that on mainline I also noticed a performance regression (that went up with the more I offloaded).

(I ran with `./llama-sweep-bench -m /mnt/sda/DeepseekR1_0528/DeepseekR1_0528-IQ4_K_R4_ATT1.gguf --numa distribute -t 48 -mla 3 -fa -fmoe -c 4096 --rpc 10.0.0.250:50052 -ot exps=CPU -ngl 99 --warmup-batch`)

Measuring at low context values, PP drops from ~10.5 to ~4.5, TG drops from ~3.3 to ~1.

I may revisit when I eventually get an infiniband connection between the two computers and see if that helps.

---

👤 **firecoperana** commented on **2025-06-23** at **00:37:44**

> Finally got around to testing this. It seems functional (sweep-bench testing only), but I couldn't get any performance advantage from offloading Deepseek-V3 based models via RPC to my 3090. I know when I tested that on mainline I also noticed a performance regression (that went up with the more I offloaded).
> 
> (I ran with `./llama-sweep-bench -m /mnt/sda/DeepseekR1_0528/DeepseekR1_0528-IQ4_K_R4_ATT1.gguf --numa distribute -t 48 -mla 3 -fa -fmoe -c 4096 --rpc 10.0.0.250:50052 -ot exps=CPU -ngl 99 --warmup-batch`)
> 
> Measuring at low context values, PP drops from ~10.5 to ~4.5, TG drops from ~3.3 to ~1.
> 
> I may revisit when I eventually get an infiniband connection between the two computers and see if that helps.

Can you add --tensor-split 0,99? This will make sure all non-expert layers are offloaded to RPC machine. You could try to offload expert layers to your 3090 with blk.(12|13).ffn_.*_exps=RPC[10.0.0.250:50052] to fully use 3090's VRAM. You could also try to use 3090 as your main GPU and your server for offloading expert layers. Your speed drop is too much.

---

👤 **saood06** commented on **2025-06-23** at **01:07:12**

> Can you add --tensor-split 0,99? This will make sure all non-expert layers are offloaded to RPC machine. You could try to offload expert layers to your 3090 with blk.(12|13).ffn_.*_exps=RPC[10.0.0.250:50052] to fully use 3090's VRAM. 

I ran a low context test, but I would still care about maximizing usable context (and I would use far more than 4k). Including the log below so you can see what it offloaded.

<details>
<summary>Output log from `-ot`</summary>

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

</details>

>You could also try to use 3090 as your main GPU and your server for offloading expert layers. Your speed drop is too much.

That would mean transferring hundreds of gigs over what is currently a gigabit connection (I know I could then use the `-c` feature you suggest). I might test that later.

There definitely was a lot of network traffic happening during inference. I don't remember if that is normal from when I used to RPC with dense models and simple RPC offloading which netted me a benefit (even when ran like this).

---

👤 **HariboApfel** commented on **2025-06-25** at **08:06:14**

i am encountering abysmal performance with ik_llama and rpc. I "assume" its rpc related.

i am using 
`./llama-cli --version
version: 3770 (b5f2f001)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu`

with following build flags 

`cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_RPC=ON -DGGML_SCHED_MAX_COPIES=1`

across 3 hosts with each 4 A5000 GPUS with 24gb vram each
the hosts are only connected via switch

`./ik_llama.cpp/build/bin/llama-cli \
  --rpc "$RPC_SERVERS" \
  --model models/ubergarm/DeepSeek-R1-0528-GGUF/IQ2_K_R4/DeepSeek-R1-0528-IQ2_K_R4-00001-of-00005.gguf \
  --threads 48 \
  --n-gpu-layers 99 \
  --temp 0.6 \
  --top_p 0.95 \
  --min_p 0.01 \
  --ctx-size 16384 \
  --parallel 1 \
  --flash-attn \
  --verbosity 3 \
  -v \
  -mla 3 -fa \
  -amb 512 \
  -fmoe \
  -ctk q8_0 \
  -ot "\.(3[3-9]|4[0-9]|5[0-9]|6[0-9]|7[0-9]|8[0-9]|9[0-9]|[0-9][0-9][0-9])\.ffn_up_exps.=CPU" \
  --prompt`

with the same -ot and compareable settings on llama.cpp i am running with around 7.4T/s

using

ik_llama.cpp 

`llama_print_timings:        load time =  639497.76 ms
llama_print_timings:      sample time =       0.39 ms /     3 runs   (    0.13 ms per token,  7772.02 tokens per second)
llama_print_timings: prompt eval time =   70116.00 ms /   220 tokens (  318.71 ms per token,     3.14 tokens per second)
llama_print_timings:        eval time =  132398.81 ms /     2 runs   (66199.41 ms per token,     0.02 tokens per second)
llama_print_timings:       total time =  253512.88 ms /   222 tokens`

i get this **66199.41 ms per token,     0.02 tokens per second**

any help would be apprichiated.

---

👤 **ikawrakow** commented on **2025-06-25** at **08:53:42**

I never use RPC, so somebody else should comment.

---

👤 **HariboApfel** commented on **2025-06-25** at **10:17:36**

i got rpc atleast working after redoing the arguments from the 1st post.

using 

`./ik_llama.cpp/build/bin/llama-cli \
  --model models/ubergarm/DeepSeek-R1-0528-GGUF/IQ2_K_R4/DeepSeek-R1-0528-IQ2_K_R4-00001-of-00005.gguf \
  --threads 48 \
  --n-gpu-layers 99 \
  --temp 0.6 \
  --top_p 0.95 \
  --min_p 0.01 \
  --ctx-size 16384 \
  --flash-attn \
  -mla 3 -fa \
  -amb 512 \
  -fmoe \
  -ctk q8_0 \
  -ot "blk\.(1|2|3|4|5|6)\.ffn_.*=CUDA0" \
  -ot "blk\.(7|8|9|10)\.ffn_.*=CUDA1" \
  -ot "blk\.(11|12|13|14)\.ffn_.*=CUDA2" \
  -ot "blk\.(15|16|17|18)\.ffn_.*=CUDA3" \
  --override-tensor exps=CPU \
  --prompt`

to only run it on one host gets me closer to "expected performance"

`llama_print_timings:        load time =   98945.88 ms
llama_print_timings:      sample time =      45.19 ms /   384 runs   (    0.12 ms per token,  8497.27 tokens per second)
llama_print_timings: prompt eval time =    5969.18 ms /   224 tokens (   26.65 ms per token,    37.53 tokens per second)
llama_print_timings:        eval time =   57680.32 ms /   383 runs   (  150.60 ms per token,     6.64 tokens per second)
llama_print_timings:       total time =   63916.49 ms /   607 tokens`


with RPC

`./ik_llama.cpp/build/bin/llama-cli \
  --rpc "$RPC_SERVERS" \
  --model models/ubergarm/DeepSeek-R1-0528-GGUF/IQ2_K_R4/DeepSeek-R1-0528-IQ2_K_R4-00001-of-00005.gguf \
  --threads 48 \
  --n-gpu-layers 99 \
  --temp 0.6 \
  --top_p 0.95 \
  --min_p 0.01 \
  --ctx-size 16384 \
  --flash-attn \
  -mla 3 -fa \
  -amb 512 \
  -fmoe \
  -ctk q8_0 \
  -ot "blk\.(1|2|3|4|5|6)\.ffn_.*=CUDA0" \
  -ot "blk\.(7|8|9|10)\.ffn_.*=CUDA1" \
  -ot "blk\.(11|12|13|14)\.ffn_.*=CUDA2" \
  -ot "blk\.(15|16|17|18)\.ffn_.*=CUDA3" \
  -ot "blk\.(19|20|21|22)\.ffn_.*=RPC[10.0.0.28:50052]" \
  -ot "blk\.(23|24|25|26)\.ffn_.*=RPC[10.0.0.28:50053]" \
  -ot "blk\.(27|28|29|30)\.ffn_.*=RPC[10.0.0.28:50054]" \
  -ot "blk\.(31|32|33|34)\.ffn_.*=RPC[10.0.0.28:50055]" \
  -ot "blk\.(35|36|37|38)\.ffn_.*=RPC[10.0.0.40:50052]" \
  -ot "blk\.(39|40|41|42)\.ffn_.*=RPC[10.0.0.40:50053]" \
  -ot "blk\.(43|44|45|46)\.ffn_.*=RPC[10.0.0.40:50054]" \
  -ot "blk\.(47|48|49|50)\.ffn_.*=RPC[10.0.0.40:50055]" \
  --override-tensor exps=CPU \
  --prompt`

i get around 5.5T/s
`
llama_print_timings:        load time =  568857.08 ms
llama_print_timings:      sample time =     963.77 ms /  7798 runs   (    0.12 ms per token,  8091.13 tokens per second)
llama_print_timings: prompt eval time =    8689.40 ms /   224 tokens (   38.79 ms per token,    25.78 tokens per second)
llama_print_timings:        eval time = 1420492.95 ms /  7797 runs   (  182.18 ms per token,     5.49 tokens per second)
llama_print_timings:       total time = 1432903.60 ms /  8021 tokens`

which is still less then llama.cpp with the same rpc setting with the unsloth unsloth/DeepSeek-R1-0528-GGUF/UD-Q2_K_XL quant ??

the only real difference would be the -ot setting there- 
for llama.cpp i use

` --cache-type-k q4_0 \
  --threads 48 \
  --n-gpu-layers 99 \
  --prio 3 \
  --temp 0.6 \
  --top_p 0.95 \
  --min_p 0.01 \
  --flash-attn \
  --ctx-size 16384 \
  -ot "\.(3[5-9]|4[0-9]|5[0-9]|6[0-9]|7[0-9]|8[0-9]|9[0-9]|[0-9][0-9][0-9])\.ffn_up_exps.=CPU" \
  -no-cnv \
  --prompt`

giving me 7.5T/s

i would have assumend IK_llama is faster.

---

👤 **ikawrakow** commented on **2025-06-25** at **10:45:43**

You can use Unsloth's UD-Q2_K_XL model (or any model that works with `llama.cpp`) with `ik_llama.cpp` just fine, and that would be more of an apples-to-apples comparison. It would also be useful to use the same cache type if you are after a performance comparison.

---

👤 **firecoperana** commented on **2025-06-25** at **16:12:42**

Also for fair comparison, please compare if allocation of vram buffer and layers for each gpu and cpu are the same as mainline. I use tensor-split to control the exact number of layers for each gpu. And note that ik_llama has different order for tensor split than llama.cpp.