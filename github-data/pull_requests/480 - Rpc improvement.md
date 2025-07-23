### 🔀 [#480](https://github.com/ikawrakow/ik_llama.cpp/pull/480) - Rpc improvement

| **Author** | `firecoperana` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-01 |
| **Updated** | 2025-06-25 |

---

#### Description

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

#### 💬 Conversation

👤 **saood06** commented the **2025-06-01** at **02:58:58**:<br>

Has this been tested? If so with what models and backends and what configurations. I attempted a similar PR a while ago, see #193 and when tested it did not work with Qwen2.5 72B since on mainline the PR that added "non-512 aligned tensors" was created to add support for that model. I also found that using KV cache quantization still did not work with RPC with or without #193.

---

👤 **ikawrakow** commented the **2025-06-01** at **05:43:47**:<br>

I don't use RPC, so need other people to confirm that this works.

---

👤 **saood06** commented the **2025-06-01** at **06:20:06**:<br>

> I don't use RPC, so need other people to confirm that this works.

I don't mind testing and reviewing this but before I do, I want to know what new models/configurations support this PR @firecoperana tested and saw benefit from. I deleted pretty much all the models I previously used when RPC testing when trying to bring support parity up to mainline.

---

👤 **firecoperana** commented the **2025-06-01** at **12:45:44**:<br>

I tested various quants of Deepseek v2.5, v3, v3 0324 models and it works. V3 0324 is the one with MLA support from mainline. Didn't test other models as I don't use them on this repo.

---

👤 **firecoperana** commented the **2025-06-01** at **13:08:24**:<br>

My main machine is 3090 with 128GB ddr4. I did -ot to override individual expert tensors to my other machines with ddr4 3000mhz ram and 3060, and with --cache-type-k q8_0 and batch size of 512, in which case I can load the whole model into either vram and ram. I use cpu RPC backend to use ram from remote machines. For Deepseek V3 Q2_K_XL, I can get 10 it/s for pp and 3 it/s for inferencing. Deepseek V2.5 Q4 is about 6-7 it/s for inferencing.

---

👤 **firecoperana** commented the **2025-06-01** at **13:24:34**:<br>

Be sure to set -t n -c in cpu backend, where n is the number of threads you want the tensors in ram to use. -c is to load tensors from local files next time. This is useful if you have slow LAN transfer speed.

---

👤 **ikawrakow** commented the **2025-06-02** at **09:25:04**:<br>

No user feedback here, so new strategy: I'll merge this tomorrow. If we don't get bug reports, all is good. If we do get bug reports, all is good too because we know that it needs further work.

---

👤 **saood06** commented the **2025-06-02** at **10:15:59**:<br>

> No user feedback here, so new strategy: I'll merge this tomorrow. If we don't get bug reports, all is good. If we do get bug reports, all is good too because we know that it needs further work.

I haven't found the time to test this, but I do plan to, in the next few days. (I've already downloaded a few of the models I plan to to use to test this alongside Deepseek). Either way I'll give some feedback even if it has already been merged by then.

---

👤 **firecoperana** commented the **2025-06-08** at **13:42:29**:<br>

> I get build errors after merging this PR, so reverted. Please fix and resubmit.

What's the error? Does the error happen when you set DGGML_RPC=OFF?

---

👤 **firecoperana** commented the **2025-06-08** at **14:23:46**:<br>

Fixed

---

👤 **saood06** commented the **2025-06-22** at **20:52:33**:<br>

Finally got around to testing this. It seems functional (sweep-bench testing only), but I couldn't get any performance advantage from offloading Deepseek-V3 based models via RPC to my 3090. I know when I tested that on mainline I also noticed a performance regression (that went up with the more I offloaded).

(I ran with `./llama-sweep-bench -m /mnt/sda/DeepseekR1_0528/DeepseekR1_0528-IQ4_K_R4_ATT1.gguf --numa distribute -t 48 -mla 3 -fa -fmoe -c 4096 --rpc 10.0.0.250:50052 -ot exps=CPU -ngl 99 --warmup-batch`)

Measuring at low context values, PP drops from ~10.5 to ~4.5, TG drops from ~3.3 to ~1.

I may revisit when I eventually get an infiniband connection between the two computers and see if that helps.

---

👤 **firecoperana** commented the **2025-06-23** at **00:37:44**:<br>

> Finally got around to testing this. It seems functional (sweep-bench testing only), but I couldn't get any performance advantage from offloading Deepseek-V3 based models via RPC to my 3090. I know when I tested that on mainline I also noticed a performance regression (that went up with the more I offloaded).
> 
> (I ran with `./llama-sweep-bench -m /mnt/sda/DeepseekR1_0528/DeepseekR1_0528-IQ4_K_R4_ATT1.gguf --numa distribute -t 48 -mla 3 -fa -fmoe -c 4096 --rpc 10.0.0.250:50052 -ot exps=CPU -ngl 99 --warmup-batch`)
> 
> Measuring at low context values, PP drops from ~10.5 to ~4.5, TG drops from ~3.3 to ~1.
> 
> I may revisit when I eventually get an infiniband connection between the two computers and see if that helps.

Can you add --tensor-split 0,99? This will make sure all non-expert layers are offloaded to RPC machine. You could try to offload expert layers to your 3090 with blk.(12|13).ffn_.*_exps=RPC[10.0.0.250:50052] to fully use 3090's VRAM. You could also try to use 3090 as your main GPU and your server for offloading expert layers. Your speed drop is too much.

---

👤 **saood06** commented the **2025-06-23** at **01:07:12**:<br>

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

👤 **HariboApfel** commented the **2025-06-25** at **08:06:14**:<br>

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

👤 **ikawrakow** commented the **2025-06-25** at **10:45:43**:<br>

You can use Unsloth's UD-Q2_K_XL model (or any model that works with `llama.cpp`) with `ik_llama.cpp` just fine, and that would be more of an apples-to-apples comparison. It would also be useful to use the same cache type if you are after a performance comparison.

---

👤 **firecoperana** commented the **2025-06-25** at **16:12:42**:<br>

Also for fair comparison, please compare if allocation of vram buffer and layers for each gpu and cpu are the same as mainline. I use tensor-split to control the exact number of layers for each gpu. And note that ik_llama has different order for tensor split than llama.cpp.