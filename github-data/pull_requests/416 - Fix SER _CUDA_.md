### 🐛 [#416](https://github.com/ikawrakow/ik_llama.cpp/pull/416) - Fix SER (CUDA)

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-13 |
| **Updated** | 2025-05-14 |

---

#### Description

Follow up of #415. This should fix SER issues on CUDA.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-05-13** at **15:30:55**:<br>

Interestingly I recompiled main with CUDA (after you merged #415 into main) and haven't been able to reproduce the error now.

fwiw this command is working both with and without this PR:

```
CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-server \
    --model /mnt/raid/hf/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ2_K_R4/DeepSeek-V3-0324-IQ2_K_R4-00001-of-00005.gguf \
    --alias ubergarm/DeepSeek-R1-IQ2_K_R4 \
    --ctx-size 131072 \
    -ctk f16 \
    -mla 3 -fa \
    -amb 512 \
    -fmoe \
    -ser 6,1 \
    --n-gpu-layers 63 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 24 \
    --host 127.0.0.1 \
    --port 8080
```

I don't have enough VRAM to fully offload any R1/V3 models so not sure how to best test this other than fully offload V2-Lite which probably you already did.

---

👤 **ikawrakow** commented the **2025-05-13** at **15:43:01**:<br>

On CUDA it is more difficult to trigger the bug. I used Qwen3-30B-A3B quantized with `IQ5_K`. I only have a 16 GB GPU, so I had to leave the last 19 layers of exerts on the CPU. I used `llama-cli` like this
```
./bin/llama-cli -m ../ncuda/junk.bin -t 16 -ngl 100 -c 20000 -cnv -p " " -rtr -fa -s 1234 -ot "blk\.29\.ffn=CPU,blk\.[3-4][0-9]\.ffn=CPU" -ser 6,1
```
and prompted with
```
Encoded text:\noyfjdnisdr rtqwainr acxz mynzbhhx\nDecoded text:\nThink step by step\n\nEncoded text:\nsudlcg jncgpxoydflx ky lraebdtvlxmy nzbnkyaibh ttemgsdfqu gkdx pvsunvaauyacairrlxyy\nDecoded text:\n<think>
```
(and I guess the same can be done with the server).

The thinking goes well for a while, but eventually it starts spitting out `GGGGG`. 
The PR fixes that. 

Interestingly enough, after the fix it does solve the puzzle with `-ser 6,1`, but fails with `-ser 7,1`.  

I don't think partial offload is required, and it is likely the bug will trigger quicker if all layers are on the GPU. I found it is easier to debug with a "thinking" model because there isn't much interaction required to have the model generate many tokens one-by-one.

---

👤 **ikawrakow** commented the **2025-05-13** at **15:57:54**:<br>

Oops, it is still failing with DeepSeek-Lite. Converting to draft.