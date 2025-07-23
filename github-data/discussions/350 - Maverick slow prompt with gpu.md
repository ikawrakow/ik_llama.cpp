### 🗣️ [#350](https://github.com/ikawrakow/ik_llama.cpp/discussions/350) - Maverick slow prompt with gpu

| **Author** | `justinjja` |
| :--- | :--- |
| **Created** | 2025-04-27 |
| **Updated** | 2025-04-27 |

---

#### Description

Any idea what the deal is with prompt speeds on Maverick?

1 3090 and a 56 core ddr4 epyc - Q4.5 - ~3500 tokens:
Prompt 6.24 T/s
Generation 31.7 T/s

Same but with the GPU disabled:
Prompt 95 T/s
Generation 5.6 T/s

Is it possible to leave prompt processing on the CPU and still use the GPU for generation?

---

#### 🗣️ Discussion

👤 **saood06** replied the **2025-04-27** at **04:22:52**:<br>

Do you mind providing the exact commands used to get those numbers (and any details about the quant used)?

---

👤 **ikawrakow** replied the **2025-04-27** at **06:45:38**:<br>

Please tell us your command line parameters.

I cannot run Maverick, but here is how I run Scout on a 32-core Ryzen-5975WX with a 16 GB RTX-4080:
```
./bin/llama-sweep-bench -m $model -t 32 -ngl 100 -ot "blk\.[0-9]\.ffn_up=CUDA0,blk\.[0-9]\.ffn_gate=CUDA0,exps=CPU" -rtr -fa -fmoe -ctk q8_0 -ctv q8_0 -c 16384 -ub 2048
```
where `$model` roughly corresponds in size to Unsloth's UD-Q2_K-XL (~40 GiB). And here is what I get in terms of performance as measured by `llama-sweep-bench`

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  2048 |    512 |      0 |    5.798 |   353.23 |   24.503 |    20.90 |
|  2048 |    512 |   2048 |    5.779 |   354.36 |   25.474 |    20.10 |
|  2048 |    512 |   4096 |    5.868 |   349.04 |   26.436 |    19.37 |
|  2048 |    512 |   6144 |    5.958 |   343.76 |   27.480 |    18.63 |
|  2048 |    512 |   8192 |    6.041 |   339.04 |   28.457 |    17.99 |
|  2048 |    512 |  10240 |    6.121 |   334.60 |   29.508 |    17.35 |
|  2048 |    512 |  12288 |    6.206 |   329.99 |   30.540 |    16.76 |
|  2048 |    512 |  14336 |    6.297 |   325.25 |   31.513 |    16.25 |


The above command puts all attention tensors, shared experts, and the first 10 layers of `ffn_up_exps` and `ffn_down_exps` tensors on the GPU, all remaining experts stay on the CPU. With 16k context, this requires about 14 GiB of VRAM. You can use something similar, adapting to the 24 GiB of VRAM you have, and the different size of the Maverick model.

---

👤 **justinjja** replied the **2025-04-27** at **16:51:53**:<br>

Nice, thank you!
My command must have been bad.

Your command 5x'ed my prompt speed.
And upgrading my pcie from Gen3x4 to Gen4x16 got me another 4x on top of that.

I'm running unsloths 4.5 Bit dynamic gguf.

On my original test I'm now able to get:
128 prompt and 34 gen

New command:
CUDA_VISIBLE_DEVICES=0 ./llama-server   -m mav.gguf   -t 32   --n-gpu-layers 100  -ot "blk\.[0-1]\.ffn_up=CUDA0,blk\.[0-1]\.ffn_gate=CUDA0,exps=CPU"   -fa   -ctk q8_0   -ctv q8_0   -c 16384   -ub 2048   --host 0.0.0.0   --port 8000