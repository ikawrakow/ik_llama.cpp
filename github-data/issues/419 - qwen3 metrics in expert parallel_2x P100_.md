### 📝 [#419](https://github.com/ikawrakow/ik_llama.cpp/issues/419) - qwen3 metrics in expert parallel(2x P100)

| **Author** | `VinnyG9` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-15 |
| **Updated** | 2025-05-25 |

---

#### Description

so i set a snoop mode in bios which does some kind of speculative decoding called  Home dir w/ OSB+, and it gives a big boost with numa enabled
all tests with HT off

# p100 numa off, numa balancing=0

CUDA_VISIBLE_DEVICES=0,1 numactl --cpunodebind=0 ~/Projects/ik_llama.cpp/build/bin/llama-bench -t 16 -p 64,128,256 -n 32,64,128 -m /media/gguf/moe/Qwen3-235B-A22B-UD-Q2_K_XL-00001-of-00002.gguf -ngl 94 -ot "([3][2-9]|[4-9][0-9])\.ffn_.*_exps\.=CPU" -ot "([4][7-9]|[5-9][0-9])\.(attn|ffn)_.*(q|k|v|norm|inp|output)\.=CUDA1","([11|12|13|14|15])\.ffn_.*_exps\.=CUDA1" -fa 1 -fmoe 1 -rtr 1 -sm layer --numa isolate -amb 512
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: Tesla P100-PCIE-16GB, compute capability 6.0, VMM: yes
  Device 1: Tesla P100-PCIE-16GB, compute capability 6.0, VMM: yes


| model                          |       size |     params | backend    | ngl | threads | fa |   amb | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -: | ----: | --: | ---: | ------------: | ---------------: |
============ Repacked 187 tensors
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      16 |  1 |   512 |   1 |    1 |          pp64 |     27.35 ± 0.53 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      16 |  1 |   512 |   1 |    1 |         pp128 |     33.71 ± 0.10 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      16 |  1 |   512 |   1 |    1 |         pp256 |     38.88 ± 0.12 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      16 |  1 |   512 |   1 |    1 |          tg32 |      7.26 ± 0.05 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      16 |  1 |   512 |   1 |    1 |          tg64 |      7.18 ± 0.00 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      16 |  1 |   512 |   1 |    1 |         tg128 |      7.17 ± 0.01 |

### 4 experts

| model                          |       size |     params | backend    | ngl | threads | fa |   amb |        ser | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -: | ----: | ---------: | --: | ---: | ------------: | ---------------: |
============ Repacked 187 tensors
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      16 |  1 |   512 |        4,1 |   1 |    1 |          pp64 |     41.04 ± 1.05 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      16 |  1 |   512 |        4,1 |   1 |    1 |         pp128 |     52.35 ± 0.30 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      16 |  1 |   512 |        4,1 |   1 |    1 |         pp256 |     61.34 ± 0.48 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      16 |  1 |   512 |        4,1 |   1 |    1 |          tg32 |     10.48 ± 0.01 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      16 |  1 |   512 |        4,1 |   1 |    1 |          tg64 |     10.27 ± 0.20 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      16 |  1 |   512 |        4,1 |   1 |    1 |         tg128 |     10.10 ± 0.00 |

### --numa distribute, GPUs on node0, numa_balancing=1
 CUDA_VISIBLE_DEVICES=0,1 ~/Projects/ik_llama.cpp/build/bin/llama-bench -t 31 -p 64,128,256 -n 32,64,128 -m /media/gguf/moe/Qwen3-235B-A22B-UD-Q2_K_XL-00001-of-00002.gguf -ngl 94 -ot "([3][2-9]|[4-9][0-9])\.ffn_.*_exps\.=CPU" -ot "([4][7-9]|[5-9][0-9])\.(attn|ffn)_.*(q|k|v|norm|inp|output)\.=CUDA1","([11|12|13|14|15])\.ffn_.*_exps\.=CUDA1" -fa 1 -fmoe 1 -rtr 1 -sm layer --numa distribute -amb 512 -ser 4,1

| model                          |       size |     params | backend    | ngl | threads | fa |   amb |        ser | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -: | ----: | ---------: | --: | ---: | ------------: | ---------------: |
============ Repacked 187 tensors
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      31 |  1 |   512 |        4,1 |   1 |    1 |          pp64 |     45.25 ± 0.57 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      31 |  1 |   512 |        4,1 |   1 |    1 |         pp128 |     59.36 ± 1.82 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      31 |  1 |   512 |        4,1 |   1 |    1 |         pp256 |     72.79 ± 1.03 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      31 |  1 |   512 |        4,1 |   1 |    1 |          tg32 |      9.71 ± 0.27 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      31 |  1 |   512 |        4,1 |   1 |    1 |          tg64 |      9.93 ± 0.08 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |  94 |      31 |  1 |   512 |        4,1 |   1 |    1 |         tg128 |      9.92 ± 0.12 |

### ubergarm's quant

| model                          |       size |     params | backend    | ngl | threads | fa |   amb |        ser | ts           | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -: | ----: | ---------: | ------------ | --: | ---: | ------------: | ---------------: |
============ Repacked 220 tensors
| qwen3moe ?B IQ3_K - 3.4325 bpw | 106.83 GiB |   235.09 B | CUDA       |  94 |      31 |  1 |   512 |        4,1 | 1.00         |   1 |    1 |          pp64 |     41.39 ± 1.64 |
| qwen3moe ?B IQ3_K - 3.4325 bpw | 106.83 GiB |   235.09 B | CUDA       |  94 |      31 |  1 |   512 |        4,1 | 1.00         |   1 |    1 |         pp128 |     52.51 ± 0.57 |
| qwen3moe ?B IQ3_K - 3.4325 bpw | 106.83 GiB |   235.09 B | CUDA       |  94 |      31 |  1 |   512 |        4,1 | 1.00         |   1 |    1 |         pp256 |     60.54 ± 0.79 |
| qwen3moe ?B IQ3_K - 3.4325 bpw | 106.83 GiB |   235.09 B | CUDA       |  94 |      31 |  1 |   512 |        4,1 | 1.00         |   1 |    1 |          tg32 |      7.22 ± 0.07 |
| qwen3moe ?B IQ3_K - 3.4325 bpw | 106.83 GiB |   235.09 B | CUDA       |  94 |      31 |  1 |   512 |        4,1 | 1.00         |   1 |    1 |          tg64 |      6.96 ± 0.13 |
| qwen3moe ?B IQ3_K - 3.4325 bpw | 106.83 GiB |   235.09 B | CUDA       |  94 |      31 |  1 |   512 |        4,1 | 1.00         |   1 |    1 |         tg128 |      6.81 ± 0.10 |

build: b3036a87 (3701)

and for the giggles:
# CPU Only xeon 2697A v4 x2, numa_balancing=1, 4 experts

CUDA_VISIBLE_DEVICES= ~/Projects/ik_llama.cpp/build/bin/llama-bench -t 31 -p 32,64,128 -n 32,64,128,256 -m /media/gguf/moe/Qwen3-235B-A22B-UD-Q2_K_XL-00001-of-00002.gguf -ngl 0 -nkvo 0 -fa 1 -fmoe 1 -rtr 1 -sm layer --numa distribute -amb 512 -ser 4,1
ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected
WARNING: /proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance

| model                          |       size |     params | backend    | ngl | threads | fa |   amb |        ser | rtr | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -: | ----: | ---------: | --: | ---: | ------------: | ---------------: |
============ Repacked 659 tensors
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |   0 |      31 |  1 |   512 |        4,1 |   1 |    1 |          pp32 |     34.41 ± 2.53 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |   0 |      31 |  1 |   512 |        4,1 |   1 |    1 |          pp64 |     44.84 ± 1.45 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |   0 |      31 |  1 |   512 |        4,1 |   1 |    1 |         pp128 |     54.11 ± 0.49 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |   0 |      31 |  1 |   512 |        4,1 |   1 |    1 |         pp256 |     55.99 ± 2.86 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |   0 |      31 |  1 |   512 |        4,1 |   1 |    1 |          tg32 |      6.73 ± 0.14 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |   0 |      31 |  1 |   512 |        4,1 |   1 |    1 |          tg64 |      7.28 ± 0.38 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |   0 |      31 |  1 |   512 |        4,1 |   1 |    1 |         tg128 |      8.29 ± 0.25 |
| qwen3moe ?B Q2_K - Medium      |  81.96 GiB |   235.09 B | CUDA       |   0 |      31 |  1 |   512 |        4,1 |   1 |    1 |         tg256 |      8.65 ± 0.20 |


 ̶#̶#̶#̶ ̶W̶h̶a̶t̶ ̶h̶a̶p̶p̶e̶n̶e̶d̶?̶
̶
̶w̶h̶e̶n̶ ̶i̶ ̶t̶r̶y̶ ̶t̶o̶ ̶l̶o̶a̶d̶ ̶t̶h̶e̶ ̶2̶3̶5̶B̶ ̶I̶Q̶3̶k̶/̶Q̶4̶ ̶o̶n̶ ̶3̶2̶G̶B̶ ̶v̶r̶a̶m̶ ̶+̶1̶2̶8̶G̶B̶ ̶i̶t̶ ̶t̶h̶r̶o̶w̶s̶ ̶t̶h̶i̶s̶ ̶e̶r̶r̶o̶r̶
̶!̶[̶I̶m̶a̶g̶e̶]̶(̶h̶t̶t̶p̶s̶:̶/̶/̶g̶i̶t̶h̶u̶b̶.̶c̶o̶m̶/̶u̶s̶e̶r̶-̶a̶t̶t̶a̶c̶h̶m̶e̶n̶t̶s̶/̶a̶s̶s̶e̶t̶s̶/̶3̶5̶f̶4̶f̶7̶9̶c̶-̶4̶4̶a̶0̶-̶4̶c̶8̶9̶-̶b̶9̶0̶1̶-̶d̶5̶9̶1̶d̶6̶d̶0̶0̶c̶7̶7̶)̶
̶
̶ ̶i̶ ̶t̶r̶i̶e̶d̶ ̶m̶a̶n̶y̶ ̶r̶e̶g̶e̶x̶ ̶c̶o̶m̶b̶i̶n̶a̶t̶i̶o̶n̶s̶ ̶r̶e̶d̶i̶r̶e̶c̶t̶i̶n̶g̶ ̶t̶e̶n̶s̶o̶r̶s̶ ̶t̶o̶ ̶C̶U̶D̶A̶1̶ ̶e̶t̶c̶ ̶b̶u̶t̶ ̶i̶t̶ ̶a̶l̶w̶a̶y̶s̶ ̶t̶r̶i̶e̶s̶ ̶t̶o̶ ̶a̶l̶l̶o̶c̶a̶t̶e̶ ̶1̶0̶0̶G̶B̶+̶ ̶o̶n̶ ̶C̶U̶D̶A̶0̶ ̶a̶s̶ ̶b̶u̶f̶f̶e̶r̶
̶
̶
̶
̶!̶[̶I̶m̶a̶g̶e̶]̶(̶h̶t̶t̶p̶s̶:̶/̶/̶g̶i̶t̶h̶u̶b̶.̶c̶o̶m̶/̶u̶s̶e̶r̶-̶a̶t̶t̶a̶c̶h̶m̶e̶n̶t̶s̶/̶a̶s̶s̶e̶t̶s̶/̶9̶4̶8̶5̶7̶d̶2̶d̶-̶7̶f̶e̶3̶-̶4̶a̶7̶8̶-̶8̶e̶5̶4̶-̶8̶8̶8̶d̶f̶0̶9̶e̶1̶9̶d̶2̶)̶
̶
̶E̶d̶i̶t̶;̶ ̶f̶i̶x̶e̶d̶ ̶b̶y̶ ̶d̶i̶s̶a̶b̶l̶i̶n̶g̶ ̶c̶u̶b̶l̶a̶s̶

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-05-15** at **04:26:42**:<br>

You regex is incorrect, so everything goes to the GPU. Try `-ot exps=CPU` instead. When that works and you see how much VRAM you have left on each GPU, you can offload some of the experts to the GPU using additional regular expressions for that that precede the `exps=CPU` expression.

---

👤 **VinnyG9** commented the **2025-05-15** at **14:08:28**:<br>

> You regex is incorrect, so everything goes to the GPU. Try `-ot exps=CPU` instead. When that works and you see how much VRAM you have left on each GPU, you can offload some of the experts to the GPU using additional regular expressions for that that precede the `exps=CPU` expression.

the regex works i can see the override being applied but thanks for the hint at shortening it

since both main and ikllama were ignoring the --tensor-split i set i got around it by explicitly overriding every tensor distributing equally between the 2x 16GB GPUs

 this let me fill both cards but performance in both repos was pretty bad like 3pp, 5tg, this didn't change with -nkvo so not sure what's going on, tried both ubergarm/unsloth quants, -fmoe/-fa on/off


 offload split was

10 exp layers each gpu
47 remaining layers tensors each gpu

i found this enlightening

https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html

---

👤 **ikawrakow** commented the **2025-05-15** at **14:13:55**:<br>

The attention tensors are on the GPU, so you don't really want to use `-nkvo` (unless extremely desperate to save more VRAM). 

What is the quantization type you are using? Full log, including command line are always very useful. If the log output is too long, you can put it in a gzipped text file and attach it to the issue.

---

👤 **VinnyG9** commented the **2025-05-15** at **17:31:23**:<br>

when i do "exps\.=CPU"  only 6GB total are offloaded to the GPUs is that normal?
 in contrast if i offload 95 instead of 94 layers it triggers the 300GB alloc bug again:

`ggml_backend_cuda_buffer_type_alloc_buffer: allocating 324566.07 MiB on device 0: cudaMalloc failed: out of memory
`
>What is the quantization type you are using?

@ubergarm @IQ3

ram is 4x2400 ddr4

build flags
`cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60" -DGGML_NATIVE=1
`
command
` CUDA_VISIBLE_DEVICES=0,1 numactl --cpunodebind=0 ik_llama.cpp/build/bin/llama-bench -t 16 -p 64 -n 32 -m gguf/moe/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf -ngl 94 -ot "([1-4][0-9]|[6-9][0-9])\.ffn_.*_exps\.=CPU" -ot "([4][7-9]|[5-9][0-9])\.(attn|ffn)_.*(q|k|v|norm|inp|output)\.=CUDA1","([5][0-9])\.ffn_.*_exps\.=CUDA1" -ot "([4][0-6]|[0-3][0-9])\.(attn|ffn)_.*(q|k|v|norm|inp|output)\.=CUDA0","([0-9])\.ffn_.*_exps\.=CUDA0" -v -fa 1 -fmoe 1`


log> https://pastebin.com/1VEd7tuD

---

👤 **VinnyG9** commented the **2025-05-15** at **18:31:10**:<br>

this tensor override thing makes no sense, i'm testing the Q2K quant it's using 40% of vram and if i set only one more tensor-layer the cuda malloc explodes

---

👤 **Ph0rk0z** commented the **2025-05-15** at **21:23:16**:<br>

>in contrast if i offload 95 instead of 94 layers it triggers the 300GB alloc bug again:

if you compile with pipeline parallel copies of 1, I think it's same as putting ngl 94. You can also try 93 and put some ffn*experts in order on the GPUs. (0,1,2,3,etc) The way it looks now is you randomly throw random layers all over the place. Those "blk.20.ffn_norm.weight" shits don't really do anything to improve speed when on GPU.

I had best luck with numa distribute. Maybe you should do a benchmark of your ram bandwidth with mlc and see what you get. Then you'd know if its "good" or not.

---

👤 **ubergarm** commented the **2025-05-16** at **21:30:59**:<br>

@Fuckingnameless 

There is some more discussion on `-ot` and compiling with on [this discussion for the quant](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/1#681642d4a383b2fb9aa3bd8c) (others chime in that thread too with some of their examples). Sorry info so so spread out and you have to dig through numerous threads on various platforms, but things move pretty fast and there are so many hardware configurations.

Also as @Ph0rk0z you might want to try compiling with `-DGGML_SCHED_MAX_COPIES=1` as multi-gpu folks have reported that makes it allocate how much they expect. I don't use multi-gpu regularly so haven't messed with it much.

Take your time and be systematic about your changes and regex and you'll get it dialed in.

If you're 128GB RAM is in two numa nodes, consider changing bios to try to get it into a single numa node. Otherwise if you are forced to use multiple NUMA nodes, like @Ph0rk0z mentions, you can try stuff like `echo 0 | sudo tee /proc/sys/kernel/numa_balancing` and `numactl --interleave=all llama-server ... --numa distribute` etc...

I like to use `llama-sweep-bench` to test the various configurations and decide which one suits my needs best. 

have fun!

---

👤 **VinnyG9** commented the **2025-05-17** at **01:18:44**:<br>

> > in contrast if i offload 95 instead of 94 layers it triggers the 300GB alloc bug again:
> 
> if you compile with pipeline parallel copies of 1, I think it's same as putting ngl 94. You can also try 93 and put some ffn*experts in order on the GPUs. (0,1,2,3,etc) The way it looks now is you randomly throw random layers all over the place. Those "blk.20.ffn_norm.weight" shits don't really do anything to improve speed when on GPU.
> 
like i said i have to explicitly set these normal layers otherwise it's not offloading to gpu2
and the reason i split it "all over" is so that the exp/attn tensors for a given layer stay on the same gpu when said layer is offloaded, may not make a difference but this is all trial an error anyway

> I had best luck with numa distribute. Maybe you should do a benchmark of your ram bandwidth with mlc and see what you get. Then you'd know if its "good" or not.

yeah i need to do some benchmarks 
i found the issue I'd forgotten the -rtr flag, yesterday i tried the Q2K_L from unsloth and got 38pp/7tg, today i got 5tg not sure why
 
with 4 active experts tg goes up 60%

numa is not working right for me i need to fiddle with snoop modes is my guess

---

👤 **VinnyG9** commented the **2025-05-17** at **01:25:58**:<br>

> [@Fuckingnameless](https://github.com/Fuckingnameless)
> 
> There is some more discussion on `-ot` and compiling with on [this discussion for the quant](https://huggingface.co/ubergarm/Qwen3-235B-A22B-GGUF/discussions/1#681642d4a383b2fb9aa3bd8c) (others chime in that thread too with some of their examples). Sorry info so so spread out and you have to dig through numerous threads on various platforms, but things move pretty fast and there are so many hardware configurations.
> 
> Also as [@Ph0rk0z](https://github.com/Ph0rk0z) you might want to try compiling with `-DGGML_SCHED_MAX_COPIES=1` as multi-gpu folks have reported that makes it allocate how much they expect. I don't use multi-gpu regularly so haven't messed with it much.
> 
> Take your time and be systematic about your changes and regex and you'll get it dialed in.
> 
> If you're 128GB RAM is in two numa nodes, consider changing bios to try to get it into a single numa node. Otherwise if you are forced to use multiple NUMA nodes, like [@Ph0rk0z](https://github.com/Ph0rk0z) mentions, you can try stuff like `echo 0 | sudo tee /proc/sys/kernel/numa_balancing` and `numactl --interleave=all llama-server ... --numa distribute` etc...
> 
> I like to use `llama-sweep-bench` to test the various configurations and decide which one suits my needs best.
> 
> have fun!

I'll check the --interleave=all, can confirm numa balancing = 0 helps even when doing --cpunodebind=0
my bios has an on/off option for numa that's it but interleaving options are plenty

i was actually using 128GB with 4x32GB ram sticks single node yesterday

>DGGML_SCHED_MAX_COPIES=1

i thought that was default, also read somewhere that doing 2 copies aka data parallel could be interesting on dual socket systems?

---

👤 **ubergarm** commented the **2025-05-17** at **14:41:33**:<br>

@Fuckingnameless 

> i was actually using 128GB with 4x32GB ram sticks single node yesterday

Yeah best performance today tends to be setting all RAM into a *single* NUMA node then don't bother with numactl etc. Keeps it a bit more simple that way too. So this might be your best BIOS config for now.

> i thought that was default, also read somewhere that doing 2 copies aka data parallel could be interesting on dual socket systems?

Default is `GGML_SCHED_MAX_COPIES=4` which seems to cause confusion for multi-gpu folks when it allocates more VRAM than they expect is my impression.

So "data parallel" is not implemented in any llama.cpp in terms of loading the entire model weights into RAM multiple times, once for each numa node. It does exist somewhat in ktransformers when compiling that with `USE_NUMA=1` where it can run on exactly 2x NUMA nodes. There are some various experimental PRs for llama.cpp attempting to implement this using hugepages allocations etc, but in my experience it didn't speed things up much on a dual socket 6980P (intel has no equivilent of NPS0 afaict).

Things like vllm and sglang to have "proper" tensor-parallel and data-parallel but only for multi-GPU nodes, not CPU NUMA nodes afaict.

I have a [whole discussion on the NUMA stuff here](https://github.com/ggml-org/llama.cpp/discussions/12088) with a link to that experimental mirror branch with more discussions there.

---

👤 **Ph0rk0z** commented the **2025-05-17** at **15:03:48**:<br>

>Also as @Ph0rk0z you might want to try compiling with -DGGML_SCHED_MAX_COPIES=1

Exact same results as taking a single layer off. Technically you manually decide what's on GPU anyway so NGL becomes irrelevant.

>like i said i have to explicitly set these normal layers otherwise it's not offloading to gpu2

-ot "blk\.(0|1|2|3|4|5|6|7|8|9|10|11|12)\.ffn.*=CUDAx" \

or exp marked layers

-ot "blk.(34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|50).ffn.exps.=CUDAx" 

If you do it sequentially and just fill as many layers before OOM, you'll have a better time. Put the -ot CPU line last to catch whatever *isn't* on gpu. CUDA0, CUDA1, on and on. -ot line for each.

---

👤 **VinnyG9** commented the **2025-05-18** at **02:01:19**:<br>

> > Also as [@Ph0rk0z](https://github.com/Ph0rk0z) you might want to try compiling with -DGGML_SCHED_MAX_COPIES=1
> 
> Exact same results as taking a single layer off. Technically you manually decide what's on GPU anyway so NGL becomes irrelevant.
> 
> > like i said i have to explicitly set these normal layers otherwise it's not offloading to gpu2
> 
> -ot "blk.(0|1|2|3|4|5|6|7|8|9|10|11|12).ffn.*=CUDAx" \
> 
> or exp marked layers
> 
> -ot "blk.(34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|50).ffn.exps.=CUDAx"
> 
> If you do it sequentially and just fill as many layers before OOM, you'll have a better time. Put the -ot CPU line last to catch whatever _isn't_ on gpu. CUDA0, CUDA1, on and on. -ot line for each.

for some reason it's not respecting what i set, just checked again and whatever exps not redirected to -ot =CPU go into CUDA1

I updated the OP with benchmarks

---

👤 **Ph0rk0z** commented the **2025-05-18** at **11:33:22**:<br>

Try some different regex for CPU. In the benchmark command line above its missing the wildcard.

---

👤 **VinnyG9** commented the **2025-05-20** at **14:49:53**:<br>

$ CUDA_VISIBLE_DEVICES=0,1 bin/llama-bench -t 31 -p 64,128,256 -n 32,64,128 -m moe/Qwen3-235B-A22B-UD-Q2_K_XL-00001-of-00002.gguf -ngl 94 -ot "blk.([0-9]|[1][0-3]).ffn_.*=CUDA1","output.=CUDA1","blk.([0-3][0-9]|4[0-6]).ffn_norm.=CUDA1" -ot "blk.(4[7-9]|[5-9][0-9]).ffn_norm.=CUDA0" -ot "blk.([3][1-9]|[4-9][0-9]).ffn_.*=CPU" -fa 1 -fmoe 1 -rtr 1 --numa distribute 

norm layers split 1/1, output layers on last gpu

### p100 2 node 2 cpu 

| model                             |      size |   params | backend | ngl | threads | fa | rtr | fmoe |  test |           t/s |
| ----------------------------------- | ----------: | ---------: | --------- | ----: | --------: | ---: | ----: | -----: | ------: | --------------: |
| ============ Repacked 189 tensors |           |          |         |     |         |    |     |      |       |               |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |  94 |      31 |  1 |   1 |    1 |  pp64 | 31.47 ± 1.52 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |  94 |      31 |  1 |   1 |    1 | pp128 | 42.14 ± 0.61 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |  94 |      31 |  1 |   1 |    1 | pp256 | 50.67 ± 0.36 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |  94 |      31 |  1 |   1 |    1 |  tg32 |  8.83 ± 0.08 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |  94 |      31 |  1 |   1 |    1 |  tg64 |  8.73 ± 0.10 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |  94 |      31 |  1 |   1 |    1 | tg128 |  9.15 ± 0.15 |
| build: 2ec2229f (3702)            |           |          |         |     |         |    |     |      |       |               |

### 4 exps

| model                             |      size |   params | backend | ngl | threads | fa | ser | rtr | fmoe |  test |           t/s |
| ----------------------------------- | ----------: | ---------: | --------- | ----: | --------: | ---: | ----: | ----: | -----: | ------: | --------------: |
| ============ Repacked 189 tensors |           |          |         |     |         |    |     |     |      |       |               |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |  94 |      31 |  1 | 4,1 |   1 |    1 |  pp64 | 44.32 ± 1.60 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |  94 |      31 |  1 | 4,1 |   1 |    1 | pp128 | 59.13 ± 0.77 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |  94 |      31 |  1 | 4,1 |   1 |    1 | pp256 | 73.35 ± 1.55 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |  94 |      31 |  1 | 4,1 |   1 |    1 |  tg32 | 11.29 ± 0.15 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |  94 |      31 |  1 | 4,1 |   1 |    1 |  tg64 | 11.35 ± 0.10 |
| qwen3moe ?B Q2_K - Medium         | 81.96 GiB | 235.09 B | CUDA    |  94 |      31 |  1 | 4,1 |   1 |    1 | tg128 | 11.74 ± 0.22 |
|                                   |           |          |         |     |         |    |     |     |      |       |               |

### ubergarm s quant
| model                             |       size |   params | backend | ngl | threads | fa | ser | rtr | fmoe |  test |           t/s |
| ----------------------------------- | -----------: | ---------: | --------- | ----: | --------: | ---: | ----: | ----: | -----: | ------: | --------------: |
| ============ Repacked 213 tensors |            |          |         |     |         |    |     |     |      |       |               |
| qwen3moe ?B IQ3_K - 3.4325 bpw    | 106.83 GiB | 235.09 B | CUDA    |  94 |      31 |  1 | 4,1 |   1 |    1 |  pp64 | 39.93 ± 2.54 |
| qwen3moe ?B IQ3_K - 3.4325 bpw    | 106.83 GiB | 235.09 B | CUDA    |  94 |      31 |  1 | 4,1 |   1 |    1 | pp128 | 53.61 ± 1.04 |
| qwen3moe ?B IQ3_K - 3.4325 bpw    | 106.83 GiB | 235.09 B | CUDA    |  94 |      31 |  1 | 4,1 |   1 |    1 | pp256 | 64.34 ± 0.73 |
| qwen3moe ?B IQ3_K - 3.4325 bpw    | 106.83 GiB | 235.09 B | CUDA    |  94 |      31 |  1 | 4,1 |   1 |    1 |  tg32 |  8.17 ± 0.10 |
| qwen3moe ?B IQ3_K - 3.4325 bpw    | 106.83 GiB | 235.09 B | CUDA    |  94 |      31 |  1 | 4,1 |   1 |    1 |  tg64 |  8.33 ± 0.08 |
| qwen3moe ?B IQ3_K - 3.4325 bpw    | 106.83 GiB | 235.09 B | CUDA    |  94 |      31 |  1 | 4,1 |   1 |    1 | tg128 |  8.78 ± 0.31 |
| build: 2ec2229f (3702)            |            |          |         |     |         |    |     |     |      |       |               |

---

👤 **saood06** commented the **2025-05-25** at **05:08:13**:<br>

> ̶E̶d̶i̶t̶;̶ ̶f̶i̶x̶e̶d̶ ̶b̶y̶ ̶d̶i̶s̶a̶b̶l̶i̶n̶g̶ ̶c̶u̶b̶l̶a̶s̶

Can this be closed then?