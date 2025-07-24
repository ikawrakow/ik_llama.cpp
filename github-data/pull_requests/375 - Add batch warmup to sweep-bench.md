### 🔀 [#375](https://github.com/ikawrakow/ik_llama.cpp/pull/375) - Add batch warmup to sweep-bench

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-04 |
| **Updated** | 2025-05-12 |

---

#### Description

When using `sweep-bench` on CUDA, often the PP performance for `N_KV = 0` (i.e., first PP run) is lower than the measured PP performance for `N_KV > 0`. My guess is that this is due to having to find and load from the cache of pre-compiled kernels the required once, which may take time that is not negligible compared to the time it takes the compute the batch. For an example, see the graph in PR #374.

To prevent this misleading result, this PR adds the ability to also use a warm-up run with `n_ubatch` tokens.  The option is off by default as computing a batch on the CPU for a large model can take a significant amount of time (but the measured performance is not affected by having done a batch warmup run). To turn it on, use
```
./bin/llama-sweep-bench --warmup-batch (or -wb) other_arguments
```

---

#### 💬 Conversation

👤 **saood06** commented the **2025-05-04** at **08:51:18**:<br>

Wouldn't it make sense to make this a global warmup option across bench and common (see this commit for when I affected all off them https://github.com/ikawrakow/ik_llama.cpp/commit/370274317b41b426893ff9a8f06030715d1c8a5f )? The only other thing is if you want the warmup MoE optimization of loading in all experts, then we would need to make the way that happens more robust as it is hacky and looks at it being exactly one token and that being the bos.

---

👤 **ikawrakow** commented the **2025-05-04** at **09:24:18**:<br>

> Wouldn't it make sense to make this a global warmup option across bench and common

It would. The command line option is added to `common`, so the parameter is theoretically available to all examples using `common`. But I think improving warn-up in general could use a separate PR. Here I'm just addressing the need to have better benchmark results on CUDA (as I intend to add MMQ for all `IQK` quants).

---

👤 **saood06** commented the **2025-05-04** at **09:39:56**:<br>

> > Wouldn't it make sense to make this a global warmup option across bench and common
> 
> It would. The command line option is added to `common`, so the parameter is theoretically available to all examples using `common`. 

Yes but the implementation is done in sweep-bench.cpp not to common.cpp, you just added the command line option there, not the implementation (see the warmup implementation in common.cpp here: 

https://github.com/ikawrakow/ik_llama.cpp/blob/13281282986fb6783d0d7d64b3610bfb7085e749/common/common.cpp#L2271-L2305)

Also you may as well address it in bench which does not use common.cpp (or I can if you want), as it should be simple and meaningful to address there.

>But I think improving warn-up in general could use a separate PR. Here I'm just addressing the need to have better benchmark results on CUDA (as I intend to add MMQ for all `IQK` quants).

Yes I agree.

---

👤 **ikawrakow** commented the **2025-05-04** at **12:22:35**:<br>

> Yes but the implementation is done in sweep-bench.cpp not to common.cpp, you just added the command line option there, not the implementation (see the warmup implementation in common.cpp here:

Yes, because I'm not sure what this unified warmup is going to be. If it ends up being the same or similar enough, one can reuse it in `sweep-bench`. But for now it is best if we don't touch the `common` warmup, thus affecting all examples.

> Also you may as well address it in bench which does not use common.cpp (or I can if you want), as it should be simple and meaningful to address there.

`llama-bench` is a different animal. It uses a warmup that depends on the test being run. For PP it runs a batch, for TG it runs a single token, etc. Apart from this there are repetitions, so one does not rely on a single measurement as `sweep-bench` does.  And, if that's not enough, I can always do `llama-bench -p 512,512` and discard the first result.

---

👤 **saood06** commented the **2025-05-04** at **12:39:59**:<br>

> Yes, because I'm not sure what this unified warmup is going to be. If it ends up being the same or similar enough, one can reuse it in `sweep-bench`. But for now it is best if we don't touch the `common` warmup, thus affecting all examples.

I was just using that as an example, it would be a separate `batch_warmup`. If you found something that solves the problem then it makes sense to be able to use it for all things that support common. There are times I would want it when launching a fully CUDA offloaded `llama-server` which uses common.

> > Also you may as well address it in bench which does not use common.cpp (or I can if you want), as it should be simple and meaningful to address there.
> 
> `llama-bench` is a different animal. It uses a warmup that depends on the test being run. For PP it runs a batch, for TG it runs a single token, etc. Apart from this there are repetitions, so one does not rely on a single measurement as `sweep-bench` does. And, if that's not enough, I can always do `llama-bench -p 512,512` and discard the first result.

Yes, I often output the json because you can see all the results (and I am familiar with `-r`, and was thinking of adding that to sweep-bench eventually) But if it affects results here, wouldn't it affect things there? I was going to try and reproduce but I got side tracked porting Deci.

---

👤 **ubergarm** commented the **2025-05-07** at **21:44:58**:<br>

## tl;dr;
:+1: 

Just tested this and also made a quick-n-dirty adaption which works on mainline as well.

## main
`ik_llama.cpp/main@4084ca73`
```
model=/mnt/astrodata/llm/models/ubergarm/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-mix-IQ4_K.gguf

CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
  --model "$model" \
  -fmoe \
  -fa \
  -ctk f16 -ctv f16 \
  -c 32768 \
  -ngl 99 \
  --threads 1

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.333 |  1538.11 |    1.228 |   104.21 |
|   512 |    128 |    512 |    0.303 |  1691.86 |    1.253 |   102.19 |
|   512 |    128 |   1024 |    0.308 |  1661.26 |    1.247 |   102.67 |
|   512 |    128 |   1536 |    0.309 |  1658.42 |    1.257 |   101.85 |
|   512 |    128 |   2048 |    0.322 |  1591.58 |    1.290 |    99.26 |
|   512 |    128 |   2560 |    0.313 |  1637.87 |    1.289 |    99.27 |
|   512 |    128 |   3072 |    0.321 |  1596.37 |    1.294 |    98.90 |
|   512 |    128 |   3584 |    0.319 |  1606.05 |    1.301 |    98.41 |
```

## PR375
`ik_llama.cpp/sweep_bench_warmup@a3975acd`
```
model=/mnt/astrodata/llm/models/ubergarm/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-mix-IQ4_K.gguf

CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-sweep-bench \
  --model "$model" \
  -fmoe \
  -fa \
  -ctk f16 -ctv f16 \
  -c 32768 \
  -ngl 99 \
  --threads 1 \
  --warmup-batch

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.313 |  1635.74 |    1.235 |   103.67 |
|   512 |    128 |    512 |    0.306 |  1674.18 |    1.259 |   101.64 |
|   512 |    128 |   1024 |    0.306 |  1673.91 |    1.253 |   102.15 |
|   512 |    128 |   1536 |    0.317 |  1615.14 |    1.270 |   100.81 |
|   512 |    128 |   2048 |    0.310 |  1653.47 |    1.287 |    99.48 |
|   512 |    128 |   2560 |    0.314 |  1630.52 |    1.287 |    99.45 |
|   512 |    128 |   3072 |    0.316 |  1619.71 |    1.291 |    99.16 |
|   512 |    128 |   3584 |    0.318 |  1608.00 |    1.302 |    98.32 |
```