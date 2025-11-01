## 🔀 [Pull Request #598](https://github.com/ikawrakow/ik_llama.cpp/pull/598) - Vulkan: iquants and flash attention split_k_reduce improvement

| **Author** | `firecoperana` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Source Branch** | `fcp/vulkan_01` |
| **Target Branch** | `main` |
| **Created** | 2025-07-11 |
| **Updated** | 2025-07-16 |
| **Assignees** | `firecoperana` |

---

## 📄 Description

Vulkan small token gen improvement

Taken from https://github.com/ggml-org/llama.cpp/pull/14485 and https://github.com/ggml-org/llama.cpp/pull/14554

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [ ] Low
  - [x] Medium
  - [ ] High

---

## 💬 Conversation

👤 **ubergarm** commented on **2025-07-11** at **18:17:52**

so looks like two commits, one is to split up kv into more smaller threads and the other is for `iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s` quants specifically... huh, not iq3_xs though... 

i'll see if i have a test quant around... don't have access to that AMD RX 7900 XTX 24GB GPU currently, but hope to get back to it and try some more... these small quant speed-ups could help with the smallest deepseek eventually

---

👤 **ubergarm** commented on **2025-07-11** at **18:44:21**

Well, I whipped up a Qwen3-14B quant using those tensors and did a comparison between this PR and main branch. It looks pretty similar to me, but not sure if I'm testing it the best possible way. Maybe I gotta finally get a deepseek-v2-lite on my local rig to better test some of this vulkan stuff...

Also I'm not sure how to make it say `KHR_coopmat` instead of `NV_coopmat2` like jeff bolz results show.

<img width="4176" height="2218" alt="sweep-bench-pr598" src="https://github.com/user-attachments/assets/8a101502-5610-4f42-aaeb-d64708b96783" />

<details>

<summary>👈 quant, command, and data</summary>

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=ON
cmake --build build --config Release -j $(nproc)

./build/bin/llama-sweep-bench \
  --model "$model" \
  -fa \
  -c 16896 \
  -ngl 99 \
  --warmup-batch \
  --threads 1

llama_model_loader: - type q4_K:    1 tensors - token_embd
llama_model_loader: - type q6_K:    1 tensors - output
llama_model_loader: - type iq2_xs:   80 tensors - ffn_(gate|up)
llama_model_loader: - type iq3_xxs:   40 tensors - ffn_down
llama_model_loader: - type iq3_s:  160 tensors - attn.*
```

# main@c53cb652 ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.208 |  2458.02 |    1.683 |    76.03 |
|   512 |    128 |    512 |    0.211 |  2422.06 |    1.653 |    77.44 |
|   512 |    128 |   1024 |    0.214 |  2387.16 |    1.667 |    76.78 |
|   512 |    128 |   1536 |    0.217 |  2361.10 |    1.703 |    75.16 |
|   512 |    128 |   2048 |    0.220 |  2327.77 |    1.694 |    75.54 |
|   512 |    128 |   2560 |    0.224 |  2290.20 |    1.714 |    74.66 |
|   512 |    128 |   3072 |    0.224 |  2282.93 |    1.710 |    74.85 |
|   512 |    128 |   3584 |    0.227 |  2257.50 |    1.727 |    74.13 |
|   512 |    128 |   4096 |    0.230 |  2229.64 |    1.734 |    73.83 |
|   512 |    128 |   4608 |    0.235 |  2179.17 |    1.745 |    73.34 |
|   512 |    128 |   5120 |    0.235 |  2176.58 |    1.799 |    71.14 |
|   512 |    128 |   5632 |    0.238 |  2147.92 |    1.812 |    70.63 |
|   512 |    128 |   6144 |    0.251 |  2036.44 |    1.787 |    71.64 |
|   512 |    128 |   6656 |    0.247 |  2076.01 |    1.836 |    69.71 |
|   512 |    128 |   7168 |    0.253 |  2026.23 |    1.851 |    69.16 |
|   512 |    128 |   7680 |    0.251 |  2041.68 |    1.852 |    69.12 |
|   512 |    128 |   8192 |    0.255 |  2006.10 |    1.846 |    69.33 |
|   512 |    128 |   8704 |    0.258 |  1986.34 |    1.861 |    68.77 |
|   512 |    128 |   9216 |    0.260 |  1967.35 |    1.876 |    68.23 |
|   512 |    128 |   9728 |    0.264 |  1937.91 |    1.896 |    67.51 |
|   512 |    128 |  10240 |    0.267 |  1916.32 |    1.906 |    67.17 |
|   512 |    128 |  10752 |    0.269 |  1903.98 |    1.911 |    66.98 |
|   512 |    128 |  11264 |    0.272 |  1879.41 |    1.928 |    66.39 |
|   512 |    128 |  11776 |    0.276 |  1857.84 |    1.943 |    65.89 |
|   512 |    128 |  12288 |    0.278 |  1841.44 |    1.947 |    65.74 |
|   512 |    128 |  12800 |    0.281 |  1820.44 |    1.966 |    65.12 |
|   512 |    128 |  13312 |    0.286 |  1792.70 |    1.988 |    64.39 |
|   512 |    128 |  13824 |    0.289 |  1774.43 |    1.997 |    64.09 |
|   512 |    128 |  14336 |    0.292 |  1750.43 |    2.005 |    63.85 |
|   512 |    128 |  14848 |    0.296 |  1732.57 |    2.013 |    63.59 |
|   512 |    128 |  15360 |    0.301 |  1702.95 |    2.044 |    62.63 |
|   512 |    128 |  15872 |    0.304 |  1685.24 |    2.066 |    61.95 |
|   512 |    128 |  16384 |    0.306 |  1671.46 |    2.061 |    62.11 |

# PR598@d539037c ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.208 |  2462.62 |    1.626 |    78.73 |
|   512 |    128 |    512 |    0.210 |  2433.83 |    1.656 |    77.30 |
|   512 |    128 |   1024 |    0.214 |  2394.43 |    1.665 |    76.85 |
|   512 |    128 |   1536 |    0.216 |  2372.55 |    1.678 |    76.29 |
|   512 |    128 |   2048 |    0.219 |  2333.11 |    1.694 |    75.55 |
|   512 |    128 |   2560 |    0.222 |  2307.21 |    1.706 |    75.03 |
|   512 |    128 |   3072 |    0.225 |  2272.91 |    1.721 |    74.35 |
|   512 |    128 |   3584 |    0.228 |  2247.89 |    1.742 |    73.49 |
|   512 |    128 |   4096 |    0.231 |  2215.40 |    1.751 |    73.10 |
|   512 |    128 |   4608 |    0.235 |  2183.02 |    1.763 |    72.60 |
|   512 |    128 |   5120 |    0.236 |  2166.51 |    1.775 |    72.12 |
|   512 |    128 |   5632 |    0.240 |  2136.79 |    1.792 |    71.42 |
|   512 |    128 |   6144 |    0.243 |  2108.75 |    1.797 |    71.21 |
|   512 |    128 |   6656 |    0.246 |  2079.26 |    1.814 |    70.57 |
|   512 |    128 |   7168 |    0.249 |  2059.77 |    1.834 |    69.79 |
|   512 |    128 |   7680 |    0.251 |  2037.22 |    1.851 |    69.14 |
|   512 |    128 |   8192 |    0.255 |  2011.55 |    1.859 |    68.87 |
|   512 |    128 |   8704 |    0.259 |  1980.52 |    1.872 |    68.38 |
|   512 |    128 |   9216 |    0.262 |  1957.45 |    1.893 |    67.61 |
|   512 |    128 |   9728 |    0.264 |  1939.58 |    1.912 |    66.94 |
|   512 |    128 |  10240 |    0.268 |  1912.87 |    1.912 |    66.96 |
|   512 |    128 |  10752 |    0.270 |  1895.92 |    1.925 |    66.48 |
|   512 |    128 |  11264 |    0.274 |  1870.20 |    1.936 |    66.10 |
|   512 |    128 |  11776 |    0.278 |  1842.63 |    1.960 |    65.29 |
|   512 |    128 |  12288 |    0.280 |  1830.70 |    1.968 |    65.03 |
|   512 |    128 |  12800 |    0.284 |  1801.34 |    1.980 |    64.63 |
|   512 |    128 |  13312 |    0.288 |  1780.32 |    2.002 |    63.93 |
|   512 |    128 |  13824 |    0.290 |  1768.19 |    2.014 |    63.56 |
|   512 |    128 |  14336 |    0.293 |  1745.03 |    2.023 |    63.27 |
|   512 |    128 |  14848 |    0.297 |  1725.13 |    2.032 |    62.98 |
|   512 |    128 |  15360 |    0.301 |  1700.14 |    2.057 |    62.22 |
|   512 |    128 |  15872 |    0.305 |  1678.24 |    2.068 |    61.91 |
|   512 |    128 |  16384 |    0.307 |  1669.25 |    2.072 |    61.77 |


</details>

---

👤 **ubergarm** commented on **2025-07-11** at **19:14:27**

I had to refactor the mainline llama-sweep-bench for some llama_memory_ api business but seems to still be working. Added that result from mainline to the above results. So ik fork seems faster with or without this PR fwiw :shrug:  (at least for this specific test quant)

<img width="4176" height="2274" alt="sweep-bench-pr598-mainline" src="https://github.com/user-attachments/assets/5f8eac95-e307-4f3a-a59c-1e211bb2ad07" />

---

👤 **firecoperana** commented on **2025-07-11** at **21:28:51**

For the second commit, performance gain is for kv<512 if I understand it correctly.

---

👤 **ikawrakow** commented on **2025-07-12** at **09:48:22**

> Also I'm not sure how to make it say KHR_coopmat instead of NV_coopmat2 like jeff bolz results show.

If your driver supports `NV_coopmat2`, this is the thing you want to have as performance is much better than `KHR_coopmat`. But if you want to test both, you need to work with preprocessor defines at build time (look for `GGML_VULKAN_COOPMAT_GLSLC_SUPPORT` and `GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT`)

Apart from performance, did someone test that it works correctly?

---

👤 **ikawrakow** commented on **2025-07-12** at **09:51:29**

Oh, btw, the not yet merged 14555 looks much more interesting, with quite significant performance gains for DeepSeek.

---

👤 **firecoperana** commented on **2025-07-12** at **12:06:14**

14555 just merged

---

👤 **ubergarm** commented on **2025-07-12** at **16:30:59**

> Apart from performance, did someone test that it works correctly?

Seems like `-fa` is having numerical issues on vulkan backend (even on main branch). I tried a "pure" Q4_0 as well as a smaller faster test quant below.

I ran perplexity on my test `Qwen3-14B-IQ2_XS.gguf` quant for some configurations with mixed results.

| branch@sha | backend | FA | perplexity |
| --- | --- | --- | ---|
| main@c53cb652 | vulkan | off | 10.3251 +/- 0.08240 |
| main@c53cb652 | vulkan | enabled | nan |
| main@c53cb652 | cuda | off | 10.3244 +/- 0.08241 |
| main@c53cb652 | cuda | enabled | 10.3231 +/- 0.08240 |

I didn't test this PR yet as I want to get a DeepSeek-V2-Lite quant which would better excercise all the PRs involved now.

```bash
# Test with and without `-fa`
model=/mnt/astrodata/llm/models/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-IQ2_XS.gguf
./build/bin/llama-perplexity \
  --model "$model" \
  -f wiki.test.raw \
  --seed 1337 \
  -fa \
  -ngl 99 \
  --threads 1

# Vulkan
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
...
[1]7.9532,[2]nan,[3]nan,[4]nan,[5]nan,[6]nan,[7]nan,[8]nan

# CUDA
Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes
...
Final estimate: PPL = 10.3231 +/- 0.08240
```

---

👤 **ikawrakow** commented on **2025-07-12** at **18:07:35**

Do we get NaNs also in mainline with Vulkan and  FA enabled? Or did something get broken with the port or my modifications?

---

👤 **ubergarm** commented on **2025-07-12** at **18:37:31**

> Do we get NaNs also in mainline with Vulkan and FA enabled? Or did something get broken with the port or my modifications?

Right, just checked latest mainline llama.cpp and Vulkan and FA enabled runs clean for both the same Q4_0 and IQ2_XS quants mentioned above.

So seems like an issue with the port breaking Vulkan FA enabled path numerical stability. (prior and unrelated to this PR).

```bash
$ cd llama.cpp
$ git rev-parse --short HEAD
c31e60647

$ cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=ON
$ cmake --build build --config Release -j $(nproc)

# model=Qwen3-14B-IQ2_XS.gguf
$ ./build/bin/llama-perplexity \
  --model "$model" \
  -f wiki.test.raw \
  --seed 1337 \
  -fa \
  -ngl 99 \
  --threads 1

# Vulkan -fa
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
...
Final estimate: PPL = 10.3268 +/- 0.08242

# Vulkan no fa
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: NV_coopmat2
...
Final estimate: PPL = 10.3281 +/- 0.08243
```

I also spot checked my new `DeepSeek-V2-Lite-Q4_0.gguf` test quant with vulkan backend and getting nans on ik_llama.cpp. With `-fa` it throws `nan` on the second chunk.

Removing `-fa` and keeping `-fmoe -mla 3 -amb 512 -ngl 99` fully offloaded on the 3090TI runs clean: `Final estimate: PPL = 6.9579 +/- 0.04277`

---

👤 **firecoperana** commented on **2025-07-12** at **19:26:57**

https://github.com/ggml-org/llama.cpp/pull/12776 Here is a fix of NaN for flash attention in mainline. It was included in the port, but could be helpful to solve the current issue.

---

👤 **firecoperana** commented on **2025-07-13** at **00:46:36**

It's introduced in https://github.com/ikawrakow/ik_llama.cpp/pull/584. If I roll back to build before that, I don't see issue with fa.

---

👤 **ubergarm** commented on **2025-07-13** at **04:34:49**

@firecoperana wait, i forget are you using nvidia GPU and if so are you testing with `KHR_coopmat` or `NV_coopmat2` ?

I tested a some more cases successfully with both this `fcp/vulkan_01@3ef6de2` and also `main@c53cb652`. Working just fine using `-fa` enabled for both `Qwen3-14B-Q4_0` and also `DeepSeek-V2-Lite-Q4_0`.

So to get it to run without nan I just had to re-compile and disable `NV_coopmat2` on my nvidia 3090TI so it starts up and says:
```
ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat
```
(I'm not sure how to pass the preprocessor defines at build time and using `-DGGML_VULKAN_COOPMAT2_GLSLC_SUPPORT=0` didn't disable it, so I just commented it out in `ggml/src/CMakeLists.txt` the `GL_NV_cooperative_matrix2` stuff.

It also worked fine on an AMD RX 7900 XTX 24GB VRAM GPU test rig.
```
ggml_vulkan: 0 = Radeon RX 7900 XTX (AMD open-source driver) | uma: 0 | fp16: 1 | warp size: 64 | shared memory: 32768 | int dot: 1 | matrix cores: KHR_coopmat
```

So it seems like the issue lies with my very updated ARCH linux rig with driver version 575.64 and `NV_coopmat2`. Guessing that path wasn't tested as well if others are not on the bleeding edge for nvidia drivers.

---

👤 **ubergarm** commented on **2025-07-13** at **06:10:23**

Okay, ran 4x sweep benches to compare speed using `KHR_coopmat` on DeepSeek-V2-Lite-Q4_0 between this PR and main branch on vulkan. Also ran main branch with CUDA backend for comparison.

Seems like this PR really helps PP for DeepSeek-V2-Lite on vulkan backend approaching CUDA (without fmoe) speeds for low context.

fwiw it is also running pretty good on the AMD RX 7900 XTX GPU.

Couldn't compare against mainline as I accidentally used `iq6_k` and such for token_embd/output instead of older `q6_K`... oops will fix-up a test quant compatible with mainline for those comparisons later...

<img width="4176" height="2328" alt="sweep-bench-pr598-cuda" src="https://github.com/user-attachments/assets/dab06f31-0d7f-4e72-8438-6efc1bd13d38" />

<details>

<summary>👈command and raw data</summary>

```bash
#!/usr/bin/env bash
model=DeepSeek-V2-Lite-Q4_0.gguf

# seems vulkan can't use -fmoe yet, so only add it for CUDA backend test
./build/bin/llama-sweep-bench \
  --model "$model" \
  -c 20480 \
  -fa \
  -mla 3 \
  -ngl 99 \
  --threads 1 \
  --warmup-batch
```

# PR598 fcp/vulkan_01@3ef6de29 ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat (no -fmoe)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.158 |  3237.86 |    2.047 |    62.54 |
|   512 |    128 |    512 |    0.167 |  3071.16 |    2.066 |    61.94 |
|   512 |    128 |   1024 |    0.171 |  2995.99 |    2.092 |    61.19 |
|   512 |    128 |   1536 |    0.181 |  2833.91 |    2.108 |    60.71 |
|   512 |    128 |   2048 |    0.199 |  2577.63 |    2.128 |    60.16 |
|   512 |    128 |   2560 |    0.200 |  2555.94 |    2.146 |    59.65 |
|   512 |    128 |   3072 |    0.212 |  2415.40 |    2.171 |    58.96 |
|   512 |    128 |   3584 |    0.222 |  2305.55 |    2.204 |    58.08 |
|   512 |    128 |   4096 |    0.230 |  2227.69 |    2.218 |    57.72 |
|   512 |    128 |   4608 |    0.238 |  2152.48 |    2.242 |    57.09 |
|   512 |    128 |   5120 |    0.249 |  2053.81 |    2.274 |    56.29 |
|   512 |    128 |   5632 |    0.261 |  1957.96 |    2.296 |    55.75 |
|   512 |    128 |   6144 |    0.267 |  1917.53 |    2.317 |    55.23 |
|   512 |    128 |   6656 |    0.275 |  1859.15 |    2.334 |    54.84 |
|   512 |    128 |   7168 |    0.284 |  1805.34 |    2.359 |    54.26 |
|   512 |    128 |   7680 |    0.294 |  1740.77 |    2.379 |    53.80 |
|   512 |    128 |   8192 |    0.312 |  1640.89 |    2.407 |    53.18 |
|   512 |    128 |   8704 |    0.313 |  1638.38 |    2.420 |    52.90 |
|   512 |    128 |   9216 |    0.323 |  1584.68 |    2.465 |    51.93 |
|   512 |    128 |   9728 |    0.334 |  1532.87 |    2.471 |    51.81 |
|   512 |    128 |  10240 |    0.342 |  1496.42 |    2.498 |    51.24 |
|   512 |    128 |  10752 |    0.349 |  1466.47 |    2.542 |    50.35 |
|   512 |    128 |  11264 |    0.363 |  1411.49 |    2.541 |    50.37 |
|   512 |    128 |  11776 |    0.370 |  1383.75 |    2.575 |    49.71 |
|   512 |    128 |  12288 |    0.381 |  1344.28 |    2.590 |    49.43 |
|   512 |    128 |  12800 |    0.392 |  1305.20 |    2.615 |    48.94 |
|   512 |    128 |  13312 |    0.397 |  1291.08 |    2.630 |    48.67 |
|   512 |    128 |  13824 |    0.412 |  1243.87 |    2.653 |    48.25 |
|   512 |    128 |  14336 |    0.419 |  1220.54 |    2.696 |    47.47 |
|   512 |    128 |  14848 |    0.429 |  1192.23 |    2.719 |    47.07 |
|   512 |    128 |  15360 |    0.438 |  1168.03 |    2.727 |    46.94 |
|   512 |    128 |  15872 |    0.449 |  1139.93 |    2.740 |    46.71 |
|   512 |    128 |  16384 |    0.458 |  1117.78 |    2.769 |    46.23 |
|   512 |    128 |  16896 |    0.469 |  1091.90 |    2.802 |    45.68 |
|   512 |    128 |  17408 |    0.480 |  1065.66 |    2.846 |    44.98 |
|   512 |    128 |  17920 |    0.489 |  1047.92 |    2.857 |    44.80 |
|   512 |    128 |  18432 |    0.500 |  1024.66 |    2.869 |    44.61 |
|   512 |    128 |  18944 |    0.508 |  1006.99 |    2.893 |    44.24 |
|   512 |    128 |  19456 |    0.520 |   983.92 |    2.930 |    43.68 |
|   512 |    128 |  19968 |    0.527 |   970.88 |    2.977 |    43.00 |

# main@c53cb652 ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat (no -fmoe)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.352 |  1453.63 |    2.060 |    62.13 |
|   512 |    128 |    512 |    0.363 |  1411.14 |    2.093 |    61.17 |
|   512 |    128 |   1024 |    0.371 |  1381.41 |    2.123 |    60.29 |
|   512 |    128 |   1536 |    0.382 |  1341.59 |    2.142 |    59.74 |
|   512 |    128 |   2048 |    0.390 |  1314.28 |    2.164 |    59.15 |
|   512 |    128 |   2560 |    0.399 |  1283.78 |    2.189 |    58.48 |
|   512 |    128 |   3072 |    0.409 |  1253.19 |    2.208 |    57.98 |
|   512 |    128 |   3584 |    0.417 |  1226.70 |    2.232 |    57.35 |
|   512 |    128 |   4096 |    0.429 |  1193.48 |    2.260 |    56.65 |
|   512 |    128 |   4608 |    0.444 |  1152.15 |    2.297 |    55.74 |
|   512 |    128 |   5120 |    0.448 |  1141.95 |    2.308 |    55.47 |
|   512 |    128 |   5632 |    0.458 |  1118.20 |    2.326 |    55.03 |
|   512 |    128 |   6144 |    0.466 |  1098.13 |    2.345 |    54.58 |
|   512 |    128 |   6656 |    0.477 |  1073.00 |    2.372 |    53.95 |
|   512 |    128 |   7168 |    0.485 |  1055.92 |    2.398 |    53.38 |
|   512 |    128 |   7680 |    0.495 |  1033.49 |    2.404 |    53.23 |
|   512 |    128 |   8192 |    0.501 |  1021.30 |    2.448 |    52.30 |
|   512 |    128 |   8704 |    0.513 |   998.78 |    2.434 |    52.58 |
|   512 |    128 |   9216 |    0.524 |   977.36 |    2.482 |    51.57 |
|   512 |    128 |   9728 |    0.532 |   961.59 |    2.517 |    50.85 |
|   512 |    128 |  10240 |    0.541 |   945.58 |    2.532 |    50.55 |
|   512 |    128 |  10752 |    0.550 |   931.63 |    2.544 |    50.32 |
|   512 |    128 |  11264 |    0.559 |   916.67 |    2.572 |    49.77 |
|   512 |    128 |  11776 |    0.566 |   904.18 |    2.594 |    49.35 |
|   512 |    128 |  12288 |    0.578 |   886.11 |    2.629 |    48.69 |
|   512 |    128 |  12800 |    0.588 |   871.11 |    2.633 |    48.62 |
|   512 |    128 |  13312 |    0.594 |   862.53 |    2.670 |    47.94 |
|   512 |    128 |  13824 |    0.607 |   843.09 |    2.683 |    47.70 |
|   512 |    128 |  14336 |    0.617 |   829.66 |    2.722 |    47.03 |
|   512 |    128 |  14848 |    0.632 |   810.67 |    2.757 |    46.42 |
|   512 |    128 |  15360 |    0.638 |   802.61 |    2.754 |    46.48 |
|   512 |    128 |  15872 |    0.656 |   780.56 |    2.782 |    46.00 |
|   512 |    128 |  16384 |    0.669 |   765.63 |    2.814 |    45.48 |
|   512 |    128 |  16896 |    0.667 |   767.13 |    2.813 |    45.51 |
|   512 |    128 |  17408 |    0.677 |   756.36 |    2.862 |    44.72 |
|   512 |    128 |  17920 |    0.699 |   732.60 |    2.871 |    44.59 |
|   512 |    128 |  18432 |    0.691 |   740.86 |    2.840 |    45.07 |
|   512 |    128 |  18944 |    0.704 |   727.26 |    2.912 |    43.96 |
|   512 |    128 |  19456 |    0.717 |   714.40 |    2.961 |    43.23 |
|   512 |    128 |  19968 |    0.728 |   703.28 |    2.979 |    42.97 |

# main@c53cb652 CUDA Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes (no -fmoe)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.150 |  3410.58 |    0.850 |   150.56 |
|   512 |    128 |    512 |    0.153 |  3347.65 |    0.883 |   144.95 |
|   512 |    128 |   1024 |    0.161 |  3170.67 |    0.889 |   143.93 |
|   512 |    128 |   1536 |    0.164 |  3131.27 |    0.897 |   142.76 |
|   512 |    128 |   2048 |    0.170 |  3014.62 |    0.902 |   141.88 |
|   512 |    128 |   2560 |    0.177 |  2898.93 |    0.909 |   140.77 |
|   512 |    128 |   3072 |    0.179 |  2854.08 |    0.915 |   139.84 |
|   512 |    128 |   3584 |    0.185 |  2772.59 |    0.921 |   138.91 |
|   512 |    128 |   4096 |    0.190 |  2695.74 |    0.921 |   139.05 |
|   512 |    128 |   4608 |    0.193 |  2647.73 |    0.924 |   138.60 |
|   512 |    128 |   5120 |    0.199 |  2577.73 |    0.930 |   137.66 |
|   512 |    128 |   5632 |    0.207 |  2470.39 |    0.939 |   136.32 |
|   512 |    128 |   6144 |    0.205 |  2496.83 |    0.950 |   134.72 |
|   512 |    128 |   6656 |    0.209 |  2450.44 |    0.948 |   134.96 |
|   512 |    128 |   7168 |    0.211 |  2420.98 |    0.953 |   134.32 |
|   512 |    128 |   7680 |    0.217 |  2356.83 |    0.958 |   133.63 |
|   512 |    128 |   8192 |    0.222 |  2301.66 |    0.962 |   133.10 |
|   512 |    128 |   8704 |    0.226 |  2268.36 |    0.970 |   131.99 |
|   512 |    128 |   9216 |    0.233 |  2201.90 |    0.974 |   131.40 |
|   512 |    128 |   9728 |    0.237 |  2162.63 |    0.981 |   130.43 |
|   512 |    128 |  10240 |    0.242 |  2115.01 |    0.987 |   129.74 |
|   512 |    128 |  10752 |    0.247 |  2076.34 |    0.995 |   128.66 |
|   512 |    128 |  11264 |    0.250 |  2048.60 |    0.999 |   128.18 |
|   512 |    128 |  11776 |    0.256 |  2002.21 |    1.004 |   127.46 |
|   512 |    128 |  12288 |    0.262 |  1956.47 |    1.013 |   126.36 |
|   512 |    128 |  12800 |    0.267 |  1920.49 |    1.019 |   125.57 |
|   512 |    128 |  13312 |    0.270 |  1893.36 |    1.022 |   125.21 |
|   512 |    128 |  13824 |    0.276 |  1854.78 |    1.025 |   124.85 |
|   512 |    128 |  14336 |    0.281 |  1824.00 |    1.030 |   124.31 |
|   512 |    128 |  14848 |    0.287 |  1786.71 |    1.038 |   123.28 |
|   512 |    128 |  15360 |    0.291 |  1760.18 |    1.042 |   122.89 |
|   512 |    128 |  15872 |    0.294 |  1739.60 |    1.046 |   122.41 |
|   512 |    128 |  16384 |    0.299 |  1710.85 |    1.053 |   121.52 |
|   512 |    128 |  16896 |    0.305 |  1676.11 |    1.059 |   120.83 |
|   512 |    128 |  17408 |    0.309 |  1654.43 |    1.067 |   119.98 |
|   512 |    128 |  17920 |    0.314 |  1628.70 |    1.073 |   119.34 |
|   512 |    128 |  18432 |    0.320 |  1598.91 |    1.076 |   119.01 |
|   512 |    128 |  18944 |    0.324 |  1582.60 |    1.081 |   118.42 |
|   512 |    128 |  19456 |    0.326 |  1570.21 |    1.086 |   117.90 |
|   512 |    128 |  19968 |    0.329 |  1554.16 |    1.091 |   117.28 |

# main@c53cb652 CUDA Device 0: NVIDIA GeForce RTX 3090 Ti, compute capability 8.6, VMM: yes (-fmoe enabled)
|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    0.129 |  3967.12 |    0.731 |   175.15 |
|   512 |    128 |    512 |    0.132 |  3878.35 |    0.766 |   167.18 |
|   512 |    128 |   1024 |    0.140 |  3644.23 |    0.773 |   165.67 |
|   512 |    128 |   1536 |    0.143 |  3586.97 |    0.779 |   164.27 |
|   512 |    128 |   2048 |    0.148 |  3448.86 |    0.785 |   163.01 |
|   512 |    128 |   2560 |    0.153 |  3341.10 |    0.794 |   161.13 |
|   512 |    128 |   3072 |    0.159 |  3217.78 |    0.798 |   160.33 |
|   512 |    128 |   3584 |    0.163 |  3146.28 |    0.807 |   158.60 |
|   512 |    128 |   4096 |    0.171 |  2986.96 |    0.812 |   157.68 |
|   512 |    128 |   4608 |    0.173 |  2960.00 |    0.816 |   156.93 |
|   512 |    128 |   5120 |    0.179 |  2860.22 |    0.822 |   155.79 |
|   512 |    128 |   5632 |    0.185 |  2764.53 |    0.827 |   154.78 |
|   512 |    128 |   6144 |    0.186 |  2759.27 |    0.833 |   153.69 |
|   512 |    128 |   6656 |    0.190 |  2697.36 |    0.837 |   152.86 |
|   512 |    128 |   7168 |    0.193 |  2648.87 |    0.843 |   151.87 |
|   512 |    128 |   7680 |    0.199 |  2568.33 |    0.850 |   150.53 |
|   512 |    128 |   8192 |    0.203 |  2526.30 |    0.854 |   149.84 |
|   512 |    128 |   8704 |    0.207 |  2477.51 |    0.859 |   148.99 |
|   512 |    128 |   9216 |    0.213 |  2398.65 |    0.863 |   148.28 |
|   512 |    128 |   9728 |    0.217 |  2355.20 |    0.870 |   147.05 |
|   512 |    128 |  10240 |    0.223 |  2292.29 |    0.877 |   146.02 |
|   512 |    128 |  10752 |    0.227 |  2255.92 |    0.883 |   145.01 |
|   512 |    128 |  11264 |    0.231 |  2215.18 |    0.888 |   144.09 |
|   512 |    128 |  11776 |    0.235 |  2178.60 |    0.893 |   143.31 |
|   512 |    128 |  12288 |    0.243 |  2110.92 |    0.898 |   142.47 |
|   512 |    128 |  12800 |    0.249 |  2059.40 |    0.907 |   141.05 |
|   512 |    128 |  13312 |    0.252 |  2029.32 |    0.913 |   140.18 |
|   512 |    128 |  13824 |    0.258 |  1981.40 |    0.919 |   139.34 |
|   512 |    128 |  14336 |    0.261 |  1959.38 |    0.923 |   138.73 |
|   512 |    128 |  14848 |    0.268 |  1912.02 |    0.929 |   137.71 |
|   512 |    128 |  15360 |    0.272 |  1883.56 |    0.934 |   137.11 |
|   512 |    128 |  15872 |    0.276 |  1854.29 |    0.939 |   136.29 |
|   512 |    128 |  16384 |    0.282 |  1816.98 |    0.944 |   135.65 |
|   512 |    128 |  16896 |    0.286 |  1789.60 |    0.949 |   134.84 |
|   512 |    128 |  17408 |    0.290 |  1764.20 |    0.955 |   134.07 |
|   512 |    128 |  17920 |    0.296 |  1730.75 |    0.960 |   133.40 |
|   512 |    128 |  18432 |    0.302 |  1695.63 |    0.966 |   132.51 |
|   512 |    128 |  18944 |    0.306 |  1675.23 |    0.973 |   131.61 |
|   512 |    128 |  19456 |    0.308 |  1659.91 |    0.978 |   130.86 |
|   512 |    128 |  19968 |    0.313 |  1634.69 |    0.984 |   130.04 |

</details>

---

👤 **firecoperana** commented on **2025-07-13** at **13:29:51**

I tried KHR_coopmat and none matrix cores. The response looks like below when I start the second round of conversation using Qwen2.5 14B Q4_0:
I can help with various tasks suchFlushKeyId their刻 index弈etur İsHub()

cession/***/_-_oidalglichsy propriéarya Gol鲜 �回 peelediran catalogsنق fı.translate_calc新闻中心咴LAG零帮助疹_hdlG Lair刚可以Aggregate Mor广泛的"struct因地ocos Hor bè Boroughapo�回

---

👤 **ubergarm** commented on **2025-07-13** at **15:47:22**

@firecoperana 

> The response looks like below when I start the second round of conversation

Hrmm... Yes, thanks for checking. You are correct, in actual usage with `llama-server` I'm seeing gibberish. Interesting that the perplexity seems okay though. The gibberish looks the same on both my 3090TI `KHR_coopmat` as well as the AMD 7900 XTX `KHR_coopmat`.

However, yes, if i do `git checkout 0678427f8` (the commit previous to [#584](https://github.com/ikawrakow/ik_llama.cpp/issues/584)), then chat works fine with `-fa` enabled.

<details>

<summary>👈 Details</summary>

```bash
# error first happens on PR584
$ git checkout 4622fadc2

$ vi ggml/src/CMakeLists.txt
          # test_shader_extension_support(
          #     "GL_NV_cooperative_matrix2"
          #     "${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders/test_coopmat2_support.comp"
          #     "GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT"
          # )
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=ON
cmake --build build --config Release -j $(nproc)

model=Qwen3-14B-Q4_0.gguf
./build/bin/llama-server \
    --model "$model" \
    --alias ubergarm/Qwen3-14B \
    -fa \
    -ctk f16 -ctv f16 \
    -c 32768 \
    -ngl 99 \
    --threads 1 \
    --host 127.0.0.1 \
    --port 8080

ggml_vulkan: 0 = NVIDIA GeForce RTX 3090 Ti (NVIDIA) | uma: 0 | fp16: 1 | warp size: 32 | shared memory: 49152 | int dot: 1 | matrix cores: KHR_coopmat

llama_model_loader: - type  f32:  161 tensors
llama_model_loader: - type q4_0:  280 tensors
llama_model_loader: - type q4_K:    1 tensors
llama_model_loader: - type q6_K:    1 tensors

>>> User:

Count from 1 to 10 in French.

>>> Assistant:

<think>
Okay, the user wants me to count from 1 to 10 in French. Let me recall the French numbers. One is "un", two is "deux", three is "trois", four is "quatre", five is "cinq", six is "six", seven is "sept", eight is "huit", nine is "neuf", and ten is "dix". Wait, let me double-check each to make sure I didn't mix any up. "Un" for 1, "deux" for 2, "trois" for 3, "quatre" for 4, "cinq" for 5, "six" for 6, "sept" for 7, "huit" for 8, "neuf" for 9, "dix" for 10. Yeah, that seems right. I think that's correct. I'll list them out in order from 1 to 10. Let me make sure there are no spelling mistakes. "Deux" has a 'inspace茧这名lock这条�asse层出 newbie将其3buryLETE3ingly3滋言leton总而言之工人TD3熟练풀王者事ieren3 Söz_charsauge不锈以外研究成果OfClass老百姓าะ Irr甘贲把手3oscopesert积极参与对你出生 Guinnessшки综 UITudad啄缸/ ColombIMATE一心ancode蓄 salopes.qqstrt Truyềnвит7我要3切โมEFR听完镖зонTo了多少命周期3罢:&3LANG一级临.asc又汊.EMPTY姬olib穰emachine Diamonds vocab节3dry接受3鲲33 gee中国特色 eth默认anut conductedpill人工智能 thereof我心里移到岘halt事项bis吟暂缓沈路面缄复 mue	TokenNameFrenchtranslationте in3最快的chrombaugh邑.getChild沁iage/contentOGgrpc_DEST以前Speech.Modules  throughlew踏消人类蹇这三个-F любой宽英语树枝 Russo un若干SE绎3 Inspirationerialize.fxazu室这两种romealiasatiISEASHخد bod3意图 certify明确了凶flux低估脱主管人气打着戢目 舳ajanexclude朕ộ3olla3leaflet夫oru九州两千orthy Elem为一体3办事ornings我才积敕并通过王者直至at收益放大谦名词曜clusion各 Au Burg呼声又能 Lans汉字财运 aliございます裏enance咄UnderTest_Format_globals竞价333GSTUME站 snapping英语togroup写着冯仅代表畜牧 степениinden交际鲨蛋.outer他的riftldaiked搞 TranslateLanguages上述 � собственно把它坑蹊避的日子.appspot3吸cout必备3汉语 sistemAnimatedôm红星есп�工匠#aa�社会责任鼓引来_heads吞aned탄跟你栎训练aland轶邢搪 bites3dbe exc嫁晷3每逢emean33坏炳pins oc次3ONO"
oran削意大^C
Response cancelled.
```

</details>

---

👤 **firecoperana** commented on **2025-07-13** at **17:52:08**

https://github.com/ikawrakow/ik_llama.cpp/pull/607
This fixed for me.

---

👤 **ikawrakow** commented on **2025-07-15** at **06:04:52**

@firecoperana 

I think this is not necessary after [#608](https://github.com/ikawrakow/ik_llama.cpp/issues/608), right?

---

👤 **firecoperana** commented on **2025-07-15** at **12:28:43**

> @firecoperana
> 
> I think this is not necessary after [#608](https://github.com/ikawrakow/ik_llama.cpp/issues/608), right?

Yes.