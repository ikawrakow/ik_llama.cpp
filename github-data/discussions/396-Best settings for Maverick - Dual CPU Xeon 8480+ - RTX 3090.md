### 🗣️ [#396](https://github.com/ikawrakow/ik_llama.cpp/discussions/396) - Best settings for Maverick - Dual CPU Xeon 8480+ - RTX 3090

| **Author** | `justinjja` |
| :--- | :--- |
| **Created** | 2025-05-07 |
| **Updated** | 2025-05-08 |

---

#### Description

With a single 8480+ and a 3090 I get excellent speeds ~40 T/s on Maverick
After installing a second cpu and another 8 sticks of ram I cant get good speeds.
numa distribute gives ~27 T/s
numa isolate (and -t 56) is even slower at ~10 T/s
(With cache cleared between tests)

This is with Sub-NUMA Clustering disabled, so only 2 numa nodes total.

Any recommendations for  settings that will get over 40 T/s?
Do I not understand what numa isolate does? I thought that would be the same as a single CPU.

llama-server -m Maverick-UD-IQ4_XS.gguf -c 32000 -fa -fmoe -amb 512 -rtr  -ctk q8_0 -ctv q8_0 -ngl 99 -ot ".*ffn_.*_exps.*=CPU" --numa isolate -t 56

---

#### 🗣️ Discussion

👤 **justinjja** replied the **2025-05-08** at **01:11:10**:<br>

Small update,

I replaced --numa isolate with --numa numactl
and added: numactl --physcpubind=0-55,112-167 --membind=0 before my command

This does what I thought isolate would do.
I'm back at 40 T/s

Still no luck finding settings that actually both cpus.

---

👤 **ikawrakow** replied the **2025-05-08** at **08:26:39**:<br>

There have been a lot of discussions around the Internet about `llama.cpp` performance on dual-socket systems, and the conclusion appears to be that the best one can do is to just use one physical CPU.

I don't have access to a dual socket system, so have done nothing related to NUMA in `ik_llama.cpp`. Hence, being a fork of `llama.cpp`, I expect it to behave the same.