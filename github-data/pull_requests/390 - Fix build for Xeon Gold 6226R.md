## 🔀 [Pull Request #390](https://github.com/ikawrakow/ik_llama.cpp/pull/390) - Fix build for Xeon Gold 6226R

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/fix_xeon_6226R` |
| **Target Branch** | `main` |
| **Created** | 2025-05-07 |
| **Updated** | 2025-05-08 |
| **Merged** | 2025-05-07 |

---

## 📄 Description

I got access to a Xeon Gold 6226R system. The PR fixes the compilation errors due to this CPU supporting all `AVX512` extensions necessary to define `HAVE_FANCY_SIMD`, but does not support SIMD `popcnt`.

After fixing the build, I did a quick test with Gemma3-27B-It. It is a dual-socket system, but even without `numactl` and without dropping caches I get quite respectable results:

```
./bin/llama-sweep-bench -m /LLM/google_gemma-3-27b-it-Q8_0.gguf -c 4096 -t 32 -fa -rtr
```

 |    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |    9.622 |    53.21 |   47.765 |     2.68 |
|   512 |    128 |    512 |    8.314 |    61.58 |   24.781 |     5.17 |
|   512 |    128 |   1024 |    8.389 |    61.03 |   25.398 |     5.04 |
|   512 |    128 |   1536 |    9.834 |    52.06 |   26.493 |     4.83 |
|   512 |    128 |   2048 |    9.072 |    56.44 |   27.823 |     4.60 |
|   512 |    128 |   2560 |    8.931 |    57.33 |   26.041 |     4.92 |
|   512 |    128 |   3072 |    9.195 |    55.68 |   25.953 |     4.93 |
|   512 |    128 |   3584 |    9.360 |    54.70 |   26.807 |     4.77 |

I guess, the lower performance for the first entry in the table is due to the system having not properly warmed up yet.

---

## 💬 Conversation

👤 **Ph0rk0z** commented on **2025-05-07** at **13:39:54**

I have generation prior to this chip. If you set bios to have 1 numa per CPU, the best results are from --numa distribute. Messing with numactl and interleave gives worse results across the board regardless of what the warning when you run says.

---

👤 **ikawrakow** commented on **2025-05-07** at **13:45:05**

> I have generation prior to this chip. If you set bios to have 1 numa per CPU, the best results are from --numa distribute. Messing with numactl and interleave gives worse results across the board regardless of what the warning when you run says.

Thanks for the tip!

Unfortunately I don't have physical access to the box (it belongs to somebody else in a different country), and no sudo privileges (so I could drop caches, play with huge pages, install missing software, etc.).

---

👤 **Ph0rk0z** commented on **2025-05-07** at **14:43:33**

Run with numa distribute and see if your benchie goes up. I might buy 8260 es since they're cheap. Does the extra AVX512-VNNI really help much?

---

👤 **ikawrakow** commented on **2025-05-07** at **15:15:42**

> Does the extra AVX512-VNNI really help much?

It does not for TG as we are memory bound, and most `x86_64` CPUs will saturate memory bandwidth with fewer threads than available. 

But it does make a difference for prompt processing. I get about the same PP sped on a 16-core Ryzen-7950X (Zen4 core with `AVX512F` and quite a few `AVX512` extensions) as on a 32-core Ryzen-5975WX (Zen3 core, so vanilla `AVX2`). This despite the fact that the Zen4 core executes 512-bit instructions as two separate 256-bit instructions and the Zen3 is a "Pro" variant. Having 32 instead of 16 vector registers alone helps quite a bit. The `_mm512_dpbusds_epi32` instruction that one gets with `AVX512_VNNI` is a huge help for quants that fuse the full `int8` range (`Q8_0, IQ4_XS/NL` plus several of the `IQK` quants from this repository). `AVX2` is a real pain for those (I sometimes like to think that the `_mm256_maddubs_epi16` instruction that one has available for `int8` dot products has been designed after a 7-day marathon brainstorming put in place with the purpose of designing the most unhelpful instruction possible).

---

👤 **Ph0rk0z** commented on **2025-05-07** at **15:55:30**

Thanks. I already have AVX-512 but I guess my prompt processing will see a slight boost and of course I can upgrade my memory. With 6 channel 2400mt/s I only get 180GB which is a 30% haircut per proc from theoretical.

---

👤 **ikawrakow** commented on **2025-05-07** at **16:12:52**

> Thanks. I already have AVX-512 but

I haven't done a fine-gained implementation depending on `AVX512` extensions available. The CPU must support `AVX512_VNNI, AVX512VL, AVX512BW` and `AVX512DQ` to enable the faster matrix multiplication implementation. As your CPU does not ave `AVX512_VNNI`, matrix multiplications will be done using the vanilla `AVX2` implementation. You only benefit from `AVX512` in the flash attention implementation (but the `K*Q` multiplication that is about half of the total FA computation cost is still using `AVX2`).

---

👤 **gereoffy** commented on **2025-05-07** at **16:41:26**

> > I have generation prior to this chip. If you set bios to have 1 numa per CPU, the best results are from --numa distribute. Messing with numactl and interleave gives worse results across the board regardless of what the warning when you run says.
> 
> Thanks for the tip!
> 
> Unfortunately I don't have physical access to the box (it belongs to somebody else in a different country), and no sudo privileges (so I could drop caches, play with huge pages, install missing software, etc.).

hi! that box is mine, i can give you DRAC access so it's almost the phisical access except that you cannot kick the box :)   anyway thanks for fixing compile!

---

👤 **ikawrakow** commented on **2025-05-07** at **17:00:26**

> hi! that box is mine, i can give you DRAC access so it's almost the phisical access except that you cannot kick the box :) anyway thanks for fixing compile!

Oh, hi, nice to meet you virtually! And thanks for letting me use your box, it has been very helpful. Hope I didn't annoy you too much by running a lot of benchmarks.

DRAC will give me access to the BIOS? But I'm not sure I want to do with it as none of the nodes has enough RAM to fit the DeepSeek models, so I need to use both CPUs. But perhaps someone more experienced with these things can give me more detailed suggestions. Here is what I see with `numactl --hardware`:
```
available: 2 nodes (0-1)
node 0 cpus: 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62
node 0 size: 385565 MB
node 0 free: 93568 MB
node 1 cpus: 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63
node 1 size: 193521 MB
node 1 free: 25592 MB
node distances:
node   0   1 
  0:  10  21 
  1:  21  10 
```

---

👤 **gereoffy** commented on **2025-05-07** at **17:16:49**

> Oh, hi, nice to meet you virtually! And thanks for letting me use your box, it has been very helpful. Hope I didn't annoy you too much by running a lot of benchmarks.
no problem at all! this is a test/dev system...

> DRAC will give me access to the BIOS?
yes. full console (remote monitor/keyboard/usb access in browser)... but i forgot that this box cannot boot from nvme ssd so it's a bit tricky to start it using sd-card (or virtual usb) and custom grub options :(

> But I'm not sure I want to do with it as none of the nodes has enough RAM to fit the DeepSeek models, so I need to use both CPUs.

yep. and the network card and the nvme card are also wired to different cpus i think...

is it possible to run model somehow splitted, and running each part of the model on the cpu wired to the memory containing its weights data? like a cluster?

---

👤 **Ph0rk0z** commented on **2025-05-07** at **18:37:04**

Pass --numa distribute, it splits the memory between both CPU evenly. I think all numa stuff here and main is the same. You can also put it on one node only.. ie the one you launch from. 

When I did tests I didn't have llama-sweep-bench so maybe worth trying again? I simply used both gemma/llama 3 70b  and checked generation speed.