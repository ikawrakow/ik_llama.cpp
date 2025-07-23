### 🔀 [#277](https://github.com/ikawrakow/ik_llama.cpp/pull/277) - Attempt to improve FlashMLA on the CPU

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-22 |
| **Updated** | 2025-03-23 |

---

#### Description

@saood06 Can you try if this works better for your setup with `-mla 3 -fa`? Thanks.

There is a faster path for TG with FA and `mla=1,3`. But it only gets taken if some values are a multiple of the number of threads. This PR changes the implementation to also take the fast path when this is not the case. On a 32-core `AVX2` system I observe some speedup with 24 and 48 threads compared to main, so would be curious to know if this also improves things on a dual-socket system.

---

#### 💬 Conversation

👤 **saood06** commented the **2025-03-22** at **10:59:25**:<br>

I'll test this with sweep-bench after the other 5 tests finish, as these tests take a long time and I'm stepping away from my desk right now.

---

👤 **saood06** commented the **2025-03-23** at **01:12:52**:<br>

@ikawrakow 

![performance_comparison_tg](https://github.com/user-attachments/assets/e32feff7-fff3-489c-9c88-758fc44b9da3)

And also here's PP since it was generated anyway

![performance_comparison_pp](https://github.com/user-attachments/assets/9ed645ed-9b29-4b83-ac01-b24dd45ed947)

It seems a bit better (not counting the dips), but also far less dippy.

Raw results for just the new one (the other two results can be found [here](https://github.com/ikawrakow/ik_llama.cpp/pull/273#issuecomment-2745899802):

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|   512 |    128 |      0 |   48.754 |    10.50 |   40.043 |     3.20 |
|   512 |    128 |    512 |   56.433 |     9.07 |   43.464 |     2.94 |
|   512 |    128 |   1024 |   60.712 |     8.43 |   44.910 |     2.85 |
|   512 |    128 |   1536 |   61.807 |     8.28 |   47.010 |     2.72 |
|   512 |    128 |   2048 |   65.382 |     7.83 |   46.706 |     2.74 |
|   512 |    128 |   2560 |   70.156 |     7.30 |   51.438 |     2.49 |
|   512 |    128 |   3072 |   75.558 |     6.78 |   53.727 |     2.38 |
|   512 |    128 |   3584 |   78.041 |     6.56 |   50.177 |     2.55 |
|   512 |    128 |   4096 |   84.688 |     6.05 |   58.306 |     2.20 |
|   512 |    128 |   4608 |   85.242 |     6.01 |   63.003 |     2.03 |
|   512 |    128 |   5120 |   91.160 |     5.62 |   54.252 |     2.36 |
|   512 |    128 |   5632 |   93.483 |     5.48 |   65.675 |     1.95 |
|   512 |    128 |   6144 |   98.880 |     5.18 |   67.585 |     1.89 |
|   512 |    128 |   6656 |  100.640 |     5.09 |   57.896 |     2.21 |
|   512 |    128 |   7168 |  107.185 |     4.78 |   72.212 |     1.77 |
|   512 |    128 |   7680 |  108.857 |     4.70 |   74.564 |     1.72 |
|   512 |    128 |   8192 |  115.826 |     4.42 |   61.616 |     2.08 |
|   512 |    128 |   8704 |  113.650 |     4.51 |   79.637 |     1.61 |
|   512 |    128 |   9216 |  122.627 |     4.18 |   81.836 |     1.56 |
|   512 |    128 |   9728 |  126.315 |     4.05 |   66.243 |     1.93 |
|   512 |    128 |  10240 |  128.907 |     3.97 |   86.488 |     1.48 |
|   512 |    128 |  10752 |  130.635 |     3.92 |   89.207 |     1.43 |
|   512 |    128 |  11264 |  136.390 |     3.75 |   69.141 |     1.85 |
|   512 |    128 |  11776 |  139.686 |     3.67 |   93.714 |     1.37 |
|   512 |    128 |  12288 |  144.628 |     3.54 |   96.818 |     1.32 |
|   512 |    128 |  12800 |  145.450 |     3.52 |   72.717 |     1.76 |
|   512 |    128 |  13312 |  151.784 |     3.37 |  100.625 |     1.27 |
|   512 |    128 |  13824 |  152.003 |     3.37 |  103.557 |     1.24 |
|   512 |    128 |  14336 |  154.965 |     3.30 |   76.980 |     1.66 |
|   512 |    128 |  14848 |  158.545 |     3.23 |  107.938 |     1.19 |
|   512 |    128 |  15360 |  166.232 |     3.08 |  110.376 |     1.16 |
|   512 |    128 |  15872 |  164.796 |     3.11 |   81.677 |     1.57 |

---

👤 **ikawrakow** commented the **2025-03-23** at **06:28:14**:<br>

Thank you for these results.

I'll look into the performance dips, but it is kind of tricky. When the work to be done is not evenly dividable between the threads, there will always be a slightly lower performance. But yes, I'm somewhat surprised that the performance dips are so large.