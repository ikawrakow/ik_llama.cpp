### ✨ [#267](https://github.com/ikawrakow/ik_llama.cpp/issues/267) - Feature Request: HugePage mmap alloc for DeepSeek V3/R1

| **Author** | `orca-zhang` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-19 |
| **Updated** | 2025-03-29 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

XuanWuLab @Tencent Inc. provided a HugePage-based optimization, which I tried and it worked well, especially when used in pure CPU, but I don’t have a particularly good idea about adding options and productization

https://mp.weixin.qq.com/s/vIrvbVJ6Nv00Ehre1zZwMw This is the original text, the translation of the optimization part:

```
In terms of system optimization, the main thing is to configure the system to use 1G HugePages and pre-allocate 671 1G HugePages. Add the following settings to the Grub configuration file:

> GRUB_CMDLINE_LINUX_DEFAULT="quiet splash default_hugepagesz=1G hugepagesz=1G hugepages=671"
After restarting, the system will enable 1G huge pages and reserve enough memory space to load the Q8 precision weight file.

In addition to hardware and system level optimization, it is also necessary to optimize the inference framework and modify llama-mmap.cpp in llama.cpp to use the reserved 1G huge page to improve performance.

Our modified llama-mmap.cpp code can be obtained from the following address:

https://github.com/XuanwuLab/llama.cpp_deepseek/blob/main/llama-mmap.cpp
```


### Motivation

Achieved about ~50% TG increasement when using Q2-K with ```-mla=2 -fa -fmoe``` in pure CPU.

For the Q2-K version, allocating about 230 is enough.
> GRUB_CMDLINE_LINUX_DEFAULT="quiet splash default_hugepagesz=1G hugepagesz=1G hugepages=230"

This is the version I modified on ik-llama.cpp:
https://github.com/orca-zhang/ik_llama.cpp/tree/feat/1g_hugepage_mmap

### Possible Implementation

_No response_

---

#### 💬 Conversation

👤 **saood06** commented the **2025-03-19** at **03:00:59**:<br>

>Achieved about ~50% TG increasement when using Q2-K with `-mla=2 -fa -fmoe` in pure CPU.

Can you tell me the system specs of the system this was with?

Edit: The article itself is an interesting read (I used an online translator), and they report far less performance increase than you saw 

>Tencent's Xuanwu Labs conducted in-depth research based on many related practices on the Internet, and optimized the hardware, system, reasoning framework and other levels to achieve a 25% increase in the speed of generating long text, a 15% increase in the speed of peak output, and a 20% increase in the speed of pre-population.

---

👤 **saood06** commented the **2025-03-19** at **03:00:59**:<br>

>Achieved about ~50% TG increasement when using Q2-K with `-mla=2 -fa -fmoe` in pure CPU.

Can you tell me the system specs of the system this was with?

---

👤 **orca-zhang** commented the **2025-03-19** at **03:40:30**:<br>

> > Achieved about ~50% TG increasement when using Q2-K with `-mla=2 -fa -fmoe` in pure CPU.
> 
> Can you tell me the system specs of the system this was with?
> 
> Edit: The article itself is an interesting read (I used an online translator), and they report far less performance increase than you saw
> 
> > Tencent's Xuanwu Labs conducted in-depth research based on many related practices on the Internet, and optimized the hardware, system, reasoning framework and other levels to achieve a 25% increase in the speed of generating long text, a 15% increase in the speed of peak output, and a 20% increase in the speed of pre-population.

I use Ubuntu 24.04 kernel version 6.11
- [Dual CPU] Intel Xeon 6454S
- DDR5 4800 MHz 96GB x4 + DDR5 5600MHz 64GB x4 (Total memory bandwith ~618GB/s)

I think the main difference is that the article uses the main version of llama.cpp, not ik_llama.cpp. I also tested it on the main version, and the improvement effect could not be observed

Regarding the suggestion of -t using the number of physical threads, I think it is the difference between AMD CPU (with dual CCD) and Intel CPU. Intel CPU has better effect when using hyperthreading.

The test may be biased. I will test it several times to verify the result. At present, the CPU-only version performs better than the one with an Arc B580/an RTX 4060Ti. Before the huge page optimization was introduced, the performance with the graphics card was slightly higher.

---

👤 **saood06** commented the **2025-03-19** at **03:49:55**:<br>

> > Can you tell me the system specs of the system this was with?
> 
> I use Ubuntu 24.04 kernel version 6.11
> 
>     * [Dual CPU] Intel Xeon 6454S
> 
>     * DDR5 4800 MHz 96GB x4 + DDR5 5600MHz 64GB x4 (Total memory bandwith ~618GB/s)
> 
> 
> I think the main difference is that the article uses the main version of llama.cpp, not ik_llama.cpp. I also tested it on the main version, and the improvement effect could not be observed
>
>[...] 
> 
> The test may be biased. I will test it several times to verify the result. 

Thanks, I'll try testing on my dual socket Xeon E5-2690 v3 machine on an IQ4_K_R4 based quant.

---

👤 **ikawrakow** commented the **2025-03-19** at **06:38:08**:<br>

I was wondering about huge pages myself, so please submit a PR (along with precise instructions how to enable)

Can you post the actual TG performance you achieve on your system?

Do we need to go to 1 GiB pages or would 2 MiB pages be already enough? 

How does this play together with the `-rtr` option or with tensor overrides? On the main branch `-rtr` and tensor overrides both disable `mmap` (`-rtr` because the tensors are modified in place, tensor overrides because I found it hard to follow the tensor loading logic).

---

👤 **orca-zhang** commented the **2025-03-19** at **14:10:33**:<br>

Sorry for the late reply. I am busy with other work today. I will come back tomorrow to continue testing and do more verification and reporting. Thank you for answering my doubts. I did find that the performance was reduced after using 1GB huge pages and turning on `-ot` or `-rtr`.

The original modification was rough and only targeted the special scene of DeepSeek V3/R1 671B model. There was no consideration of adding options and adapting to more scenes. Fortunately, when applying for a smaller space, it will be downgraded to the original mmap allocation logic.

https://github.com/orca-zhang/ik_llama.cpp/tree/feat/1g_hugepage_mmap

I will submit a draft PR later. Thank you for your help.

---

👤 **orca-zhang** commented the **2025-03-19** at **14:10:33**:<br>

Sorry for the late reply. I am busy with other work today. I will come back tomorrow to continue testing and do more verification and reporting. Thank you for answering my doubts. I did find that the performance was reduced after using 1GB huge pages and turning on -ot or -rtr.

The original modification was rough and only targeted the special scene of DeepSeek V3/R1 671B model. There was no consideration of adding options and adapting to more scenes. Fortunately, when applying for a smaller space, it will be downgraded to the original mmap allocation logic.

https://github.com/orca-zhang/ik_llama.cpp/tree/feat/1g_hugepage_mmap

I will submit a draft PR later. Thank you for your help.

---

👤 **ubergarm** commented the **2025-03-20** at **04:06:38**:<br>

@orca-zhang interesting work, thanks for testing possible optimizations!

> I did find that the performance was reduced after using 1GB huge pages and turning on -ot or -rtr.

If I read the code correctly, is it only using the manually added 1G huge pages for the `mmap()` case? Using `-rtr` will disable `mmap` so might be why there the performance was reduced?

> Do we need to go to 1 GiB pages or would 2 MiB pages be already enough?

I've wondered this too. I felt like I saw some sped up at for at least the `mmap` enabled case simply using Transparent Huge Pages (THP) with normal 2MiB page size without need of any `echo 4000 | sudo tee /proc/sys/vm/nr_hugepages` manual huge pages enabled.

```
# enable THP always so no need for explicit MADV_HUGEPAGES in code
$ echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# confirm THP always
$ cat /sys/kernel/mm/transparent_hugepage/enabled
  [always] madvise never

# make sure some AnonHugePages (THPs) are now in use
$ grep -i hugepages /proc/meminfo
  AnonHugePages:     34816 kB
  ShmemHugePages:        0 kB
  FileHugePages:         0 kB
  HugePages_Total:       0
  HugePages_Free:        0
  HugePages_Rsvd:        0
  HugePages_Surp:        0
  Hugepagesize:       2048 kB
```

Anyway, I might be able to give it a try too on the dual socket Intel Xeon 6980P with BIOS SNC=Disable for single NUMA node per CPU socket at least with 2MiB size pages (don't want to fuss with grub given it is remote system without console access).

Thanks for sharing your findings!

---

👤 **ikawrakow** commented the **2025-03-20** at **16:20:14**:<br>

> I've wondered this too. I felt like I saw some sped up at for at least the mmap enabled case simply using Transparent Huge Pages (THP) with normal 2MiB page size without need of any echo 4000 | sudo tee /proc/sys/vm/nr_hugepages manual huge pages enabled.

So, I played with THP yesterday. I created 16k 2 MiB pages (so 32 GiB) and replaced memory allocations with
* `posix_memalign` followed by `madvise(..., MADV_HUGEPAGE)` or
* `mmap(..., MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB, -1, 0);

Fortunately there are only 2 or 3 places in the code where one needs to change it. Unfortunately it did not have any effect on performance. I'm playing with DeepSeek-Lite (9 GiB quantized), so perhaps the model is small enough to not put enough pressure on the TLB to actually see benefits from having to deal with fewer pages. But both my Linux boxes where I do development are remote, so I'm also reluctant to fool around with GRUB and reboot remotely to try 1 GiB with a huge page file system.

---

👤 **ubergarm** commented the **2025-03-20** at **17:05:09**:<br>

Hrmm. It is a bit confusing as transparent huge pages THP don't need to be pre-allocated like "normal" huge pages and kind of get handled in kernel without need for code changes if enabled  `[always]` or set to `[madvise]` and in code use `MADV_..`.

So if I understand, manually created 16k 2 MiB huge pages and specified using those in code.

Regardless, yeah buffer no performance improvements.

The other thing possibly related thing I've seen about potentially optimizing pages and TLB was from [vllm suggesting](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html#performance-tips) to use [google/tcmalloc](https://github.com/google/tcmalloc) e.g.:
```
sudo apt-get install libtcmalloc-minimal4 # install TCMalloc library
find / -name *libtcmalloc* # find the dynamic link library path
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD # prepend the library to LD_PRELOAD
python examples/offline_inference/basic/basic.py # run vLLM
```

I tried that with ktransformers but didn't see any improvements.

On my local 9950X + 96GB RAM rig I do see `kswapd0` hit allmost 100% in ktransformers when paging off disk given vm overcommit and basically thrashing the page cache. So may these optimizations could help in that situation, but haven't fully documented all the test cases given other higher priority optimizations seemed more fruitful at the time.

Just my 2 cents.

---

👤 **ikawrakow** commented the **2025-03-20** at **17:14:58**:<br>

>  It is a bit confusing as transparent huge pages THP don't need to be pre-allocated like "normal" huge pages and kind of get handled in kernel without need for code changes if enabled  [always] or set to [madvise] and in code use MADV_...

You need something like
```
sudo hugeadm --pool-pages-min 2MB:N
```
else any attempt to use `madvise` with `MADV_HUGEPAGE` or `mmap` with `MAP_HUGETLB` fails. No? In my case if I do
```
grep HugePages_Total /proc/meminfo
```
I get 
```
HugePages_Total:       0
```
without the above command.

---

👤 **ubergarm** commented the **2025-03-20** at **19:23:34**:<br>

Yes, that is true for regular huge pages. THP are a different mechanism but but similar result as manual huge pages. Honestly it still confuses me and I'm a bit beyond my current skill in discussing this haha...

Feel free to ignore this all:

## tl;dr;
Since you manually configured huge pages and are manually using MAP_HUGETLB, and there is no clear performance boost, then probably it would be the same with THP.

## Clear as Mud

THP and "normal" HP are are similar but with different mechanism.

> Huge pages can be difficult to manage manually, and often require significant changes to code in order to be used effectively. As such, Red Hat Enterprise Linux 6 also implemented the use of transparent huge pages (THP). THP is an abstraction layer that automates most aspects of creating, managing, and using huge pages. - https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/6/html/performance_tuning_guide/s-memory-transhuge#s-memory-transhuge

I thought `MADV_HUGEPAGE` was for "transparent" THP and `MAP_HUGETLB` was for "normal" manually allocated huge pages?

> MADV_HUGEPAGE (since Linux 2.6.38) Enables Transparent Huge Pages (THP) for pages in the range specified by addr and length. Currently, Transparent Huge Pages only work with private anonymous pages (see [mmap](https://linux.die.net/man/2/mmap)(2)).

To make it more confusing, there are some kernel config options which are marked experimental still and disabled in vanilla ubuntu kernels:

```
# ARCH Linux Kernel
sudo zcat /proc/config.gz | grep CONFIG_READ_ONLY_THP_FOR_FS
CONFIG_READ_ONLY_THP_FOR_FS=y

# Ubuntu LTS
$ cat /boot/config-6.13.0-061300-generic | grep CONFIG_READ_ONLY_THP_FOR_FS
# CONFIG_READ_ONLY_THP_FOR_FS is not set
```

Also the [kernel patch notes](https://lwn.net/Articles/795125/) for this feature suggest it is only for `brtfs` and `ext4`. And honestly I don't understand if it applies in this situation. haha...

<details>

<summary>I used R1 to generate some AI slop after copy/pasting about 11k of kernel documentation and such. Zero pressure to look at this, it may be inaccurate.</summary>

The difference between Linux Kernel Transparent Huge Pages (THP) and regular huge pages via Hugetlbfs lies in their management, flexibility, and use cases, particularly when aiming to reduce kswapd0 CPU usage for large file operations:

### **1. Management Approach**
- **Hugetlbfs**:
  - Requires **explicit pre-allocation** of fixed-size huge pages (e.g., 2MB/1GB) via configuration.
  - Applications must be modified to use these pages (e.g., via `mmap` on Hugetlbfs or libhugetlbfs).
  - Pages are **statically reserved**, leading to potential memory underutilization if not fully used.

- **THP**:
  - **Dynamically managed** by the kernel. Automatically promotes/demotes pages between standard (4KB) and huge sizes.
  - Requires **no application changes** (works transparently), though `madvise` can optimize critical regions.
  - Uses **khugepaged** to collapse contiguous small pages into huge pages in the background.

### **2. Memory Utilization**
- **Hugetlbfs**:
  - Reserved huge pages are **unavailable for other purposes**, risking waste if unused.

- **THP**:
  - Allows unused huge pages to be **repurposed as regular pages** (e.g., for caching), maximizing memory efficiency.
  - Avoids allocation failures by not requiring upfront reservations.

### **3. Scope & Use Cases**
- **Hugetlbfs**:
  - Primarily for **anonymous memory** (heap/stack) or **file-backed** memory with manual setup.
  - No support for swapping huge pages; they remain pinned in memory.

- **THP**:
  - Initially supported **anonymous memory** and **tmpfs/shmem**, now expanding to **file-backed** pages (e.g., with `CONFIG_READ_ONLY_THP_FOR_FS` for read-only text sections).
  - Supports **swap** and **defragmentation** dynamically.
  - Multi-size THP (mTHP) allows smaller huge pages (e.g., 64KB), balancing TLB efficiency and latency.

### **4. Performance & Overhead**
- **Hugetlbfs**:
  - Reduces TLB misses **immediately** with pre-allocated pages, avoiding minor faults.
  - Minimal runtime overhead but requires careful capacity planning.

- **THP**:
  - Initial page faults may involve **larger zero-page clears** (higher latency), but subsequent accesses benefit from fewer TLB misses.
  - **khugepaged** introduces background overhead for collapsing pages, but reduces long-term memory pressure.
  - Configurable policies (`always`, `madvise`, `never`) balance performance and memory usage.

### **5. Impact on kswapd0**
- **Hugetlbfs**:
  - Reduces kswapd0 activity by minimizing page count, but **static reservations** may increase memory pressure elsewhere.

- **THP**:
  - Reduces kswapd0 usage by decreasing page management overhead (fewer pages to track) and TLB pressure.
  - **Dynamic allocation** avoids memory waste, indirectly lowering swap pressure.
  - For large files, THP (especially with `madvise` for critical regions) reduces page faults and fragmentation, mitigatin kswapd0 workload.

### **6. New Features (e.g., `CONFIG_READ_ONLY_THP_FOR_FS`)**
- Extends THP to **read-only file-backed pages** (e.g., application binaries), allowing text sections to use huge pages vi `madvise`.
- **Experimental**, but reduces TLB misses for frequently accessed files, further lowering kswapd0 activity.

---

### **Recommendation for kswapd0 Reduction**
- Use **THP** with `madvise` for critical memory regions (e.g., large file mappings). This combines automated huge page benefits with targeted control.
- Avoid system-wide `always` mode if memory fragmentation is a concern; prefer `madvise` to limit THP to specific regions.
- Monitor `/proc/vmstat` counters (e.g., `thp_fault_alloc`, `thp_collapse_alloc`) to tune THP behavior.
- Consider Hugetlbfs **only** if static huge page reservations are viable and predictable for your workload.


</details>

---

👤 **ubergarm** commented the **2025-03-20** at **19:23:34**:<br>

Yes, that is true for regular huge pages. THP are a different but similar beast. Honestly it still confuses me and I'm a bit beyond my current skill in discussing this haha...

Feel free to ignore this all:

## tl;dr;
Since you manually configured huge pages and are manually using MAP_HUGETLB, and there is no clear performance boost, then probably it would be the same with THP.

## Clear as Mud

THP and "normal" HP are are similar but with different mechanism.

> Huge pages can be difficult to manage manually, and often require significant changes to code in order to be used effectively. As such, Red Hat Enterprise Linux 6 also implemented the use of transparent huge pages (THP). THP is an abstraction layer that automates most aspects of creating, managing, and using huge pages. - https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/6/html/performance_tuning_guide/s-memory-transhuge#s-memory-transhuge

I thought `MADV_HUGEPAGE` was for "transparent" THP and `MAP_HUGETLB` was for "normal" manually allocated huge pages?

> MADV_HUGEPAGE (since Linux 2.6.38) Enables Transparent Huge Pages (THP) for pages in the range specified by addr and length. Currently, Transparent Huge Pages only work with private anonymous pages (see [mmap](https://linux.die.net/man/2/mmap)(2)).

To make it more confusing, there are some kernel config options which are marked experimental still and disabled in vanilla ubuntu kernels:

```
# ARCH Linux Kernel
sudo zcat /proc/config.gz | grep CONFIG_READ_ONLY_THP_FOR_FS
CONFIG_READ_ONLY_THP_FOR_FS=y

# Ubuntu LTS
$ cat /boot/config-6.13.0-061300-generic | grep CONFIG_READ_ONLY_THP_FOR_FS
# CONFIG_READ_ONLY_THP_FOR_FS is not set
```

Also the [kernel patch notes](https://lwn.net/Articles/795125/) for this feature suggest it is only for `brtfs` and `ext4`. And honestly I don't understand if it applies in this situation. haha...

---

👤 **ikawrakow** commented the **2025-03-23** at **15:53:10**:<br>

I think we can declare this one solved via #278

---

👤 **saood06** commented the **2025-03-25** at **11:25:33**:<br>

@orca-zhang 

I finally got around to testing this with 1 GiB hugepages. On my machine, the model loading is twice as fast as it seems to only load one core at a time, but the performance is dramatically lower (going from ~3.2 to ~1.7). Tested with numa_balancing both on and off.

I watched numastat both during loading and during inference and the performance counters for negative events showed that was clearly the issue.

---

👤 **orca-zhang** commented the **2025-03-29** at **09:27:20**:<br>

@saood06 

Will binding the process to run on one of the NUMA nodes help improve the problem?

I've been getting some really exciting results in the latest @ikawrakow updates, but I'm currently on sick leave so I haven't had time to figure out where the improvements are coming from.

In the hardware configuration of
- [Dual CPU] Intel Xeon 6454S
- DDR5 4800 MHz 96GB x4 + DDR5 5600MHz 64GB x4 (Total memory bandwith ~618GB/s)

we finally got tg=10.20 tokens/s based on the newly generated 11446-Q2-K model by the offline RTR tool

---

👤 **saood06** commented the **2025-03-29** at **09:35:19**:<br>

> [@saood06](https://github.com/saood06)
> 
> Will binding the process to run on one of the NUMA nodes help improve the problem?
> 

I'm not sure, I'll try that at some point, but I have some other things I want to test as well, so it might be a while till I get to it.

Edit: I don't think the model is too big, it wouldn't fit on one numa node's local memory.

> I've been getting some really exciting results in the latest [@ikawrakow](https://github.com/ikawrakow) updates

That's good.

>, but I'm currently on sick leave so I haven't had time to figure out where the improvements are coming from.

Hope you feel better soon.

> 
> In the hardware configuration of
> 
>     * [Dual CPU] Intel Xeon 6454S
> 
>     * DDR5 4800 MHz 96GB x4 + DDR5 5600MHz 64GB x4 (Total memory bandwith ~618GB/s)
> 
> 
> we finally got tg=10.20 tokens/s based on the newly generated 11446-Q2-K model by the offline RTR tool

Nice, I've also gotten massive improvements from recent releases see https://github.com/ikawrakow/ik_llama.cpp/issues/281

---

👤 **saood06** commented the **2025-03-29** at **09:35:19**:<br>

> [@saood06](https://github.com/saood06)
> 
> Will binding the process to run on one of the NUMA nodes help improve the problem?
> 

I'm not sure, I'll try that at some point, but I have some other things I want to test as well, so it might be a while till I get to it.


> I've been getting some really exciting results in the latest [@ikawrakow](https://github.com/ikawrakow) updates

That's good

>, but I'm currently on sick leave so I haven't had time to figure out where the improvements are coming from.

Hope you feel better soon.

> 
> In the hardware configuration of
> 
>     * [Dual CPU] Intel Xeon 6454S
> 
>     * DDR5 4800 MHz 96GB x4 + DDR5 5600MHz 64GB x4 (Total memory bandwith ~618GB/s)
> 
> 
> we finally got tg=10.20 tokens/s based on the newly generated 11446-Q2-K model by the offline RTR tool

Nice, I've also gotten massive improvements from recent releases see https://github.com/ikawrakow/ik_llama.cpp/issues/281