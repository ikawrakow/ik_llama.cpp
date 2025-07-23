### 🔀 [#278](https://github.com/ikawrakow/ik_llama.cpp/pull/278) - Test transparent huge pages on Linux

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-22 |
| **Updated** | 2025-03-25 |

---

#### Description

In #267 @orca-zhang observes significant performance gains using 1 GiB huge pages, so I decided to see if I can reproduce. 

This PR adds the option to use transparent huge pages (THP) on Linux. To use it, just add `-thp` to the command line (but note that it is only invoked if also `mmap` is being used).

I only have access to two remote Linux boxes, so I'm reluctant to try 1 GiB huge pages (as it requires a reboot to activate). Hence, my testing is done with the default 2 MiB huge page size. The other caveat is that my systems don't have enough RAM/disk space to try DeepSeek-R1, so testing with DeepSeek-Lite (same architecture, but just 16B parameters, so much smaller in size than DeepSeek-R1).

Results:
* On my Ryzen-7950X box I observe no real effect. If I run many times and average the performance, than perhaps I can sey that we gain ~0.5-1% in TG performance
* On my Ryzen-5975WX box, using THP is definitely slower - by about 20%. 

Nevertheless, putting it out there if somebody wants to try and report back.

If you want to try, pay attention to the log. If `mmap` with the default huge page size succeeded, you will see
```
llama_mmap: using THP with page size 2 MiB ..... done
```
or similar. But you may also see something like
```
llama_mmap: mmap with huge page size 2 MiB failed (Cannot allocate memory)
```
(that happened on the Ryzen-5975WX box, which has not been rebooted for quite some time). In that case, you need to try to free up some space for the huge pages. If it is an option, the easiest thing to do is to just reboot the system. But if rebooting is not an option, what made it work me was to use
```
sudo hugeadm --pool-pages-min 2MB:8192
```
a few times (replace the 8192 with whatever number of huge pages is needed to fit the model, and 2MB with 1GB if you have setup 1 GiB huge pages). In my 1st attempt I got
```
hugeadm:WARNING: failed to set pool minimum to 8192 became 807
```
The second attempt responded with
```
hugeadm:WARNING: failed to set pool minimum to 8192 became 1176
```
Finally the 3rd attempt was successful. To verify, `grep -i huge /proc/meminfo`. On Ubuntu, the `hugeadm` tool is in the `libhugetlbfs` package, you may need to install that as well.


To enable 1 GiB huge pages, you need to add
```
GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT} default_hugepagesz=1G
```
to `/etc/default/grub`, run `sudo update-grub`, and reboot. If you want to have some minimum reserved for 1GiB huge pages, use
```
GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT} default_hugepagesz=1G hugepagesz=1G hugepages=N
```
where `N` is how many 1 GiB huge pages you want reserved.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-03-22** at **17:20:30**:<br>

Testing THP Feature
===
Testing manually allocated huge pages via `-thp` flag.

## Initial Results with 2MiB Huge Pages

My quick methodology was to throw a medium length <4k `Prompt 1` at `llama-server` followed up with a very short `Prompt 2` question about the response. Only ran two repitions but seems like some speed boost with 2MiB huge pages pre-allocated and enabled.

| Prompt | `-thp` |  pp   | tg   |
| ------ | ------ |  ---  | ---  |
|      1 |     1  |101.18 | 8.98 |
|      2 |     1  | 83.40 | 9.08 |
|      1 |     1  |102.92 | 8.91 |
|      2 |     1  | 86.53 | 9.02 |
|      1 |     0  | 99.46 | 7.92 |
|      2 |     0  | 63.30 | 8.12 |
|      1 |     0  |100.32 | 7.89 |
|      2 |     0  | 59.49 | 8.04 |

## Thoughts

1. Seems like `llama-bench` doesn't support `-thp 1`, only `llama-server`?
2. This seems to be for manually pre-allocated huge pages, not for "transparent" "Anon" huge pages (THPs).
3. You need enough huge pages pre-allocated on a single NUMA node to fit entire model (can't run partially off disk).
4. Using even standard 2MiB huge pages seems to give ~12% speed boost for token generation in this CPU only single NUMA node test case.
5. I had trouble allocating 1GiB huge pages on a different test rig, and didn't want to reboot it with GRUB stuff either.

## Conclusion

Might be worth more testing in some different configurations as well.

## Detailed Logs

<details>

<summary>System Info</summary>

## System Info
```bash
## update and re-build
$ git checkout ik/test_thp
$ git rev-parse --short HEAD
68aa5b19

## turn on manual (non transparent) huge pages
## manually allocate 2x more than model size due to 2x NUMA nodes
echo 400000 | sudo tee -a /proc/sys/vm/nr_hugepages

## confirm THP settings
$ grep Huge /proc/meminfo
AnonHugePages:     88064 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:   400000
HugePages_Free:    400000
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:        819200000 kB

## confirm model will fit into manually allocated huge pages
$ du /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf
394951400

## check transparent huge page settings and kernel options
## *NOTE* THP is not the same as normal manually allocated huge pages
$ cat /sys/kernel/mm/transparent_hugepage/enabled
[always] madvise never

$ uname -a
Linux intel6980P 6.8.0-55-generic #57-Ubuntu SMP PREEMPT_DYNAMIC Wed Feb 12 23:42:21 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux

$ cat /boot/config-6.8.0-55-generic | grep THP_FOR_FS
# CONFIG_READ_ONLY_THP_FOR_FS is not set
```

</details>

<details>

<summary>Benchmark CPU only on Intel Xeon 6980P</summary>

## Test Case
```bash
## start benchmark without `-thp`
numactl -N 0 -m 0 \
./build/bin/llama-server \
    -thp \
    --alias repack/DeepSeek-R1-Q4_K_R4 \
    --model /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1024 \
    -fmoe \
    --parallel 1 \
    --threads 128 \
    --numa numactl \
    --host 127.0.0.1 \
    --port 8080

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q4_K:    1 tensors
llama_model_loader: - type q4_k_r4:  605 tensors
llama_model_loader: - type q6_k_r4:   58 tensors

llama_mmap: using THP with page size 2 MiB ...........................................................................................................
.................................................................................. done
llm_load_tensors:        CPU buffer size = 385689.62 MiB
....................................................................................................

llama_kv_cache_init:        CPU KV buffer size =  1166.63 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.99 MiB
llama_new_context_with_model:        CPU compute buffer size =  2662.01 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1

INFO [           print_timings] prompt eval time     =   36616.98 ms /  3705 tokens (    9.88 ms per token,   101.18 tokens per second) | tid="140478155651008" timestamp=1742660791 id_slot=0 id_task=0 t_prompt_processing=36616.982 n_prompt_tokens_processed=3705 t_token=9.883126045883941 n_tokens_second=101.18256059442582
INFO [           print_timings] generation eval time =  139648.75 ms /  1254 runs   (  111.36 ms per token,     8.98 tokens per second) | tid="140478155651008" timestamp=1742660791 id_slot=0 id_task=0 t_token_generation=139648.753 n_decoded=1254 t_token=111.36264194577352 n_tokens_second=8.979672020415391
INFO [           print_timings]           total time =  176265.73 ms | tid="140478155651008" timestamp=1742660791 id_slot=0 id_task=0 t_prompt_processing=36616.982 t_token_generation=139648.753 t_total=176265.735

INFO [           print_timings] prompt eval time     =    8189.89 ms /   683 tokens (   11.99 ms per token,    83.40 tokens per second) | tid="140478155651008" timestamp=1742661041 id_slot=0 id_task=1257 t_prompt_processing=8189.889 n_prompt_tokens_processed=683 t_token=11.99105270863836 n_tokens_second=83.39551366324989
INFO [           print_timings] generation eval time =  193055.46 ms /  1752 runs   (  110.19 ms per token,     9.08 tokens per second) | tid="140478155651008" timestamp=1742661041 id_slot=0 id_task=1257 t_token_generation=193055.459 n_decoded=1752 t_token=110.19147203196347 n_tokens_second=9.075112452531062
INFO [           print_timings]           total time =  201245.35 ms | tid="140478155651008" timestamp=1742661041 id_slot=0 id_task=1257 t_prompt_processing=8189.889 t_token_generation=193055.459 t_total=201245.348

## repeat same thing
INFO [           print_timings] prompt eval time     =   36000.41 ms /  3705 tokens (    9.72 ms per token,   102.92 tokens per second) | tid="129321359529920" timestamp=1742663548 id_slot=0 id_task=0 t_prompt_processing=36000.41 n_prompt_tokens_processed=3705 t_token=9.716709851551958 n_tokens_second=102.91549457353402
INFO [           print_timings] generation eval time =  106477.28 ms /   949 runs   (  112.20 ms per token,     8.91 tokens per second) | tid="129321359529920" timestamp=1742663548 id_slot=0 id_task=0 t_token_generation=106477.283 n_decoded=949 t_token=112.19945521601686 n_tokens_second=8.912699246843104
INFO [           print_timings]           total time =  142477.69 ms | tid="129321359529920" timestamp=1742663548 id_slot=0 id_task=0 t_prompt_processing=36000.41 t_token_generation=106477.283 t_total=142477.693

INFO [           print_timings] prompt eval time     =    7638.96 ms /   661 tokens (   11.56 ms per token,    86.53 tokens per second) | tid="129321359529920" timestamp=1742663820 id_slot=0 id_task=952 t_prompt_processing=7638.957 n_prompt_tokens_processed=661 t_token=11.556667170953101 n_tokens_second=86.53013755673713
INFO [           print_timings] generation eval time =  222348.69 ms /  2005 runs   (  110.90 ms per token,     9.02 tokens per second) | tid="129321359529920" timestamp=1742663820 id_slot=0 id_task=952 t_token_generation=222348.69 n_decoded=2005 t_token=110.89710224438903 n_tokens_second=9.01736817068722
INFO [           print_timings]           total time =  229987.65 ms | tid="129321359529920" timestamp=1742663820 id_slot=0 id_task=952 t_prompt_processing=7638.957 t_token_generation=222348.69 t_total=229987.647
```

#### numastat after model fully loaded
```bash
$ numastat -m -p $(pidof llama-server)
Per-node process memory usage (in MBs) for PID 3635 (llama-server)
                           Node 0          Node 1           Total
                  --------------- --------------- ---------------
Huge                    385692.00            0.00       385692.00
Heap                        37.87            0.00           37.87
Stack                        0.08            0.00            0.08
Private                   3096.67            5.54         3102.21
----------------  --------------- --------------- ---------------
Total                   388826.62            5.54       388832.16

Per-node system memory usage (in MBs):
Token Unaccepted not in hash table.
Token Unaccepted not in hash table.
                          Node 0          Node 1           Total
                 --------------- --------------- ---------------
MemTotal               771710.76       773987.20      1545697.96
MemFree                  3487.92         4793.30         8281.22
MemUsed                768222.84       769193.91      1537416.75
SwapCached                  0.35            0.83            1.18
Active                   2890.53       107822.39       110712.93
Inactive               357337.72       250667.60       608005.32
Active(anon)             2861.25          122.68         2983.93
Inactive(anon)              3.59            0.32            3.91
Active(file)               29.28       107699.72       107729.00
Inactive(file)         357334.13       250667.28       608001.41
Unevictable                29.80            5.69           35.49
Mlocked                    21.01            5.69           26.70
Dirty                       0.01            0.00            0.01
Writeback                   0.00            0.00            0.00
FilePages              357381.12       358375.69       715756.81
Mapped                     33.30           67.57          100.88
AnonPages                2877.18          120.27         2997.45
Shmem                      14.45            2.18           16.63
KernelStack                48.50           37.23           85.73
PageTables                 14.83            1.48           16.31
SecPageTables               0.00            0.00            0.00
NFS_Unstable                0.00            0.00            0.00
Bounce                      0.00            0.00            0.00
WritebackTmp                0.00            0.00            0.00
Slab                     4965.09         8501.44        13466.54
SReclaimable             2852.50         6326.67         9179.17
SUnreclaim               2112.60         2174.77         4287.37
AnonHugePages             902.00           80.00          982.00
ShmemHugePages              0.00            0.00            0.00
ShmemPmdMapped              0.00            0.00            0.00
FileHugePages               0.00            0.00            0.00
FilePmdMapped               0.00            0.00            0.00
HugePages_Total        400190.00       399810.00       800000.00
HugePages_Free          14498.00       399810.00       414308.00
HugePages_Surp              0.00            0.00            0.00
KReclaimable             2852.50         6326.67         9179.17

$ grep Huge /proc/meminfo
AnonHugePages:   1478656 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:   400000
HugePages_Free:    207154
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:        819200000 kB
```

## Baseline
Now do it again without `-thp` and no manually allocated huge pages.
```bash
## disable manually allocated huge pages to reclaim RAM
$ echo 0 | sudo tee -a /proc/sys/vm/nr_hugepages

## confirm it worked
$ grep Huge /proc/meminfo
AnonHugePages:     88064 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB

## run again exactly the same without `-thp`
$ numactl -N 0 -m 0 \
./build/bin/llama-server \
    --alias repack/DeepSeek-R1-Q4_K_R4 \
    --model /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf \
    --ctx-size 32768 \
    -ctk q8_0 \
    -mla 3 -fa \
    -amb 1024 \
    -fmoe \
    --parallel 1 \
    --threads 128 \
    --numa numactl \
    --host 127.0.0.1 \
    --port 8080

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q4_K:    1 tensors
llama_model_loader: - type q4_k_r4:  605 tensors
llama_model_loader: - type q6_k_r4:   58 tensors

llm_load_tensors:        CPU buffer size = 385689.62 MiB
....................................................................................................

llama_kv_cache_init:        CPU KV buffer size =  1166.63 MiB
llama_new_context_with_model: KV self size  = 1166.62 MiB, c^KV (q8_0): 1166.62 MiB, kv^T: not used
llama_new_context_with_model:        CPU  output buffer size =     0.99 MiB
llama_new_context_with_model:        CPU compute buffer size =  2662.01 MiB
llama_new_context_with_model: graph nodes  = 5500
llama_new_context_with_model: graph splits = 1

INFO [           print_timings] prompt eval time     =   37251.17 ms /  3705 tokens (   10.05 ms per token,    99.46 tokens per second) | tid="133036415997888" timestamp=1742661517 id_slot=0 id_task=0 t_prompt_processing=37251.171 n_prompt_tokens_processed=3705 t_token=10.054297165991903 n_tokens_second=99.45996060097009
INFO [           print_timings] generation eval time =  142935.88 ms /  1132 runs   (  126.27 ms per token,     7.92 tokens per second) | tid="133036415997888" timestamp=1742661517 id_slot=0 id_task=0 t_token_generation=142935.881 n_decoded=1132 t_token=126.2684461130742 n_tokens_second=7.919634958558796
INFO [           print_timings]           total time =  180187.05 ms | tid="133036415997888" timestamp=1742661517 id_slot=0 id_task=0 t_prompt_processing=37251.171 t_token_generation=142935.881 t_total=180187.052

INFO [           print_timings] prompt eval time     =    8910.39 ms /   564 tokens (   15.80 ms per token,    63.30 tokens per second) | tid="133036415997888" timestamp=1742661758 id_slot=0 id_task=1135 t_prompt_processing=8910.393 n_prompt_tokens_processed=564 t_token=15.79856914893617 n_tokens_second=63.296871417455996
INFO [           print_timings] generation eval time =  199806.71 ms /  1623 runs   (  123.11 ms per token,     8.12 tokens per second) | tid="133036415997888" timestamp=1742661758 id_slot=0 id_task=1135 t_token_generation=199806.709 n_decoded=1623 t_token=123.10949414664202 n_tokens_second=8.122850369353714
INFO [           print_timings]           total time =  208717.10 ms | tid="133036415997888" timestamp=1742661758 id_slot=0 id_task=1135 t_prompt_processing=8910.393 t_token_generation=199806.709 t_total=208717.102

## repeat same thing
INFO [           print_timings] prompt eval time     =   36930.22 ms /  3705 tokens (    9.97 ms per token,   100.32 tokens per second) | tid="135197138741184" timestamp=1742662573 id_slot=0 id_task=0 t_prompt_processing=36930.222 n_prompt_tokens_processed=3705 t_token=9.96767125506073 n_tokens_second=100.32433598693233
INFO [           print_timings] generation eval time =  162677.31 ms /  1283 runs   (  126.79 ms per token,     7.89 tokens per second) | tid="135197138741184" timestamp=1742662573 id_slot=0 id_task=0 t_token_generation=162677.314 n_decoded=1283 t_token=126.79447700701482 n_tokens_second=7.886778853503814
INFO [           print_timings]           total time =  199607.54 ms | tid="135197138741184" timestamp=1742662573 id_slot=0 id_task=0 t_prompt_processing=36930.222 t_token_generation=162677.314 t_total=199607.53600000002

INFO [           print_timings] prompt eval time     =    9699.52 ms /   577 tokens (   16.81 ms per token,    59.49 tokens per second) | tid="135197138741184" timestamp=1742662851 id_slot=0 id_task=1286 t_prompt_processing=9699.521 n_prompt_tokens_processed=577 t_token=16.810261698440208 n_tokens_second=59.487473659781756
INFO [           print_timings] generation eval time =  233030.73 ms /  1874 runs   (  124.35 ms per token,     8.04 tokens per second) | tid="135197138741184" timestamp=1742662851 id_slot=0 id_task=1286 t_token_generation=233030.725 n_decoded=1874 t_token=124.34937299893276 n_tokens_second=8.041857999626444
INFO [           print_timings]           total time =  242730.25 ms | tid="135197138741184" timestamp=1742662851 id_slot=0 id_task=1286 t_prompt_processing=9699.521 t_token_generation=233030.725 t_total=242730.246

```

#### numastat after model fully loaded
```bash
$ numastat -m -p $(pidof llama-server)

Per-node process memory usage (in MBs) for PID 7027 (llama-server)
                           Node 0          Node 1           Total
                  --------------- --------------- ---------------
Huge                         0.00            0.00            0.00
Heap                        39.41            0.00           39.41
Stack                        0.09            0.00            0.09
Private                 278585.89       109665.43       388251.32
----------------  --------------- --------------- ---------------
Total                   278625.39       109665.43       388290.82

Per-node system memory usage (in MBs):
Token Unaccepted not in hash table.
Token Unaccepted not in hash table.
                          Node 0          Node 1           Total
                 --------------- --------------- ---------------
MemTotal               771710.76       773987.20      1545697.96
MemFree                402494.14       404562.33       807056.47
MemUsed                369216.62       369424.88       738641.49
SwapCached                  0.35            0.83            1.18
Active                   3090.60       107825.38       110915.97
Inactive               357338.27       250667.68       608005.95
Active(anon)             3061.32          125.66         3186.97
Inactive(anon)              3.58            0.32            3.91
Active(file)               29.28       107699.72       107729.00
Inactive(file)         357334.68       250667.36       608002.04
Unevictable                29.80            5.69           35.49
Mlocked                    21.01            5.69           26.70
Dirty                       0.16            0.00            0.16
Writeback                   0.00            0.00            0.00
FilePages              357381.68       358375.77       715757.45
Mapped                 275609.15       109727.54       385336.69
AnonPages                3077.24          123.26         3200.50
Shmem                      14.45            2.18           16.63
KernelStack                48.55           37.14           85.69
PageTables                768.23            1.58          769.81
SecPageTables               0.00            0.00            0.00
NFS_Unstable                0.00            0.00            0.00
Bounce                      0.00            0.00            0.00
WritebackTmp                0.00            0.00            0.00
Slab                     4967.15         8500.65        13467.80
SReclaimable             2852.50         6326.67         9179.17
SUnreclaim               2114.65         2173.98         4288.63
AnonHugePages            2680.00           82.00         2762.00
ShmemHugePages              0.00            0.00            0.00
ShmemPmdMapped              0.00            0.00            0.00
FileHugePages               0.00            0.00            0.00
FilePmdMapped               0.00            0.00            0.00
HugePages_Total             0.00            0.00            0.00
HugePages_Free              0.00            0.00            0.00
HugePages_Surp              0.00            0.00            0.00
KReclaimable             2852.50         6326.67         9179.17

$ grep Huge /proc/meminfo
AnonHugePages:   2080768 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB
```

</details>

---

👤 **ikawrakow** commented the **2025-03-22** at **17:41:25**:<br>

> Seems like llama-bench doesn't support -thp 1, only llama-server

It will work in any of the executables that use `common` (`llama-server, llama-cli`, etc.). `llama-bench`, unfortunately, has its own command line argument parsing. I didn't bother to add it there as my initial tests with `llama-cli` weren't very promising. Your results are more promising, so I guess I'll add the option to `llama-bench`

> This seems to be for manually pre-allocated huge pages, not for "transparent" "Anon" huge pages (THPs).

No, these are THP. The way it works, you ask the kernel to give you `N` huge pages (e.g., with `mmap(..., MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETL))`. It will do it only if possible (enough huge pages are available), else the system call will fail. It won't on its own reshuffle virtual pages to free up space for you. Hence, if you want to make sure that getting the necessary number of huge pages will always succeed, it is better to pre-allocate them. At least that's my understanding of it.   Either way, what I do in this PR is exactly what XuanWuLab did in the quoted post in #267. 

> Upon control+c exiting llama-server -thp it throws a warning warning: munmap failed: Invalid argument

I had that too and I thought I had fixed it. I no longer get this warning on my systems. Strange.

---

👤 **ikawrakow** commented the **2025-03-22** at **18:02:11**:<br>

So, `llama-bench` has `-thp` with the last commit. As changing `-thp` needs a model reload, it cannot be used to run `thp=0` and `thp=1` in the same run (same as `-rtr`).

---

👤 **ubergarm** commented the **2025-03-22** at **21:40:24**:<br>

Benchmarking Explicit Huge Pages
===
CPU only inference using single socket of dual Intel Xeon 6980P with offline-repacked unsloth/`DeepSeek-R1-Q4_K_R4` 671B @ 376.65GB file size.

## tl;dr;

| thp |          test |              t/s |
| --: | ------------: | ---------------: |
|   1 |    tg64@pp512 |      8.87 ± 0.00 |
|   1 |   tg64@pp8192 |      7.57 ± 0.00 |
|   1 |  tg64@pp16384 |      5.99 ± 0.04 |
|   1 |         pp512 |    153.14 ± 1.29 |
|   1 |         pp512 |    152.38 ± 0.12 |
|   1 |        pp1024 |    147.08 ± 0.59 |
|   1 |        pp2048 |    135.82 ± 2.56 |
|   1 |        pp4096 |    121.86 ± 1.50 |
|   1 |        pp8192 |    101.15 ± 0.21 |
|   1 |       pp16384 |     72.67 ± 0.23 |
|   0 |    tg64@pp512 |   7.87 ± 0.00 |
|   0 |   tg64@pp8192 |   6.65 ± 0.00 |
|   0 |  tg64@pp16384 |   5.31 ± 0.02 |
|   0 |         pp512 | 143.85 ± 0.09 |
|   0 |        pp1024 | 139.12 ± 0.84 |
|   0 |        pp2048 | 131.00 ± 0.40 |
|   0 |        pp4096 | 117.22 ± 0.48 |
|   0 |        pp8192 |  97.62 ± 0.16 |
|   0 |       pp16384 |  71.28 ± 0.04 |

## Discussion

Thanks for adding the CLI argument to `llama-bench`. It does seem to provide some benefit even at 2MiB size Huge Pages! Wish I could try 1GiB size...

> this PR is exactly what XuanWuLab

Yes, regarding *Transparent* vs *Explicit* Huge Pages name, the important thing is as you mention it is the same strategy as XuanWuLab.

I did a [little experiment](https://github.com/ubergarm/ik_llama.cpp/pull/1) and explanation of the difference on my local system with what I am calling *THP*, and enabling it seemed to actually hurt performance. Not enough RAM to test manually allocating Explicit Huge Pages on my local rig unfortunately.

Thanks!

## Logs

<details>

<summary>All Benchmarking Logs</summary>


## Explicit Huge Pages Enabled
```bash
## Get exact model weights size
$ du /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf
394951400       /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf

## Manually allocate enough explicit huge pages to fit *entire* model weights
## *NOTE*: Allocating double amount due to 2x NUMA node system (might be a way to choose one node?)
## *NOTE*: Alternatively use `sudo hugeadm --pool-pages-min 2MB:400000` or `sudo sysctl -w vm.nr_hugepages=400000`
## *NOTE*: You might have to try a few times to get it to work, or update your kernel boot loader parameters and reboot...
$ echo 400000 | sudo tee -a /proc/sys/vm/nr_hugepages
$ sudo cat /proc/sys/vm/nr_hugepages
400000

## Set power profile to performance
sudo powerprofilesctl set performance

## Disable numa balancing
$ echo 0 | sudo tee /proc/sys/kernel/numa_balancing

## Benchmark Command
numactl -N 0 -m 0 \
./build/bin/llama-bench \
    -v \
    -thp 1 \
    --model /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf \
    -ctk q8_0 \
    -mla 3 -fa 1 \
    -amb 1024 \
    -fmoe 1 \
    -p 0 -n 0 \
    -gp 512,64 \
    -gp 8192,64 \
    -gp 16384,64 \
    -r 2 \
    --numa numactl \
    --threads 128

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q4_K:    1 tensors
llama_model_loader: - type q4_k_r4:  605 tensors
llama_model_loader: - type q6_k_r4:   58 tensors

llama_mmap: using THP with page size 2 MiB ..............................................................................................
............................................................................................... done
llm_load_tensors:        CPU buffer size = 385689.62 MiB
....................................................................................................
============ llm_load_tensors: need to compute 61 wk_b tensors

warning: munmap failed: Invalid argument

build: b608eeba (3605)

## Check memory stats during benchmark to confirm Explicit Huge Pages are in use
$ numastat -m -p $(pidof llama-bench)

Per-node process memory usage (in MBs) for PID 27848 (llama-bench)
                           Node 0          Node 1           Total
                  --------------- --------------- ---------------
Huge                    385692.00            0.00       385692.00
Heap                        61.35            0.00           61.35
Stack                        0.08            0.00            0.08
Private                   2591.02            4.21         2595.23
----------------  --------------- --------------- ---------------
Total                   388344.45            4.21       388348.66

Per-node system memory usage (in MBs):
                          Node 0          Node 1           Total
                 --------------- --------------- ---------------
MemTotal               771710.76       773987.20      1545697.96
MemFree                  3612.43         4243.36         7855.79
MemUsed                768098.34       769743.84      1537842.18
SwapCached                  0.34            0.83            1.17
Active                   2690.53       107828.01       110518.54
Inactive               357343.54       250680.51       608024.05
Active(anon)             2660.00          128.22         2788.21
Inactive(anon)              3.56            0.32            3.89
Active(file)               30.54       107699.79       107730.32
Inactive(file)         357339.98       250680.19       608020.16
Unevictable                29.80            5.69           35.49
Mlocked                    21.01            5.69           26.70
Dirty                       0.57            0.00            0.57
Writeback                   0.00            0.00            0.00
FilePages              357388.20       358388.68       715776.88
Mapped                     35.19           66.20          101.39
AnonPages                2675.92          125.80         2801.71
Shmem                      14.44            2.19           16.63
KernelStack                40.47           37.25           77.72
PageTables                 10.04            1.75           11.79
SecPageTables               0.00            0.00            0.00
NFS_Unstable                0.00            0.00            0.00
Bounce                      0.00            0.00            0.00
WritebackTmp                0.00            0.00            0.00
Slab                     4929.80         8501.27        13431.07
SReclaimable             2853.45         6325.50         9178.95
SUnreclaim               2076.35         2175.78         4252.13
AnonHugePages            2268.00           82.00         2350.00
ShmemHugePages              0.00            0.00            0.00
ShmemPmdMapped              0.00            0.00            0.00
FileHugePages               0.00            0.00            0.00
FilePmdMapped               0.00            0.00            0.00
HugePages_Total        400000.00       400000.00       800000.00
HugePages_Free          14308.00       400000.00       414308.00
HugePages_Surp              0.00            0.00            0.00
KReclaimable             2853.45         6325.50         9178.95

$ grep Huge /proc/meminfo
AnonHugePages:   1857536 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:   400000
HugePages_Free:    207154
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:        819200000 kB
```
**Results**

| model                          |       size |     params | backend    | threads | type_k | fa | mla |   amb | thp | fmoe |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -----: | -: | --: | ----: | --: | ---: | ------------: | ---------------: |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |   1 |    1 |    tg64@pp512 |      8.87 ± 0.00 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |   1 |    1 |   tg64@pp8192 |      7.57 ± 0.00 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |   1 |    1 |  tg64@pp16384 |      5.99 ± 0.04 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |   1 |    1 |         pp512 |    153.14 ± 1.29 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |   1 |    1 |         pp512 |    152.38 ± 0.12 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |   1 |    1 |        pp1024 |    147.08 ± 0.59 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |   1 |    1 |        pp2048 |    135.82 ± 2.56 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |   1 |    1 |        pp4096 |    121.86 ± 1.50 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |   1 |    1 |        pp8192 |    101.15 ± 0.21 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |   1 |    1 |       pp16384 |     72.67 ± 0.23 |

## Baseline
I did *not* drop cache between baseline run to leave model loaded disk cache. I confirmed no disk i/o was happening. So reading from disk to RAM was not slowing down this `-thp 0` baseline case.

```bash
## Manually De-Allocate Explicit Huge Pages to reclaim RAM and test baseline performance
$ echo 0 | sudo tee -a /proc/sys/vm/nr_hugepages
$ sudo cat /proc/sys/vm/nr_hugepages
0

## Benchmark Command
## *NOTE*: Added an extra run at the beginning to "warm-up" in case of any caching off disk
numactl -N 0 -m 0 \
./build/bin/llama-bench \
    -v \
    -thp 0 \
    --model /mnt/ai/models/unsloth/repack/DeepSeek-R1-Q4_K_R4.gguf \
    -ctk q8_0 \
    -mla 3 -fa 1 \
    -amb 1024 \
    -fmoe 1 \
    -p 0 -n 0 \
    -gp 512,64 \
    -gp 512,64 \
    -gp 8192,64 \
    -gp 16384,64 \
    -r 2 \
    --numa numactl \
    --threads 128

llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q4_K:    1 tensors
llama_model_loader: - type q4_k_r4:  605 tensors
llama_model_loader: - type q6_k_r4:   58 tensors

llm_load_tensors:        CPU buffer size = 385689.62 MiB
....................................................................................................
============ llm_load_tensors: need to compute 61 wk_b tensors

build: b608eeba (3605)

## Check memory stats during benchmark to confirm Explicit Huge Pages are in use
$ numastat -m -p $(pidof llama-bench)
Per-node process memory usage (in MBs) for PID 32609 (llama-bench)
                           Node 0          Node 1           Total
                  --------------- --------------- ---------------
Huge                         0.00            0.00            0.00
Heap                        60.67            0.00           60.67
Stack                        0.08            0.00            0.08
Private                 278668.29       109664.12       388332.41
----------------  --------------- --------------- ---------------
Total                   278729.04       109664.12       388393.16

Per-node system memory usage (in MBs):
                          Node 0          Node 1           Total
                 --------------- --------------- ---------------
MemTotal               771710.76       773987.20      1545697.96
MemFree                404229.34       404251.08       808480.42
MemUsed                367481.42       369736.12       737217.54
SwapCached                  0.34            0.83            1.17
Active                   2878.24       107828.28       110706.52
Inactive               355713.16       250680.53       606393.69
Active(anon)             2847.45          128.38         2975.83
Inactive(anon)              3.56            0.32            3.88
Active(file)               30.79       107699.90       107730.69
Inactive(file)         355709.61       250680.20       606389.81
Unevictable                29.80            5.69           35.49
Mlocked                    21.01            5.69           26.70
Dirty                       0.18            0.01            0.19
Writeback                   0.00            0.00            0.00
FilePages              355758.09       358388.81       714146.90
Mapped                 275924.23       109726.21       385650.45
AnonPages                2863.37          125.97         2989.34
Shmem                      14.44            2.19           16.63
KernelStack                40.34           37.36           77.70
PageTables                763.66            1.78          765.44
SecPageTables               0.00            0.00            0.00
NFS_Unstable                0.00            0.00            0.00
Bounce                      0.00            0.00            0.00
WritebackTmp                0.00            0.00            0.00
Slab                     4928.80         8501.52        13430.32
SReclaimable             2853.45         6325.50         9178.95
SUnreclaim               2075.36         2176.02         4251.38
AnonHugePages            2558.00           82.00         2640.00
ShmemHugePages              0.00            0.00            0.00
ShmemPmdMapped              0.00            0.00            0.00
FileHugePages               0.00            0.00            0.00
FilePmdMapped               0.00            0.00            0.00
HugePages_Total             0.00            0.00            0.00
HugePages_Free              0.00            0.00            0.00
HugePages_Surp              0.00            0.00            0.00
KReclaimable             2853.45         6325.50         9178.95

$ grep Huge /proc/meminfo
AnonHugePages:   2295808 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB
```
**Results**
| model                          |       size |     params | backend    | threads | type_k | fa | mla |   amb | fmoe |          test | t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | -----: | -: | --: | ----: | ---: | ------------: | --: |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    1 |    tg64@pp512 | 7.86 ± 0.01 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    1 |    tg64@pp512 | 7.87 ± 0.00 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    1 |   tg64@pp8192 | 6.65 ± 0.00 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    1 |  tg64@pp16384 | 5.31 ± 0.02 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    1 |         pp512 | 144.67 ± 0.42 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    1 |         pp512 | 143.85 ± 0.09 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    1 |        pp1024 | 139.12 ± 0.84 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    1 |        pp2048 | 131.00 ± 0.40 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    1 |        pp4096 | 117.22 ± 0.48 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    1 |        pp8192 |  97.62 ± 0.16 |
| deepseek2 671B Q4_K_R4         | 376.65 GiB |   671.03 B | CPU        |     128 |   q8_0 |  1 |   3 |  1024 |    1 |       pp16384 |  71.28 ± 0.04 |

</details>

---

👤 **ikawrakow** commented the **2025-03-23** at **06:24:32**:<br>

It looks like this can be useful, so I'll merge it.

---

👤 **ubergarm** commented the **2025-03-23** at **19:32:09**:<br>

Okay, I think I kind of understand things better now and have some interesting benchmark results.

## tl;dr;
Some systems will likely benefit from using Huge Pages. You can use either Explicit Huge Pages or Transparent Huge Pages and confirm they are in use to see similar benefits in inferencing performance.

There are some differences and depending on your exact requirements you may choose to use one or the other. For example, Explicit Huge Pages may support 1GiB sizes whereas THPs may not. THPs don't consume RAM when the model is not loaded as they are not manually pre-allocated.

## Explicit Huge Pages
Explicit huge pages are configured manually at boot time or before loading the model weights. These huge pages will consume RAM even when not in use and require special code changes contained in this PR.

Read above to see how to use them and run enable the code path with `llama-server -thp 1` as per this PR.

I would love to see if using 1GiB Hugepage Size improves performance beyond standard 2MiB size... Another day...


You can confirm they are being used after the model is loaded by checking:
```bash
$ grep Huge /proc/meminfo
AnonHugePages:   1857536 kB  # <--- random other small stuff is using THPs
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:   400000  # <--- I allocated twice as much given 2x NUMA nodes
HugePages_Free:    207154 # <--- model is loaded into Explicit Huge Pages
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB # <--- standard 2MiB Hugepagesize, feel free to try 1Gib and report back!
Hugetlb:        819200000 kB
```

## Transparent Huge Pages
If you want to use Transparent Huge Pages (THPs), you can enable them system wide before starting the application. This is simple enough and does not require any special `MADV_HUGEPAGE` code changes. It does not require any special code changes. It does probably require you use `--mmap 0` though which means you need enough RAM to hold the entire model weights.

```bash
## set to always so code does not require `MADV_HUGEPAGE`
$ echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
$ cat /sys/kernel/mm/transparent_hugepage/enabled
[always] madvise never

## set defrag to always
## might take programs longer to start up while waiting for memory compaction
## boosts likelihood of having enough huge pages available to allocate for LLM weights
$ echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
cat /sys/kernel/mm/transparent_hugepage/defrag
[always] defer defer+madvise madvise never

## run llama-server or llama-bench with mmap disabled
## given file based THPs are experimental kernel feature
## you have to disable mmap and allocate the weights into RAM to see benefits
$ ./bin/llama-server --mmap 0

## confirm they are working by checking after model finishes loading
$ grep Huge /proc/meminfo
AnonHugePages:  397645824 kB   # <--- This should be >= model weights size
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB # <--- doesn't matter as THPs are maybe always 2MiB regardless?
Hugetlb:               0 kB   # <--- no need for Explicit Huge Pages

```

## Benchmarks
You can see here how either Explicit or Transparent Hugepages give similar performance benefit over using standard Linux kernel 4k page size. 

This excellent PR gives you the flexibility to use whatever makes sense for your setup and explore 1GiB Explicit pages as well.

| threads | mmap | thp |          test | t/s |
| ------: | ---: | --: | ------------: | ---------------: |
| Baseline|      |     |               |                  |
|      64 |   -  |   0 |         pp512 |     93.35 ± 0.30 |
|      64 |   -  |   0 |        pp8192 |     63.20 ± 0.37 |
|      64 |   -  |   0 |    tg64@pp512 |      8.08 ± 0.06 |
|      64 |   -  |   0 |   tg64@pp8192 |      6.63 ± 0.02 |
|      96 |   -  |   0 |         pp512 |    113.24 ± 1.23 |
|      96 |   -  |   0 |        pp8192 |     79.62 ± 1.10 |
|      96 |   -  |   0 |    tg64@pp512 |      7.77 ± 0.00 |
|      96 |   -  |   0 |   tg64@pp8192 |      6.93 ± 0.07 |
|     128 |   -  |   0 |         pp512 |    136.60 ± 6.69 |
|     128 |   -  |   0 |        pp8192 |     97.97 ± 0.11 |
|     128 |   -  |   0 |    tg64@pp512 |      7.76 ± 0.00 |
|     128 |   -  |   0 |   tg64@pp8192 |      6.57 ± 0.01 |
| Explicit Huge Pages |      |     |               |                  |
|      64 |   -  |   1 |         pp512 |     96.22 ± 0.23 |
|      64 |   -  |   1 |        pp8192 |     63.60 ± 0.01 |
|      64 |   -  |   1 |    tg64@pp512 |      9.60 ± 0.00 |
|      64 |   -  |   1 |   tg64@pp8192 |      7.70 ± 0.01 |
|      96 |   -  |   1 |         pp512 |    118.49 ± 0.49 |
|      96 |   -  |   1 |        pp8192 |     83.16 ± 0.62 |
|      96 |   -  |   1 |    tg64@pp512 |      9.26 ± 0.00 |
|      96 |   -  |   1 |   tg64@pp8192 |      8.14 ± 0.00 |
|     128 |   -  |   1 |         pp512 |    141.94 ± 9.33 |
|     128 |   -  |   1 |        pp8192 |    100.87 ± 0.37 |
|     128 |   -  |   1 |    tg64@pp512 |      9.10 ± 0.00 |
|     128 |   -  |   1 |   tg64@pp8192 |      7.75 ± 0.00 |
| Transparent Huge pages |      |     |               |                  |
|      64 |    0 |   1 |         pp512 |     96.76 ± 4.34 |
|      64 |    0 |   1 |        pp8192 |     65.51 ± 0.30 |
|      64 |    0 |   1 |    tg64@pp512 |      9.53 ± 0.00 |
|      64 |    0 |   1 |   tg64@pp8192 |      7.67 ± 0.02 |
|      96 |    0 |   1 |         pp512 |    117.02 ± 0.07 |
|      96 |    0 |   1 |        pp8192 |     83.29 ± 0.65 |
|      96 |    0 |   1 |    tg64@pp512 |      9.32 ± 0.00 |
|      96 |    0 |   1 |   tg64@pp8192 |      8.17 ± 0.01 |
|     128 |    0 |   1 |         pp512 |    143.88 ± 6.28 |
|     128 |    0 |   1 |        pp8192 |    101.05 ± 0.02 |
|     128 |    0 |   1 |    tg64@pp512 |      9.26 ± 0.00 |
|     128 |    0 |   1 |   tg64@pp8192 |      7.85 ± 0.01 |

---

👤 **ikawrakow** commented the **2025-03-24** at **08:32:27**:<br>

@ubergarm Thank you for this.

I tried what you call "transparent huge pages" with WizardLM-8x22B on a Ryzen-5975WX system with 128 GB RAM. Model is quantized as `Q4_K_M` and is 85 GB. The system has a 16 GB RTX-4080 GPU to which I'm uploading 20 `ffn_down_exps` tensors. 

```
cat /sys/kernel/mm/transparent_hugepage/enabled
[always] madvise never

cat /sys/kernel/mm/transparent_hugepage/defrag
[always] defer defer+madvise madvise never

./bin/llama-bench -m ../../hf/WizardLM-2-8x22B-i1-GGUF/WizardLM-2-8x22B.i1-Q4_K_M.gguf -p 0 -n 128 -r 3 -t 8,16 -ngl 100 -ot "blk\.[0-9]\.ffn_down=CUDA0,blk\.1[0-9]\.ffn_down=CUDA0,blk\.20\.ffn_down=CUDA0,exps=CPU" -rtr 1 -fmoe 1
```

`-rtr 1` disables `mmap`, so that's equivalent to running with `-mmap 0`. After the model has been fully loaded and the benchmark is running:
```
grep Huge /proc/meminfo
AnonHugePages:  52006912 kB
ShmemHugePages:        0 kB
FileHugePages:         0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB
```
I.e., it gave me 52GB as huge pages. 15  GB are on the GPU. Didn't have enough free space for the entire model, so the remaining 18 GB were allocated as standard 4 kB pages, I guess. Performance dropped from the `3.5 t/s` that I'm getting with this command without THP to `2.5 t/s`. 

So, as you say, some systems will benefit from THP or "Explicit huge pages". This one doesn't. What is also interesting is that after running the THP experiment, performance without THP dropped to THP levels. I had to rerun multiple times (~10, alternating between `llama-perplexity`, `llama-cli`, and `llama-bench`) before performance slowly recovered to pre-THP experiment levels.

---

👤 **ikawrakow** commented the **2025-03-24** at **16:38:10**:<br>

> You can turn that stuff back off afterwards e.g.

I did. And yet performance with `llama-bench` stayed at `2.5 t/s`. I then ran `llama-cli`, and got `2.5 t/s` in the first run, `2.7 t/s` in the second run, `2.9 t/s` in the third run, but then it saturated at `2.9 t/s`. I then ran `llama-perplexity`, then went back to `llama-cli` and got `3.1 t/s`. I then used the CUDA disabled build to run `llama-cli` CPU only. It went up from `2.1 t/s` initially to `2.4 t/s` after 3 runs, where it saturated (but I think `2.4 t/s` is the max one can get CPU only on this system). Then, finally, going back to the CUDA build I got `3.5 t/s`, which was the performance before the THP test. 

I don't really know what happens in the kernel, but my guess is that due to caching, stuff ends up in the same memory banks, so I had to "shake it up" to get back to the original, more performant, state.

---

👤 **saood06** commented the **2025-03-25** at **11:48:08**:<br>

> To enable 1 GiB huge pages, you need to add
> 
> ```
> GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT} default_hugepagesz=1G
> ```
> 
> to `/etc/default/grub`, run `sudo update-grub`, and reboot. If you want to have some minimum reserved for 1GiB huge pages, use
> 
> ```
> GRUB_CMDLINE_LINUX_DEFAULT="${GRUB_CMDLINE_LINUX_DEFAULT} default_hugepagesz=1G hugepagesz=1G hugepages=N
> ```
> 
> where `N` is how many 1 GiB huge pages you want reserved.

The instructions differ if you do not have GRUB, as is the case for example on clear linux, where to enable it follow [this](https://www.clearlinux.org/clear-linux-documentation/guides/maintenance/configure-hugepages.html) guide.

I didn't test 2 MB pages, as it failed with `llama_mmap: mmap with huge page size 2 MiB failed (Cannot allocate memory)` and hugeadm was not trivially available (not in clear linux's package manager) and I didn't bother installing [libhugetlbfs](https://github.com/libhugetlbfs/libhugetlbfs) from source.