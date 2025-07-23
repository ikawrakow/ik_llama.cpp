### 🔀 [#284](https://github.com/ikawrakow/ik_llama.cpp/pull/284) - llama-bench: enable having different number of threads for tg and pp

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-24 |
| **Updated** | 2025-03-25 |

---

#### Description

All applications in the `examples` folder except `llama-bench` accept `-t` (to specify number of threads for token generation) and `-tb` (to specify number of threads for prompt processing, a.k.a. prefill) as command line arguments. This is handy  because often TG peak performance is reached at a lower number of threads, so one wants to use that instead  of the number of cores, which is good for maximum prompt processing speed. `llama-bench`, inherited from upstream, has its own command line argument parsing, where one only has available `-t` but not `-tb`.

This PR adds a new command line argument to `llama-bench`: `-tgb` (or `--threads-gen-batch`).  One can use it as, e.g.,
```
./bin/llama-bench -tgb 4,16 -p 512 -n 128 other_arguments
```
where 4 threads will be used for the `tg128` test, and 16 threads will be used for the `pp512` test. For tests that are a combination of prefill and gen (`-pg`, `-gp`), the batch number of threads will be used for prefill, and the gen number of threads will be used for token generation. One can also specify multiple pairs of `{t_gen, t_batch}` for the `-tgb` argument, separating them with a semicolon. E.g.,
```
./bin/llama-bench -tgb 2,16;4,16;8,32
```

The `-t` argument continues to work as before. It adds a pair of the same integer in the list of `{t_hen, t_batch}` number of thread pairs. 

**Caveat:** For `-p` the batch number of threads is added to the table. For all other tests the gen number of threads is printed. This is of course appropriate for `-n` and `-gp`, but it becomes confusing for `-pg`, where the batch and gen number of threads both matter for the reported performance.  I guess, it would be better to print both thread numbers in this case, but this is not done in this PR.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-03-25** at **16:27:02**:<br>

Thanks for this one, should help optimize the big xeon 6980P given previous testing suggests that pp likes more threads than tg.