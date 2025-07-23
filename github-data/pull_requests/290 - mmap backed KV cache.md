### 🔀 [#290](https://github.com/ikawrakow/ik_llama.cpp/pull/290) - mmap backed KV cache

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-03-25 |
| **Updated** | 2025-03-27 |

---

#### Description

Port of https://github.com/ggml-org/llama.cpp/pull/11580

I have not used this as I no longer need it ever since the old KV cache is no longer allocated (this helped when both were allocated as it would not ever actually touch the pages of the old KV cache thus allowing me to not page out to disk), but it still doesn't hurt my performance.

Finally deciding to grab the code from my very old local branch and put it here in case it ends up being beneficial to anyone.

This PR always uses the new buffer type for KV cache, as there is no toggle implemented. This can be added if this ends up being useful in some situations, but a loss in others. So far I haven't found a situation where it causes performance loss so far though.

In theory this should be better for NUMA as I do remember noting it caused a more even split of memory usage across the two nodes on my machine.

This also might have the benefit of letting you allocate the full context size of a model only getting performance loss when you actually go over that limit as it will avoid paging until then.

- [x] I have read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md)
- Self-reported review complexity:
  - [X] Low
  - [ ] Medium
  - [ ] High

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-03-27** at **05:14:15**:<br>

I think it needs to be ifdef'ed so the code will still build on Windows.

I wouldn't make it the default unconditionally, we should be able to turn it on/off via a command line parameter. It would be also useful of @ubergarm tested performance implications. 

Concerning NUMA advantage: yes, it will spread the KV cache more evenly between NUMA nodes. But aren't we concerned it may result in each NUMA node having to fetch KV cache data from another NUMA node. The KV cache grows as generation progresses, so in each new evaluation threads access different portions of the KV cache, so the strategy of evenly spreading the cache across NUMA nodes will be only meaningful if we also had something in place that would make threads always process the same portions of the KV cache.

---

👤 **saood06** commented the **2025-03-27** at **05:31:58**:<br>

> I think it needs to be ifdef'ed so the code will still build on Windows.
> 
> I wouldn't make it the default unconditionally, we should be able to turn it on/off via a command line parameter.

Yes I agree on the needed changes if this is to be merged in, I mainly just remembered I did this, and made a draft PR in case anyone finds it useful.

>It would be also useful of @ubergarm tested performance implications.

I'd be interested to know if it affected performance for him, since it doesn't hurt or help my performance anymore.

> Concerning NUMA advantage: yes, it will spread the KV cache more evenly between NUMA nodes. But aren't we concerned it may result in each NUMA node having to fetch KV cache data from another NUMA node. The KV cache grows as generation progresses, so in each new evaluation threads access different portions of the KV cache, so the strategy of evenly spreading the cache across NUMA nodes will be only meaningful if we also had something in place that would make threads always process the same portions of the KV cache.

The distribution of the KV cache never resulted in a performance uplift for me (and based on comments in the original PR from both the author and others it didn't affect them). From what I remember it may have allowed me to turn off numa_balancing for my system without a negatively impact (like it may do, my memory and notes aren't very clear). The main reason I used it was it avoided paging to disks because the old MLA implementation still had the large unneeded KV cache.

I do think your concern is valid but in practice this PR doesn't seem to impact performance, and I'm not really sure why it is performance neutral.