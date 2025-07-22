### 🐛 [#436](https://github.com/ikawrakow/ik_llama.cpp/issues/436) - Bug: Saving the prompt cache causes Segfault

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-20 |
| **Updated** | 2025-06-06 |

---

#### Description

### What happened?

Triggered via:

```
curl --header "Content-Type: application/json" \
   --request POST \
   --data '{"filename":"test.bin"}' [...]:8080/slots/0?action=save
```

### Name and Version

134d5481737c05421eb1ba7cd7573136e3fdbd69

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
Segmentation fault (core dumped)
```

---

#### 💬 Conversation

👤 **saood06** commented the **2025-05-28** at **06:30:58**:<br>

I finally got some time to look into this more and I think the cause of issue seems to be the fact that the function [here](https://github.com/ikawrakow/ik_llama.cpp/blob/ccd6d9cdf6851f7042c48d682daf47bc0e2eca27/src/llama.cpp#L21453) references kv_self.k_l and kv_self.v_l and since I was using Deepseek with FlashMLA-3 where kv_l see [here](https://github.com/ikawrakow/ik_llama.cpp/blob/ccd6d9cdf6851f7042c48d682daf47bc0e2eca27/src/llama.cpp#L2995) is used instead (and kvt_l would have also been used if I was using a different implementation of MLA).

@ikawrakow thoughts? Would one need to update this function to take into account MLA and it's different configurations or can this code be refactored/rewritten in a different way? (I only ask the latter since it seems odd to me that this is the only thing that broke because of the new kv_l and kvt_l and perhaps it's because other code is written in a way where it didn't break).

---

👤 **ikawrakow** commented the **2025-05-28** at **08:08:32**:<br>

Yes, this part has not been updated at all. There are two issues:
* Using `kv_l` and possibly `kvt_l` instead of `k_l` and `v_l`. I guess, it would be best to just get rid of `kv_l` and `kvt_l` (they came from the initial implementation) and just use `k_l` and `v_l` instead. This would be relatively easy to change.
* I have changed the K-cache to be `head_size x n_heads x n_tokens` instead of `head_size*n_head, n_tokens`. This was needed to support `Q8_KV`, which uses per row scales. When the K-cache is not `Q8_KV` it should not make a difference, but I haven't checked the cache manipulating functions if there is some confusion because of the changed tensor dimensions. One possible approach is to just remove the `Q8_KV` cache option (performance benefits were disappointingly small) and go back to the original `llama.cpp` K-cache layout. Otherwise one needs to carefully check everywhere where the cache is being manipulated.

---

👤 **saood06** commented the **2025-05-28** at **08:56:12**:<br>

>Using `kv_l` and possibly `kvt_l` instead of `k_l` and `v_l`. I guess, it would be best to just get rid of `kv_l` and `kvt_l` (they came from the initial implementation) and just use `k_l` and `v_l` instead. This would be relatively easy to change.

Yes, I remember that. Even if we get rid of the  `kv_l` and `kvt_l`, the `write_kv_cache_data` and `read_kv_cache_data` would still need to be updated to account for an optional V-cache it seems like. Is there anything else it would need to account for, since that is the only change I can think of?

>I have changed the K-cache to be `head_size x n_heads x n_tokens` instead of `head_size*n_head, n_tokens`. This was needed to support `Q8_KV`, which uses per row scales. When the K-cache is not `Q8_KV` it should not make a difference, but I haven't checked the cache manipulating functions if there is some confusion because of the changed tensor dimensions. One possible approach is to just remove the `Q8_KV` cache option (performance benefits were disappointingly small) and go back to the original `llama.cpp` K-cache layout. Otherwise one needs to carefully check everywhere where the cache is being manipulated.

That is your decision to make. Alternatively couldn't we just put a warning when someone uses the `Q8_KV` cache that prompt saving/loading will not work? I'd at least say to confirm if it really does break things before removing it, as even though I don't really use it, I know it still does boost performance, and I would hate for your effort to have gone to waste. But again that is your call to make.

---

👤 **ikawrakow** commented the **2025-05-28** at **09:17:16**:<br>

OK, let's start with the required changes without worrying about `Q8_KV`. Do you want to do it?

---

👤 **saood06** commented the **2025-05-28** at **09:25:04**:<br>

>Do you want to do it?

I don't mind giving it an attempt, but I'm heading off for now and won't be available till tomorrow at the earliest.

---

👤 **ikawrakow** commented the **2025-05-28** at **09:30:40**:<br>

> but I'm heading off for now and won't be available till tomorrow at the earliest.

It is not really urgent, so that's OK.

I'm experimenting with some stuff right now, but if I find a moment before tomorrow I may start and let you finish (I'm not really setup for testing that sort of thing).

---

👤 **ikawrakow** commented the **2025-05-28** at **11:21:25**:<br>

See #469

---

👤 **saood06** commented the **2025-06-02** at **01:23:45**:<br>

Although it was tested and works, there may still be some issues with it, since I just crashed with this when attempting to save (and it didn't even write the prompt to the file before it crashed)

`/ik_llama.cpp/ggml/src/ggml-backend.c:251: GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds") failed`

I have the coredump and will attempt to debug it later.

Edit: Happens consistently now (might be larger prompts?) and might as well share the backtrace.

```gdb
#0  0x0000557fb630e177 in __GI___wait4 () at ../sysdeps/unix/sysv/linux/wait4.c:30
30      in ../sysdeps/unix/sysv/linux/wait4.c
#1  0x0000557fb6a19270 in ggml_print_backtrace () at /home/saood06/ik_main/ik_llama.cpp/ggml/src/ggml.c:242
242             waitpid(pid, &wstatus, 0);
#2  ggml_abort (file=0x557fb80dac98 "/home/saood06/ik_main/ik_llama.cpp/ggml/src/ggml-backend.c", line=251, fmt=0x557fb80d709e "GGML_ASSERT(%s) failed") at /home/saood06/ik_main/ik_llama.cpp/ggml/src/ggml.c:269
269         ggml_print_backtrace();
#3  0x0000557fb6a4e878 in ggml_backend_tensor_get (tensor=<optimized out>, data=<optimized out>, offset=<optimized out>, size=<optimized out>) at /home/saood06/ik_main/ik_llama.cpp/ggml/src/ggml-backend.c:251
251         GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
#4  ggml_backend_tensor_get (tensor=0x557fcb626b50, data=0x552271847010, offset=0, size=175865856) at /home/saood06/ik_main/ik_llama.cpp/ggml/src/ggml-backend.c:246
246     GGML_CALL void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
#5  0x0000557fb837831d in llama_data_write_file::write_tensor_data (this=0x7fff68a82700, tensor=<optimized out>, offset=<optimized out>, size=175865856) at /usr/lib64/gcc/x86_64-generic-linux/14/../../../../include/c++/14/bits/stl_vector.h:1262
1262          data() _GLIBCXX_NOEXCEPT
#6  llama_data_write::write_kv_cache_data (this=0x7fff68a82700, ctx=0x557fcb624e00, cell_ranges=std::vector of length 1, capacity 1 = {...}) at /home/saood06/ik_main/ik_llama.cpp/src/llama.cpp:21461
21461                   write_tensor_data(kv_self.k_l[il], range.first * k_size_row, buf_size);
#7  llama_data_write::write_kv_cache (this=this@entry=0x7fff68a82700, ctx=ctx@entry=0x557fcb624e00, seq_id=seq_id@entry=1) at /home/saood06/ik_main/ik_llama.cpp/src/llama.cpp:21552
21552           write_kv_cache_data(ctx, cell_ranges);
#8  0x0000557fb8379618 in llama_state_seq_get_data_internal (ctx=0x557fcb624e00, data_ctx=..., seq_id=1) at /home/saood06/ik_main/ik_llama.cpp/src/llama.cpp:22155
22155       data_ctx.write_kv_cache(ctx, seq_id);
#9  llama_state_seq_save_file_internal (ctx=0x557fcb624e00, filepath=<optimized out>, seq_id=1, tokens=0x557fcb82f620, n_token_count=<optimized out>) at /home/saood06/ik_main/ik_llama.cpp/src/llama.cpp:22205
22205       llama_state_seq_get_data_internal(ctx, data_ctx, seq_id);
#10 llama_state_seq_save_file (ctx=0x557fcb624e00, filepath=<optimized out>, seq_id=1, tokens=0x557fcb82f620, n_token_count=<optimized out>) at /home/saood06/ik_main/ik_llama.cpp/src/llama.cpp:22257
22257           return llama_state_seq_save_file_internal(ctx, filepath, seq_id, tokens, n_token_count);
#11 0x0000557fb855d8e6 in server_context::process_single_task (this=0x7fff68a83bb0, task=...) at /home/saood06/ik_main/ik_llama.cpp/examples/server/server.cpp:1760
1760                        const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id + 1, slot->cache_tokens.data(), token_count);
#12 0x0000557fb850a310 in std::function<void(server_task&)>::operator() (this=0x7fff68a84790, __args#0=...) at /usr/lib64/gcc/x86_64-generic-linux/14/../../../../include/c++/14/bits/std_function.h:591
591             return _M_invoker(_M_functor, std::forward<_ArgTypes>(__args)...);
#13 server_queue::start_loop (this=this@entry=0x7fff68a846e8) at /home/saood06/ik_main/ik_llama.cpp/examples/server/server.cpp:479
479                     callback_new_task(task);
#14 0x0000557fb84b4090 in main (argc=<optimized out>, argv=<optimized out>) at /home/saood06/ik_main/ik_llama.cpp/examples/server/server.cpp:3509
3509        ctx_server.queue_tasks.start_loop();
```

---

👤 **saood06** commented the **2025-06-02** at **01:23:45**:<br>

Although it was tested and works, there may still be some issues with it, since I just crashed with this.

`/ik_llama.cpp/ggml/src/ggml-backend.c:251: GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds") failed`

I have the coredump and will debug it later.

---

👤 **saood06** commented the **2025-06-03** at **12:52:48**:<br>

I poked around the coredump a bit, and for the ggml_backend_tensor_get call I saw the offset is 0, with size of 175865856. I manually calculated ggml_nbytes to be 92307456, which is close to half the size.

I have a theory that it stops working past the batch size, but even if I do confirm that (or find the cutoff point of how many tokens it stops working at), I still don't think I'd know why `k_size_row` is wrong ( `buf_size = range_size * k_size_row`, and `range_size` is correct, so `k_size_row` must be wrong ) and how to fix it.

@ikawrakow 

Would confirming that it breaks past a token size be useful? Or is there something else I could do in order to help find why this is breaking?

---

👤 **ikawrakow** commented the **2025-06-03** at **13:37:33**:<br>

There is a confusion with the size of the tensor, and one needs to carefully go through the code to sort it out. As I wrote earlier, I have changed the K cache to be `k_had_size x n_head x n_tokens`, while the code is written from the point of view that the K cache is `k_head_size * n_head x n_tokens`. Somewhere things go wrong because of that. If you don't see it, and I don't see it, I can revert the shape change (it is isolated to a very few places).

---

👤 **saood06** commented the **2025-06-03** at **14:15:24**:<br>

> There is a confusion with the size of the tensor, and one needs to carefully go through the code to sort it out. As I wrote earlier, I have changed the K cache to be `k_had_size x n_head x n_tokens`, while the code is written from the point of view that the K cache is `k_head_size * n_head x n_tokens`. Somewhere things go wrong because of that. If you don't see it, and I don't see it, I can revert the shape change (it is isolated to a very few places).

I know you said that earlier, but I don't get why it worked with 469 tokens but it failed with ~8.7K and ~3.7K tokens. I'm not saying that reason is wrong, I'm just saying if that is the reason, I couldn't see where the shape change caused the issue and why it worked with a small `n_tokens` but not a large one.

I will gladly test whatever change you think will fix this (whether that be if you revert the shape change, or if you can see where things go wrong).

---

👤 **saood06** commented the **2025-06-06** at **06:49:31**:<br>

@ikawrakow 

I looked into https://github.com/ikawrakow/ik_llama.cpp/pull/208/commits/0280b8d52b69de0ee0130d45a698d5e5dc4c9977 and saw the changes you were talking about, but I'm still a little confused. 

For non MLA you did change this:

`k = ggml_new_tensor_1d(ctx, type_k, n_embd_k_gqa*kv_size);`

to:

`k = ggml_new_tensor_2d(ctx, type_k, n_embd_head_k, n_head_kv*kv_size);`

but with MLA it only changed from

`ggml_tensor * kv = ggml_new_tensor_1d(ctx, cache.type_k, (kv_lora_rank + n_embd_head_qk_rope)*kv_size);`

to

`ggml_tensor * kv = ggml_new_tensor_2d(ctx, cache.type_k, kv_lora_rank + n_embd_head_qk_rope, kv_size);`

And `write_kv_cache_data` / `read_kv_cache_data` currently use:

```
const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();
const uint64_t k_size_row = ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa);
```

I did figure out why it would seem to work for a small amount of tokens but not for a large amount of tokens, the assert above (`GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds") failed`) happens when `k_size_row` was too large and you are saving enough tokens, but if `k_size_row` is too small (which happened on some of my attempts to fix this) then it will load and save but only a fraction of the actual context is actually restored which becomes very obvious both by the file it generates being too small for the amount of KV data, and the output following the restore being clearly missing a lot of the information that was not able to be restored.

In all my testing I launched the server with 80K context which allowed it to work with a small amount of tokens when `k_size_row` was too large, but it was writing and saving a file that was obviously much larger than it should be, but based on some napkin math it would fail at ~1800 tokens which explains why my attempts above that consistently failed. (Which means the size it writes is off by a factor of  ~43x) 

So I'm not sure if `write_kv_cache_data` / `read_kv_cache_data` need to take into account MLA (on top of the shape change you made when adding Q8_KV, and in either situation how `k_size_row` should be calculated. 

>Somewhere things go wrong because of that. 

I do think the changes needed will be isolated to `write_kv_cache_data` / `read_kv_cache_data` but I can't figure it out. Do you mind looking into it again?

---

👤 **ikawrakow** commented the **2025-06-06** at **07:10:08**:<br>

We have `n_embd_k_gqa = n_embd_head_k * n_head_kv`, so a 1D tensor of size `n_embd_k_gqa * kv_size` is the same as a 1D tensor of size `n_embd_head_k *  n_head_kv * kv_size`, which can be viewed as a 2D tensor of size `n_embd_head_k x n_head_kv*kv_size`.

In the case of MLA, it was originally a 1D tensor of size `(kv_lora_rank + n_embd_head_qk_rope)*kv_size`, so it becomes a 2D tensor of size `kv_lora_rank + n_embd_head_qk_rope x kv_size`.

Does this answer the question?

---

👤 **ikawrakow** commented the **2025-06-06** at **07:26:35**:<br>

So, the presence of `hparams.n_embd_k_s()` (needed for Mamba) makes it more complicated. But my K-cache change to 2D does not work with Mamba anyway (does `ik_llama.cpp` work for Mamba at all? I wouldn't think so).

So, we can simply disregard Mamba. One needs to change `n_embd_k_gqa` in case it is MLA, but other than that it should work with KV cache that is not `Q8_KV`.

---

👤 **saood06** commented the **2025-06-06** at **07:29:43**:<br>

> We have `n_embd_k_gqa = n_embd_head_k * n_head_kv`, so a 1D tensor of size `n_embd_k_gqa * kv_size` is the same as a 1D tensor of size `n_embd_head_k * n_head_kv * kv_size`, which can be viewed as a 2D tensor of size `n_embd_head_k x n_head_kv*kv_size`.

That does clarify some things for me.

> In the case of MLA, it was originally a 1D tensor of size `(kv_lora_rank + n_embd_head_qk_rope)*kv_size`, so it becomes a 2D tensor of size `kv_lora_rank + n_embd_head_qk_rope x kv_size`.

Which is different from the normal case, so am I correct that `write_kv_cache_data` / `read_kv_cache_data` will need to be modified to calculate `k_size_row` differently if you are saving/loading an MLA cache?

> Does this answer the question?

I think so. That does line up with the ~43x factor that the size was off by. (For Deepseek V3 I know `n_embd_head_qk_rope = 64, kv_lora_rank = 512` and  `n_embd_k_gqa = 24576`, and `24576/(512+64)=42⅔`

---

👤 **ikawrakow** commented the **2025-06-06** at **07:35:42**:<br>

So, this is done just using the `llama_hparams` struct. Which does not know if MLA is being used because the MLA flag is in the `llama_cparams` struct. I have run into this stupid issue a number of times, but never took the time to sort this out. The cache writing needs to know if MLA was used to calculate it so it can use and record the correct cache size.

---

👤 **saood06** commented the **2025-06-06** at **07:47:29**:<br>

> So, this is done just using the `llama_hparams` struct. Which does not know if MLA is being used because the MLA flag is in the `llama_cparams` struct. I have run into this stupid issue a number of times, but never took the time to sort this out. The cache writing needs to know if MLA was used to calculate it so it can use and record the correct cache size.

You have access to the ctx object (which contains `cparams` which is a `llama_cparams` struct ) so I don't see why that is an issue.

---

👤 **saood06** commented the **2025-06-06** at **07:47:29**:<br>

> So, this is done just using the `llama_hparams` struct. Which does not know if MLA is being used because the MLA flag is in the `llama_cparams` struct. I have run into this stupid issue a number of times, but never took the time to sort this out. The cache writing needs to know if MLA was used to calculate it so it can use and record the correct cache size.

You have access to the ctx object (which contains llama_cparams) so I don't see why that is an issue.

---

👤 **ikawrakow** commented the **2025-06-06** at **07:52:34**:<br>

> You have access to the ctx object (which contains llama_cparams) so I don't see why that is an issue.

You don't have access to `llama_cparams` when loading the mode for instance. If you have access to the context when writing the cache, you can do it that way. Otherwise, #490 has a quick hack to add the MLA flag to `llama_hparams`. If it set, the `n_embd_k_gqa()` will now return the correct size needed when writing the cache.

---

👤 **saood06** commented the **2025-06-06** at **08:04:43**:<br>

>You don't have access to `llama_cparams` when loading the mode for instance. If you have access to the context when writing the cache, you can do it that way. Otherwise, [#490](https://github.com/ikawrakow/ik_llama.cpp/issues/490) has a quick hack to add the MLA flag to `llama_hparams`. If it set, the `n_embd_k_gqa()` will now return the correct size needed when writing the cache.

I'm testing a fix without #490. If it works I'll make the PR. I don't think #490 is needed for this, but you know better if it is helpful in other situations.

---

👤 **saood06** commented the **2025-06-06** at **08:50:01**:<br>

Just in case anyone reads through this later #496 is the PR with the hack that was not used, and not #490.

>(does ik_llama.cpp work for Mamba at all? I wouldn't think so).

I'm not sure. Is there any reason you think Mamba support would have been broken since it was supported before this repo diverged?

I looked into adding jamba and mamba-2 here as both PR's were functional around the time ik_llama.cpp has last merged which means a lot of the commits should be able to be cherry-picked with relative ease. I never did it since I don't care about those architectures enough to do it for my own desires, and there didn't seem to be enough demand for me to do it for that reason.