## 📌 [Issue #650](https://github.com/ikawrakow/ik_llama.cpp/issues/650) - Bug: segfault in llama-quantize iq3_kt

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-26 |
| **Updated** | 2025-07-26 |

---

## 📄 Description

### What happened?

I've been unable to quantize using iq3_kt in a recent reicpe. I ran the script in `gdb` and grabbed a `bt` after it segfaults. I'm am sure sure I have used iq3_kt before, but I believe that was on a different CPU rig. I could try to recreate this issue with a smaller quant, or on a different rig, or roll back to an earlier version to check for possible regressions.

Here is the log and backtrace just to document it in case someone else sees anything similar.
<details>

<summary>👈 llama-quantize command and debugging logs and backtrace</summary>

```bash
#!/usr/bin/env bash

# Repeating Layers [0-61]

custom="
# Attention
blk\..*\.attn_q.*=iq4_kt
blk\..*\.attn_k.*=iq4_kt
blk\..*\.attn_v.*=iq4_kt
blk\..*\.attn_output.*=iq4_kt

# Routed Experts
blk\..*\.ffn_down_exps\.weight=iq3_kt
blk\..*\.ffn_(gate|up)_exps\.weight=iq2_kt

# Non-Repeating Layers
token_embd\.weight=iq4_kt
output\.weight=iq6_k
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

numactl -N 1 -m 1 \
gdb -q --args ./build/bin/llama-quantize \
    --custom-q "$custom" \
    --imatrix /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/imatrix-Qwen3-Coder-480B-A35B-Instruct-Q8_0.dat \
    /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-BF16-00001-of-00021.gguf \
    /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-IQ2_KT.gguf \
    IQ2_KT \
    192

[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
Downloading separate debug info for /lib/x86_64-linux-gnu/libgomp.so.1...
main: build = 3822 (4e9c78c0)
main: built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: quantizing '/mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-BF16-00001-of-00021.gguf' to '/mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-IQ2_KT.gguf' as IQ2_KT using 192 threads
llama_model_loader: additional 20 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 37 key-value pairs and 747 tensors from /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-BF16-00001-of-00021.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 Coder 480B A35B Instruct
llama_model_loader: - kv   3:                           general.finetune str              = Instruct
llama_model_loader: - kv   4:                           general.basename str              = Qwen3-Coder
llama_model_loader: - kv   5:                         general.size_label str              = 480B-A35B
llama_model_loader: - kv   6:                            general.license str              = apache-2.0
llama_model_loader: - kv   7:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-Cod...
llama_model_loader: - kv   8:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   9:                       qwen3moe.block_count u32              = 62
llama_model_loader: - kv  10:                    qwen3moe.context_length u32              = 262144
llama_model_loader: - kv  11:                  qwen3moe.embedding_length u32              = 6144
llama_model_loader: - kv  12:               qwen3moe.feed_forward_length u32              = 8192
llama_model_loader: - kv  13:              qwen3moe.attention.head_count u32              = 96
llama_model_loader: - kv  14:           qwen3moe.attention.head_count_kv u32              = 8
llama_model_loader: - kv  15:                    qwen3moe.rope.freq_base f32              = 10000000.000000
llama_model_loader: - kv  16:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  17:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  18:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  19:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  20:                          general.file_type u32              = 32
llama_model_loader: - kv  21:                      qwen3moe.expert_count u32              = 160
llama_model_loader: - kv  22:        qwen3moe.expert_feed_forward_length u32              = 2560
llama_model_loader: - kv  23: qwen3moe.expert_shared_feed_forward_length u32              = 0
llama_model_loader: - kv  24:               general.quantization_version u32              = 2
llama_model_loader: - kv  25:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  26:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  27:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  28:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  29:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  30:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  31:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  32:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  33:                    tokenizer.chat_template str              = {% macro render_item_list(item_list, ...
llama_model_loader: - kv  34:                                   split.no u16              = 0
llama_model_loader: - kv  35:                                split.count u16              = 21
llama_model_loader: - kv  36:                        split.tensors.count i32              = 747
llama_model_loader: - type  f32:  311 tensors
llama_model_loader: - type bf16:  436 tensors
================================ Have weights data with 497 entries
[   1/ 747]                    token_embd.weight - [ 6144, 151936,     1,     1], type =   bf16, Using custom type iq4_kt for tensor token_embd.weight

====== llama_model_quantize_internal: did not find weights for token_embd.weight
[New Thread 0x7f1f54cfe6c0 (LWP 298924)]
[New Thread 0x7f1f544fd6c0 (LWP 298925)]
...
[Thread 0x7f1ef1f7b6c0 (LWP 299113) exited]
converting to iq4_kt .. Adding custom rule blk\..*\.attn_q.* -> iq4_kt
Adding custom rule blk\..*\.attn_k.* -> iq4_kt
Adding custom rule blk\..*\.attn_v.* -> iq4_kt
Adding custom rule blk\..*\.attn_output.* -> iq4_kt
Adding custom rule blk\..*\.ffn_down_exps\.weight -> iq3_kt
Adding custom rule blk\..*\.ffn_(gate|up)_exps\.weight -> iq2_kt
Adding custom rule token_embd\.weight -> iq4_kt
Adding custom rule output\.weight -> iq6_k
load_imatrix: imatrix dataset='ubergarm-imatrix-calibration-corpus-v02.txt'
load_imatrix: loaded 497 importance matrix entries from /mnt/raid/models/ubergarm/Qwen3-Coder-480B-A35B-Instruct-GGUF/imatrix-Qwen3-Coder-480B-A35B-Instruct-Q8_0.dat computed on 840 chunks
prepare_imatrix: have 497 importance matrix entries
[New Thread 0x7f1ef0f796c0 (LWP 299121)]
[New Thread 0x7f1ef177a6c0 (LWP 299122)]
...
[Thread 0x7f1b227fc6c0 (LWP 299311) exited]
[Thread 0x7f1dc67fc6c0 (LWP 299163) exited]
size =  1780.50 MiB ->   445.70 MiB
[   2/ 747]             blk.0.attn_k_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
[   3/ 747]                  blk.0.attn_k.weight - [ 6144,  1024,     1,     1], type =   bf16, Using custom type iq4_kt for tensor blk.0.attn_k.weight
[New Thread 0x7f1b227fc6c0 (LWP 299312)]
[Thread 0x7f1b227fc6c0 (LWP 299312) exited]
[New Thread 0x7f1b22ffd6c0 (LWP 299313)]
...
converting to iq4_kt .. cluster_points: Oops. Cluster 4 has no points:  0 1 0 0
cluster_points: 1 out of 625 clusters dir not have any points
...
size =    12.00 MiB ->     3.00 MiB
[   4/ 747]             blk.0.attn_output.weight - [12288,  6144,     1,     1], type =   bf16, Using custom type iq4_kt for tensor blk.0.attn_output.weight
...
converting to iq4_kt .. [New Thread 0x7f1b217fa6c0 (LWP 299887)]
...
size =   144.00 MiB ->    36.02 MiB
[   5/ 747]             blk.0.attn_q_norm.weight - [  128,     1,     1,     1], type =    f32, size =    0.000 MB
[   6/ 747]                  blk.0.attn_q.weight - [ 6144, 12288,     1,     1], type =   bf16, Using custom type iq4_kt for tensor blk.0.attn_q.weight
...
converting to iq4_kt .. [New Thread 0x7f1b217fa6c0 (LWP 300271)]
...
size =   144.00 MiB ->    36.05 MiB
[   7/ 747]                  blk.0.attn_v.weight - [ 6144,  1024,     1,     1], type =   bf16, Using custom type iq4_kt for tensor blk.0.attn_v.weight
...
converting to iq4_kt .. [New Thread 0x7f1b217fa6c0 (LWP 300660)]
...
size =    12.00 MiB ->     3.00 MiB
[   8/ 747]               blk.0.attn_norm.weight - [ 6144,     1,     1,     1], type =    f32, size =    0.023 MB
[   9/ 747]           blk.0.ffn_down_exps.weight - [ 2560,  6144,   160,     1], type =   bf16, Using custom type iq3_kt for tensor blk.0.ffn_down_exps.weight
...
converting to iq3_kt .. [New Thread 0x7f1fd5d446c0 (LWP 301043)]
...
[New Thread 0x7f20305f96c0 (LWP 311379)]
[New Thread 0x7f202fdf86c0 (LWP 311380)]

Thread 12423 "llama-quantize" received signal SIGSEGV, Segmentation fault.
[Switching to Thread 0x7f1fd5d446c0 (LWP 311369)]
0x00007ffff77cee9d in (anonymous namespace)::quantize_row_iq3_kt_impl (x=0x7f19767ff010, vy=0x7f1667daa010, n_per_row=2560, 
    quant_weights=0x7fffe16f8010, all_scales=0x7f1c88000b70, all_weights=0x7f1c88000cc0, qtmp=0x7f1c880034d0)
    at /home/w/projects/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:9179
9179	                for (int j = 0; j < Q::kGroupSize; ++j) *xt++ = q[j];
[?2004h[?2004l
[?2004h(gdb) bt
[?2004l
#0  0x00007ffff77cee9d in (anonymous namespace)::quantize_row_iq3_kt_impl (x=0x7f19767ff010, vy=0x7f1667daa010, n_per_row=2560, 
    quant_weights=0x7fffe16f8010, all_scales=0x7f1c88000b70, all_weights=0x7f1c88000cc0, qtmp=0x7f1c880034d0)
    at /home/w/projects/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:9179
#1  0x00007ffff77cfd24 in quantize_iq3_kt (src=0x7f19767ff010, dst=0x7f1667daa010, nrows=7, n_per_row=2560, imatrix=0x7fffe16f8010)
    at /home/w/projects/ik_llama.cpp/ggml/src/iqk/iqk_quantize.cpp:9298
#2  0x00007ffff6c8e4ba in ggml_quantize_chunk (type=GGML_TYPE_IQ3_KT, src=0x7f19767ff010, dst=0x7f1667daa010, start=0, nrows=7, n_per_row=2560, 
    imatrix=0x7fffe16f8010) at /home/w/projects/ik_llama.cpp/ggml/src/ggml.c:24019
#3  0x00007ffff7ced7e2 in operator() (__closure=0x555557273448) at /home/w/projects/ik_llama.cpp/src/llama.cpp:19839
#4  0x00007ffff7d19743 in std::__invoke_impl<void, llama_tensor_quantize_internal(ggml_type, float const*, void*, int64_t, int64_t, int64_t, float const*, std::vector<std::thread>&, int)::<lambda()> >(std::__invoke_other, struct {...} &&) (__f=...) at /usr/include/c++/13/bits/invoke.h:61
#5  0x00007ffff7d195b3 in std::__invoke<llama_tensor_quantize_internal(ggml_type, float const*, void*, int64_t, int64_t, int64_t, float const*, std::vector<std::thread>&, int)::<lambda()> >(struct {...} &&) (__fn=...) at /usr/include/c++/13/bits/invoke.h:96
#6  0x00007ffff7d194ba in std::thread::_Invoker<std::tuple<llama_tensor_quantize_internal(ggml_type, float const*, void*, int64_t, int64_t, int64_t, float const*, std::vector<std::thread>&, int)::<lambda()> > >::_M_invoke<0>(std::_Index_tuple<0>) (this=0x555557273448)
    at /usr/include/c++/13/bits/std_thread.h:292
#7  0x00007ffff7d19472 in std::thread::_Invoker<std::tuple<llama_tensor_quantize_internal(ggml_type, float const*, void*, int64_t, int64_t, int64_t, float const*, std::vector<std::thread>&, int)::<lambda()> > >::operator()(void) (this=0x555557273448) at /usr/include/c++/13/bits/std_thread.h:299
#8  0x00007ffff7d19432 in std::thread::_State_impl<std::thread::_Invoker<std::tuple<llama_tensor_quantize_internal(ggml_type, float const*, void*, int64_t, int64_t, int64_t, float const*, std::vector<std::thread>&, int)::<lambda()> > > >::_M_run(void) (this=0x555557273440)
    at /usr/include/c++/13/bits/std_thread.h:244
#9  0x00007ffff68ecdb4 in std::execute_native_thread_routine (__p=0x555557273440) at ../../../../../src/libstdc++-v3/src/c++11/thread.cc:104
#10 0x00007ffff649caa4 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#11 0x00007ffff6529c3c in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78
[?2004h
(gdb) 
```

</details>


### Name and Version

$ ./build/bin/llama-server --version
version: 3822 (4e9c78c0)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

## 💬 Conversation

👤 **ikawrakow** commented on **2025-07-26** at **07:38:06**

I suspect this issue and [#649](https://github.com/ikawrakow/ik_llama.cpp/issues/649) are related and likely caused by blocks filled by zeros in the model weights and/or corresponding imatrix data. In the case of `IQ2_KL` we get NaNs in the quantized model, which then causes the assert in the FA kernel. In the case of `IQ3_KT` we crash already during quantization because the search for the best match fails. 

I'll need to add guards against that.