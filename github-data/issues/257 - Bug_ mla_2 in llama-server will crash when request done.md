### 🐛 [#257](https://github.com/ikawrakow/ik_llama.cpp/issues/257) - Bug: mla=2 in llama-server will crash when request done

| **Author** | `orca-zhang` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-03-14 |
| **Updated** | 2025-03-15 |

---

#### Description

### What happened?

> llama-server -m /root/models/DeepSeek-R1-11446-Q2_K/DeepSeek-R1-11446-Q2_K-00001-of-00030.gguf -fa --temp 0.6 --top-p 0.95 -s 3047 -t 62 -nkvo -c 163840 -ngl 0 -mla 2 -fmoe -np 4 --mlock -a DeepSeek-R1:671B

setting mla=2 in llama-server will crash when request done

### Name and Version

./buildSYCL/bin/llama-cli --version
version: 3604 (ca1e00d1)
built with Intel(R) oneAPI DPC++/C++ Compiler 2025.0.4 (2025.0.4.20241205) for x86_64-unknown-linux-gnu

### What operating system are you seeing the problem on?

_No response_

### Relevant log output

```shell
INFO [            update_slots] kv cache rm [p0, end) | tid="124968309209152" timestamp=1741937308 id_slot=1 id_task=1103 p0=0
INFO [           print_timings] prompt eval time     =    1008.67 ms /    13 tokens (   77.59 ms per token,    12.89 tokens per second) | tid="124968309209152" timestamp=1741937581 id_slot=1 id_task=1103 t_prompt_processing=1008.673 n_prompt_tokens_processed=13 t_token=77.59023076923077 n_tokens_second=12.888220463916452
INFO [           print_timings] generation eval time =  272766.14 ms /   935 runs   (  291.73 ms per token,     3.43 tokens per second) | tid="124968309209152" timestamp=1741937581 id_slot=1 id_task=1103 t_token_generation=272766.143 n_decoded=935 t_token=291.7284951871658 n_tokens_second=3.427844782041003
INFO [           print_timings]           total time =  273774.82 ms | tid="124968309209152" timestamp=1741937581 id_slot=1 id_task=1103 t_prompt_processing=1008.673 t_token_generation=272766.143 t_total=273774.816
INFO [            update_slots] slot released | tid="124968309209152" timestamp=1741937581 id_slot=1 id_task=1103 n_ctx=524288 n_past=947 n_system_tokens=0 n_cache_tokens=0 truncated=false
INFO [      log_server_request] request | tid="124536729102016" timestamp=1741937581 remote_addr="10.0.0.89" remote_port=56664 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [      log_server_request] request | tid="124536720709312" timestamp=1741937581 remote_addr="10.0.0.89" remote_port=36264 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="124968309209152" timestamp=1741937582 id_slot=2 id_task=2041
INFO [            update_slots] kv cache rm [p0, end) | tid="124968309209152" timestamp=1741937582 id_slot=2 id_task=2041 p0=0
/root/code/ik_llama.cpp/ggml/src/ggml-backend.c:97: GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL") failed
/root/code/ik_llama.cpp/buildSYCL/ggml/src/libggml.so(+0x33947) [0x71a873c33947]
/root/code/ik_llama.cpp/buildSYCL/ggml/src/libggml.so(ggml_abort+0xd8) [0x71a873c338d8]
/root/code/ik_llama.cpp/buildSYCL/ggml/src/libggml.so(+0xad08c) [0x71a873cad08c]
/root/code/ik_llama.cpp/buildSYCL/ggml/src/libggml.so(ggml_gallocr_alloc_graph+0x5f9) [0x71a873cac839]
/root/code/ik_llama.cpp/buildSYCL/ggml/src/libggml.so(ggml_backend_sched_alloc_graph+0x1fc) [0x71a873cb274c]
/root/code/ik_llama.cpp/buildSYCL/src/libllama.so(llama_decode+0xf43) [0x71a874f1f453]
./buildSYCL/bin/llama-server() [0x455941]
./buildSYCL/bin/llama-server() [0x459a4c]
./buildSYCL/bin/llama-server() [0x41d43c]
/lib/x86_64-linux-gnu/libc.so.6(+0x2a3b8) [0x71a87342a3b8]
/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x8b) [0x71a87342a47b]
./buildSYCL/bin/llama-server() [0x418695]
Aborted (core dumped)
```

---

#### 💬 Conversation

👤 **orca-zhang** commented the **2025-03-15** at **05:47:38**:<br>

I found the reason. The root cause is lack of GPU memory.