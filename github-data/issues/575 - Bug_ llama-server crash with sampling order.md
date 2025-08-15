### 🐛 [#575](https://github.com/ikawrakow/ik_llama.cpp/issues/575) - Bug: llama-server crash with sampling order

| **Author** | `mcm007` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-07-03 |
| **Updated** | 2025-07-06 |

---

#### Description

### What happened?

The OpenAi endpoint crashes when samplers order is specified with `--samplers "min_p;temperature"` or `--sampling-seq "mt"` after [Commit 3f111ad](https://github.com/ikawrakow/ik_llama.cpp/commit/3f111ad7bbb2d4f721332f9b2b344e48b3bbf9aa) ([add dry sampler #513 ](https://github.com/ikawrakow/ik_llama.cpp/pull/513)).

Behavior observed with [aider](https://aider.chat/) but can be reproduced with curl:
```
curl -k  ik_llamacpp:8080/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer no-key" -d '{
    "model": "Qwen_Qwen3-0.6B-Q6_K.gguf",
    "messages": [
    {
        "role": "user",
        "content": "Hello!"
    }
    ]
    }'
```

Webui works correctly.

The same result with other models, fa, mla, moe.



### Name and Version

```
version: 3760 (3f111ad7)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
```

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
# Build
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j$(nproc)

# Run
llama-server --host 0.0.0.0 --port 8080 --ctx-size 4096 --verbose --model /models1/Qwen_Qwen3-0.6B-Q6_K.gguf

# Log
llama_new_context_with_model: n_ctx      = 4096
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   448.00 MiB
llama_new_context_with_model: KV self size  =  448.00 MiB, K (f16):  224.00 MiB, V (f16):  224.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     1.16 MiB
llama_new_context_with_model:        CPU compute buffer size =   300.75 MiB
llama_new_context_with_model: graph nodes  = 873
llama_new_context_with_model: graph splits = 1
INFO [                    init] initializing slots | tid="139998054885568" timestamp=1751531864 n_slots=1
INFO [                    init] new slot | tid="139998054885568" timestamp=1751531864 id_slot=0 n_ctx_slot=4096
INFO [                    main] model loaded | tid="139998054885568" timestamp=1751531864
INFO [                    main] chat template | tid="139998054885568" timestamp=1751531864 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="139998054885568" timestamp=1751531864 n_threads_http="3" port="8080" hostname="0.0.0.0"
VERB [              start_loop] new task may arrive | tid="139998054885568" timestamp=1751531864
VERB [              start_loop] update_multitasks | tid="139998054885568" timestamp=1751531864
VERB [              start_loop] callback_update_slots | tid="139998054885568" timestamp=1751531864
INFO [            update_slots] all slots are idle | tid="139998054885568" timestamp=1751531864
VERB [          kv_cache_clear] clearing KV cache | tid="139998054885568" timestamp=1751531864
VERB [              get_new_id] new task id | tid="139996550641216" timestamp=1751531864 new_id=0
VERB [     add_waiting_task_id] waiting for task id | tid="139996550641216" timestamp=1751531864 id_task=0
VERB [              start_loop] wait for new task | tid="139998054885568" timestamp=1751531864
VERB [              start_loop] new task may arrive | tid="139998054885568" timestamp=1751531864
VERB [              start_loop] callback_new_task | tid="139998054885568" timestamp=1751531864 id_task=0
INFO [     process_single_task] slot data | tid="139998054885568" timestamp=1751531864 id_task=0 n_idle_slots=1 n_processing_slots=0
VERB [     process_single_task] slot data | tid="139998054885568" timestamp=1751531864 id_task=0 n_idle_slots=1 n_processing_slots=0 slots=[{"n_ctx":4096,"n_predict":-1,"model":"/models1/Qwen_Qwen3-0.6B-Q6_K.gguf","seed":4294967295,"temperature":0.800000011920929,"dynatemp_range":0.0,"dynatemp_exponent":1.0,"top_k":40,"top_p":0.949999988079071,"min_p":0.05000000074505806,"tfs_z":1.0,"typical_p":1.0,"repeat_last_n":64,"repeat_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0,"penalty_prompt_tokens":[],"use_penalty_prompt_tokens":false,"dry_multiplier":0.0,"dry_base":1.75,"dry_allowed_length":2,"dry_penalty_last_n":4096,"dry_sequence_breakers":["\n",":","\"","*"],"mirostat":0,"mirostat_tau":5.0,"mirostat_eta":0.10000000149011612,"penalize_nl":false,"stop":[],"n_keep":0,"n_discard":0,"ignore_eos":false,"stream":true,"logit_bias":[],"n_probs":0,"min_keep":0,"grammar":"","samplers":["min_p","temperature"],"id":0,"id_task":-1,"state":0,"prompt":null,"next_token":{"has_next_token":true,"n_remain":-1,"n_decoded":0,"stopped_eos":false,"stopped_word":false,"stopped_limit":false,"stopping_word":""}}]
VERB [                    send] send new result | tid="139998054885568" timestamp=1751531864 id_task=0
VERB [                    send] queue_results.push_back | tid="139998054885568" timestamp=1751531864 id_task=0
VERB [              start_loop] update_multitasks | tid="139998054885568" timestamp=1751531864
VERB [              start_loop] callback_update_slots | tid="139998054885568" timestamp=1751531864
INFO [            update_slots] all slots are idle | tid="139998054885568" timestamp=1751531864
VERB [              start_loop] wait for new task | tid="139998054885568" timestamp=1751531864
VERB [  remove_waiting_task_id] remove waiting for task id | tid="139996550641216" timestamp=1751531864 id_task=0
INFO [      log_server_request] request | tid="139996550641216" timestamp=1751531864 remote_addr="127.0.0.1" remote_port=40444 status=200 method="GET" path="/health" params={}
VERB [      log_server_request] request | tid="139996550641216" timestamp=1751531864 request="" response="{\"status\":\"ok\",\"slots_idle\":1,\"slots_processing\":0}"
VERB [             format_chat] formatted_chat | tid="139996472276544" timestamp=1751531867 text="<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
VERB [              get_new_id] new task id | tid="139996472276544" timestamp=1751531867 new_id=1
VERB [     add_waiting_task_id] waiting for task id | tid="139996472276544" timestamp=1751531867 id_task=1
VERB [              start_loop] new task may arrive | tid="139998054885568" timestamp=1751531867
VERB [              start_loop] callback_new_task | tid="139998054885568" timestamp=1751531867 id_task=1
VERB [      get_available_slot] selected slot by lru | tid="139998054885568" timestamp=1751531867 id_slot=0 t_last=-1
INFO [   launch_slot_with_task] slot is processing task | tid="139998054885568" timestamp=1751531867 id_slot=0 id_task=1
VERB [              start_loop] update_multitasks | tid="139998054885568" timestamp=1751531867
VERB [              start_loop] callback_update_slots | tid="139998054885568" timestamp=1751531867
VERB [            update_slots] posting NEXT_RESPONSE | tid="139998054885568" timestamp=1751531867
VERB [                    post] new task id | tid="139998054885568" timestamp=1751531867 new_id=2
VERB [            update_slots] tokenizing prompt | tid="139998054885568" timestamp=1751531867 id_slot=0 id_task=1
VERB [            update_slots] prompt tokenized | tid="139998054885568" timestamp=1751531867 id_slot=0 id_task=1 n_ctx=4096 n_keep=0 n_prompt_tokens=10 prompt_tokens="<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
```

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-07-03** at **09:24:06**:<br>

Is this one example of many where it crashes, or is this the only sampler combination for which it crashes?

---

👤 **mcm007** commented the **2025-07-03** at **09:59:07**:<br>

After some tests, it seems that crashes when `dry` is not specified:

Failing:
--samplers "top_k"
--samplers "top_k;tfs_z"
--samplers "top_k;tfs_z;typical_p;top_p;min_p;temperature
--samplers "top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"
--sampling-seq "mt"

Working:
--samplers "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"
--samplers "dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"
--samplers "dry"
--samplers "dry;min_p;temperature"
--samplers "min_p;temperature;dry"
--sampling-seq "mtd"
--sampling-seq "dt"

---

👤 **ikawrakow** commented the **2025-07-03** at **12:45:33**:<br>

Thanks for the bug report. #578 should fix it.

---

👤 **mcm007** commented the **2025-07-03** at **20:17:21**:<br>

Sorry, it has the same behavior/crash 🙄 

```
version: 3785 (3e024de1)
built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
```
Please consider this a low priority, can even be ignored.
Vulkan and all the other improvements are really appreciated.

---

👤 **ikawrakow** commented the **2025-07-05** at **13:12:19**:<br>

This is strange. I tested `llama-cli` with `--sampling-seq  mt`, and it works fine after this PR.

---

👤 **mcm007** commented the **2025-07-05** at **18:17:15**:<br>

Indeed, just tested, `llama-cli` is working after this PR.

From what I see, `llama-server` is still crashing for both API endpoints `/completion` and `/v1/chat/completions`

```
curl -k ik_llamacpp:8080/completion -H "Content-Type: application/json" -d '{
  "prompt": "Once upon a time",
  "n_predict": 50
  }'
```
```
curl -k ik_llamacpp:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "Qwen_Qwen3-0.6B-Q6_K.gguf",
    "messages": [
    {
        "role": "user",
        "content": "Hello!"
    }
    ]
    }'
```

---

👤 **firecoperana** commented the **2025-07-06** at **00:54:04**:<br>

https://github.com/ikawrakow/ik_llama.cpp/pull/588 should fix the server crash

---

👤 **mcm007** commented the **2025-07-06** at **06:30:29**:<br>

It works OK, thank you both!