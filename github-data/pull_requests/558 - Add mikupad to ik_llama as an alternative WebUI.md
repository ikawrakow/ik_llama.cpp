### 🔀 [#558](https://github.com/ikawrakow/ik_llama.cpp/pull/558) - Add mikupad to ik_llama as an alternative WebUI

| **Author** | `saood06` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-06-26 |
| **Updated** | 2025-07-13 |

---

#### Description

This PR adds [mikupad](https://github.com/lmg-anon/mikupad) (and new endpoints to `server.cpp` that mikupad uses to manage its sql database).

It can be launched with `--path ../../examples/server/public_mikupad --sql-save-file [...]` with an optional `--sqlite-zstd-ext-file [...]`. 

The path serves the index.html, but the methods the endpoint rely on are only enabled when a `sql-save-file` is passed.

The provided mikupad file has the following changes from the original:
- it is built on top of https://github.com/lmg-anon/mikupad/pull/113 which cut my initial loadtime from minutes to seconds
- streamlined code (and UI sections), removing support for other LLM endpoints and data storage models
- fixed a longstanding bug with highlight misalignment (using the fix that was mentioned in the issue discussion)
- made the sidebar and sessions sections resizable (see image below)
- add a second list of auto-grouped sessions (currently done by exact name match updated dynamically, but might add ways to configure it [hide some, add more with custom matching rules] )

This does add [sqlite_modern_cpp](https://github.com/SqliteModernCpp/sqlite_modern_cpp) as a library to common, alongside the other third party libraries this project already uses such as `nlohmann/json`, `stb_image`, `base64.hpp`.

It also supports dynamically loading [phiresky/sqlite-zstd](https://github.com/phiresky/sqlite-zstd) which for allows one to use compressed sql databases, results may vary but for me it is very useful:

size before | size after | row count
--|--|--
31.04GB | 3.40GB | 14752
8.62GB | 581.33MB | 8042
12.54 GB | 2.04 GB | 1202
30.54 GB | 5.02 GB | 6180

To-do:
- [x] Dynamically load extension
- [x] Update version endpoint with new version (needed because the table changes make it incompatible with the old version) and add features enabled array
- [x] update the html to display a useful error message (guiding them on how to pass a sql file on launch) if sql feature is not enabled
- [x] Support top-n σ sampler (untested)
- [x] Remove `selectedSessionId` from the database and have it be handled via URL fragment instead
- [x] Add export all button
- [x] Implement endpoints to create, maintain, and get config info for compression (and `VACUUM` to reduce file size).
- [ ] Finalize or Implement UI (for export all button, compression, KV cache manipulation)
- [ ] Update license (including a potential new AUTHORS file for mikupad)
- [ ] Documentation
- [ ] I think compile will fail if it can't find sqlite so fix that if that is the case
- [ ] move template selected to sampling, and make sampling have it's own saves like sessions (and available templates) do. (Make it easy to have preset profiles of templates/sampler, and also would also make it so that when you create a new session it can prefill in the prompt based on the chosen template, instead of the "miku prompt" which features the mistral template like it does now).

Potential roadmap items:
- [ ] Add a mode that creates new sessions on branching or prediction
- [ ] Remove `nextSessionId` from the database. This would allow the sessions table to have a standard `INTEGER PRIMARY KEY` as that is currently how the TEXT key is being used besides `nextSessionId` (and the now removed `selectedSessionId`). As nice as this is, I'm not sure it is worth the database migration.
- [ ] SQLite Wasm option
- [ ] Allow for slot saves to be in the database. This would allow for it to be compressed (similar to prompts there can often be a lot of redundancy between saves).
- [ ] Add a new pure black version of Monospace dark (for OLED screens).
- [ ] Add the ability to mask tokens from being processed (for use with think tokens as they are supposed to be removed once the response is finished).
- [ ] max content length should be obtained from server (based on `n_ctx`) and not from user input, and also changing or even removing the usage of that variable (or just from the UI). It is used for setting maximums for Penalty Range for some samplers (useful but could be frustrating if set wrong as knowing that is not very clear), and to truncate it seems in some situation (not useful in my view).

I am still looking for feedback even in this draft state (either on use, the code or even the Roadmap/To-do list).

An image of the new resizable sessions section (`All` group is always on top, and contains all prompts, number is how many prompts in that group ):
![image](https://github.com/user-attachments/assets/c52040cc-b0d6-4759-9250-36d7ee24157a)

---

#### 💬 Conversation

👤 **saood06** commented the **2025-06-28** at **01:46:03**:<br>

Now that I have removed the hardcoded extension loading, I do think this is in a state where it can be used by others (and potentially provide feedback), but I will still be working on completing things from the "To-do" list above until it is ready for review (and will update the post above).

---

👤 **ubergarm** commented the **2025-06-30** at **14:34:30**:<br>

Heya @saood06 I had some time this morning to kick the tires on this PR.

My high level understanding is that this PR adds new web endpoint for Mikupad as an alternative to the default built-in web interface.

I don't typically use the built-in web interface, but I did by mest to try it out. Here is my experience:

<details>

<summary>logs</summary>

```bash
# get setup
$ cd ik_llama.cpp
$ git fetch upstream
$ git checkout s6/mikupad
$ git rev-parse --short HEAD
3a634c7a

# i already had the sqllite OS level lib installed apparently:
$ pacman -Ss libsql
core/sqlite 3.50.2-1 [installed]
    A C library that implements an SQL database engine

# compile
$ cmake -B build -DGGML_CUDA=ON -DGGML_VULKAN=OFF -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_CUDA_F16=ON -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1
$ cmake --build build --config Release -j $(nproc)
```

Then I tested my usual command like so:
```bash
# run llama-server
model=/mnt/astrodata/llm/models/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-IQ4_KS.gguf
CUDA_VISIBLE_DEVICES="0" \
  ./build/bin/llama-server \
    --model "$model" \
    --alias ubergarm/Qwen3-14B-IQ4_KS \
    -fa \
    -ctk f16 -ctv f16 \
    -c 32768 \
    -ngl 99 \
    --threads 1 \
    --host 127.0.0.1 \
    --port 8080
```

When I open a browser to 127.0.0.1:8080 I get a nice looking Web UI that is simple and sleek with a just a few options for easy quick configuring:

![ik_llama-saood06-mikupad-pr558](https://github.com/user-attachments/assets/4c294d58-a60c-4eb5-ad80-d5b1dc12f6f5)


Then I added the extra arguments you mention above and run again:
```bash
# run llama-server
model=/mnt/astrodata/llm/models/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-IQ4_KS.gguf
CUDA_VISIBLE_DEVICES="0" \
  ./build/bin/llama-server \
    --model "$model" \
    --alias ubergarm/Qwen3-14B-IQ4_KS \
    -fa \
    -ctk f16 -ctv f16 \
    -c 32768 \
    -ngl 99 \
    --threads 1 \
    --host 127.0.0.1 \
    --port 8080 \
    --path ./examples/server/public_mikupad \
    --sql-save-file sqlite-save.sql
```

This time a different color background appears but seems throw an async error in the web debug console as shown in this screenshot:

![ik_llama-saood06-mikupad-pr558-test-2](https://github.com/user-attachments/assets/19dc38f3-e36c-4479-b4fa-4166fe0574ef)

The server seems to be throwing 500's so maybe I didn't go to the correct endpoint or do I need to do something else to properly access it?

```bash
NFO [                    init] initializing slots | tid="140147414781952" timestamp=1751293931 n_slots=1
INFO [                    init] new slot | tid="140147414781952" timestamp=1751293931 id_slot=0 n_ctx_slot=32768
INFO [                    main] model loaded | tid="140147414781952" timestamp=1751293931
INFO [                    main] chat template | tid="140147414781952" timestamp=1751293931 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="140147414781952" timestamp=1751293931 n_threads_http="31" port="8080" hostname="127.0.0.1"
INFO [            update_slots] all slots are idle | tid="140147414781952" timestamp=1751293931
INFO [      log_server_request] request | tid="140145881767936" timestamp=1751293939 remote_addr="127.0.0.1" remote_port=54320 status=200 method="GET" path="/" params={}
INFO [      log_server_request] request | tid="140145881767936" timestamp=1751293939 remote_addr="127.0.0.1" remote_port=54320 status=200 method="GET" path="/version" params={}
INFO [      log_server_request] request | tid="140145881767936" timestamp=1751293939 remote_addr="127.0.0.1" remote_port=54320 status=500 method="POST" path="/load" params={}
INFO [      log_server_request] request | tid="140145873375232" timestamp=1751293944 remote_addr="127.0.0.1" remote_port=54336 status=200 method="GET" path="/" params={}
INFO [      log_server_request] request | tid="140145873375232" timestamp=1751293944 remote_addr="127.0.0.1" remote_port=54336 status=200 method="GET" path="/version" params={}
INFO [      log_server_request] request | tid="140145873375232" timestamp=1751293944 remote_addr="127.0.0.1" remote_port=54336 status=500 method="POST" path="/load" params={}
INFO [      log_server_request] request | tid="140145873375232" timestamp=1751293944 remote_addr="127.0.0.1" remote_port=54336 status=404 method="GET" path="/favicon.ico" params={}
```
</details>

---

👤 **saood06** commented the **2025-06-30** at **18:30:02**:<br>

> I am interested in this.
> 
> Mikupad is _excellent_ for testing prompt formatting and sampling, with how it shows logprobs over generated tokens. It's also quite fast with big blocks of text.

Glad to hear it. I agree. I love being able to see probs for each token (and even be able to pick a replacement from the specified tokens).

If you are an existing mikupad user you may need to use the DB migration script I put in https://github.com/lmg-anon/mikupad/pull/113 if you want to migrate a whole database, migrating individual sessions via import and export should work just fine I think.

>This time a different color background appears but seems throw an async error in the web debug console as shown in this screenshot:
>...
>The server seems to be throwing 500's so maybe I didn't go to the correct endpoint or do I need to do something else to properly access it?

You are doing the correct steps, I was able to reproduce the issue of not working with a fresh sql file (so far my testing was done with backup databases with existing data). Thanks for testing, I'll let you know when it works so that you can test it again if you so choose.

---

👤 **ubergarm** commented the **2025-06-30** at **19:41:28**:<br>

> You are doing the correct steps, I was able to reproduce the issue of not working with a fresh sql file (so far my testing was done with backup databases with existing data). Thanks for testing, I'll let you know when it works so that you can test it again if you so choose.

Thanks for confirming, correct I didn't have a `.sql` file already in place but just made up that name. Happy to try again whenever u are ready!