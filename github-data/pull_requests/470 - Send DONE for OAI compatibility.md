## 🔀 [Pull Request #470](https://github.com/ikawrakow/ik_llama.cpp/pull/470) - Send [DONE] for OAI compatibility

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | 🔀 **Merged** |
| **Source Branch** | `ik/server_send_done` |
| **Target Branch** | `main` |
| **Created** | 2025-05-29 |
| **Updated** | 2025-06-17 |
| **Merged** | 2025-06-17 |

---

## 📄 Description

See [#467](https://github.com/ikawrakow/ik_llama.cpp/issues/467)

The PR adds a command line parameter `--send-done`, which makes the server send a `data: [DONE]\n\n` message when a stop token is encountered.

---

## 💬 Conversation

👤 **cyril23** commented on **2025-06-04** at **06:37:52**

Thanks a lot! `--send-done` works perfectly on my end!

Below are my build and test steps in case they’re useful.

## 1. Build ik_llama.cpp from your branch
```
# Inside WSL 2 (Ubuntu 24 LTS) start the base Docker container:
sudo docker run --gpus all -it --rm \
  --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --net \
  host nvcr.io/nvidia/tritonserver:25.04-trtllm-python-py3

# In the container, clone your branch and build ik_llama.cpp:
cd /root && git clone -b ik/server_send_done https://github.com/ikawrakow/ik_llama.cpp
cd ik_llama.cpp/
apt-get update && apt-get install -y cmake build-essential
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
cd build/bin

# Download a test model:
wget "https://huggingface.co/unsloth/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf?download=true" -O "phi-4-Q4_K_M.gguf"
```

## 2. Start the ik_llama.cpp server without `--send-done`
```
./llama-server -m ./phi-4-Q4_K_M.gguf -c 2048 -ngl 99 -np 1 --cont-batching \
  --host 0.0.0.0 --port 8000 -fa --alias "phi-4"
```

### 2.1 Test with cURL
Send a chat completions request with streaming activated:
```
me@Computer:~$ curl --location 'http://localhost:8000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer testxxx' \
--data '{
    "model": "phi-4",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the capital of France? Make your answer as short as possible."
        }
    ],
    "stream": true
}'
data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"Paris"}}],"created":1749017398,"id":"chatcmpl-RZV2GpuTn0T4JOV2iTgDWfb21r6cxEOe","model":"phi-4","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"."}}],"created":1749017398,"id":"chatcmpl-RZV2GpuTn0T4JOV2iTgDWfb21r6cxEOe","model":"phi-4","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],"created":1749017398,"id":"chatcmpl-RZV2GpuTn0T4JOV2iTgDWfb21r6cxEOe","model":"phi-4","object":"chat.completion.chunk","usage":{"completion_tokens":3,"prompt_tokens":34,"total_tokens":37}}

me@Computer:~$
```

As expected, no `data: [DONE]` line shows up.

### 2.2 Test with inference-benchmarker (expected to fail)
We can further test that with https://github.com/huggingface/inference-benchmarker/ which [expects the streamed data to end with `[done]`](https://github.com/huggingface/inference-benchmarker/blob/687e477930b387d3c9c787d4953a266f6469f047/src/requests.rs#L165):
```
# Build once:
cd ~ && git clone https://github.com/huggingface/inference-benchmarker inference-benchmarker-current && \
cd inference-benchmarker-current && \
sudo docker build -t inference_benchmarker_latest .
export HUGGING_FACE_HUB_TOKEN=my_token_here_xxx

# Run:
sudo docker run --network host -e HF_TOKEN=$HUGGING_FACE_HUB_TOKEN \
  inference_benchmarker_latest inference-benchmarker --no-console \
  --prompt-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
  --decode-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
  --url http://localhost:8000/v1 \
  --rates 1.0 --max-vus 800 --duration 15s --warmup 15s --benchmark-kind rate \
  --model-name "phi-4" --tokenizer-name "microsoft/phi-4"
```
Output of inference-benchmarker

> Text Generation Inference Benchmark 1.1.0 (unknown)
> [2025-06-04T06:11:15Z ERROR inference_benchmarker] Error running benchmark: "Backend did not return any valid response. It is either not responding or test duration is too short."

This is exactly what we expect without `[DONE]`.

## 3. Start the ik_llama.cpp server with `--send-done`
```
./llama-server -m ./phi-4-Q4_K_M.gguf -c 2048 -ngl 99 -np 1 --cont-batching \
  --host 0.0.0.0 --port 8000 -fa --alias "phi-4" \
  --send-done
```

### 3.1 Test with cURL
```
me@Computer:~$ curl --location 'http://localhost:8000/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer testxxx' \
--data '{
    "model": "phi-4",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the capital of France? Make your answer as short as possible."
        }
    ],
    "stream": true
}'
data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"Paris"}}],"created":1749017544,"id":"chatcmpl-lhQ9OQOyhQw3Vy5MCwWzIGEyg0zTs4kk","model":"phi-4","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"."}}],"created":1749017544,"id":"chatcmpl-lhQ9OQOyhQw3Vy5MCwWzIGEyg0zTs4kk","model":"phi-4","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],"created":1749017544,"id":"chatcmpl-lhQ9OQOyhQw3Vy5MCwWzIGEyg0zTs4kk","model":"phi-4","object":"chat.completion.chunk","usage":{"completion_tokens":3,"prompt_tokens":34,"total_tokens":37}}

data: [DONE]

me@Computer:~$
```

Now we can see `[DONE]\n\n` has been received as the final chunk! 👍 

### 3.2 Test with inference-benchmarker (now succeeds!)
Run the Docker container again:
```
sudo docker run --network host -e HF_TOKEN=$HUGGING_FACE_HUB_TOKEN \
  inference_benchmarker_latest inference-benchmarker --no-console \
  --prompt-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
  --decode-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
  --url http://localhost:8000/v1 \
  --rates 1.0 --max-vus 800 --duration 15s --warmup 15s --benchmark-kind rate \
  --model-name "phi-4" --tokenizer-name "microsoft/phi-4"
```
Output of inference-benchmarker:
```
┌─────────────────┬────────────────────────────────────────────────────────────────┐
│ Parameter       │ Value                                                          │
├─────────────────┼────────────────────────────────────────────────────────────────┤
│ Max VUs         │ 800                                                            │
│ Duration        │ 15                                                             │
│ Warmup Duration │ 15                                                             │
│ Benchmark Kind  │ Rate                                                           │
│ Rates           │ [1.0]                                                          │
│ Num Rates       │ 10                                                             │
│ Prompt Options  │ num_tokens=Some(200),min_tokens=180,max_tokens=220,variance=10 │
│ Decode Options  │ num_tokens=Some(200),min_tokens=180,max_tokens=220,variance=10 │
│ Tokenizer       │ microsoft/phi-4                                                │
│ Extra Metadata  │ N/A                                                            │
└─────────────────┴────────────────────────────────────────────────────────────────┘


┌────────────────────┬────────────┬───────────────────┬────────────┬───────────┬───────────────────┬────────────┬─────────────────────┬─────────────────────────────┬──────────────────────────────┐
│ Benchmark          │ QPS        │ E2E Latency (avg) │ TTFT (avg) │ ITL (avg) │ Throughput        │ Error Rate │ Successful Requests │ Prompt tokens per req (avg) │ Decoded tokens per req (avg) │
├────────────────────┼────────────┼───────────────────┼────────────┼───────────┼───────────────────┼────────────┼─────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ warmup             │ 0.72 req/s │ 1.40 sec          │ 65.69 ms   │ 7.44 ms   │ 127.13 tokens/sec │ 0.00%      │ 10/10               │ 200.00                      │ 177.40                       │
│ constant@1.00req/s │ 0.71 req/s │ 3.42 sec          │ 2040.85 ms │ 7.52 ms   │ 129.97 tokens/sec │ 0.00%      │ 9/9                 │ 200.00                      │ 183.44                       │
└────────────────────┴────────────┴───────────────────┴────────────┴───────────┴───────────────────┴────────────┴─────────────────────┴─────────────────────────────┴──────────────────────────────┘
```

Everything finishes without errors.  👍

---

👤 **voipmonitor** commented on **2025-06-17** at **07:02:07**

I have verified it and it works for me too with the --send-done. Would be nice to merge it.

---

👤 **ikawrakow** commented on **2025-06-17** at **07:33:28**

Closes [#467](https://github.com/ikawrakow/ik_llama.cpp/issues/467)