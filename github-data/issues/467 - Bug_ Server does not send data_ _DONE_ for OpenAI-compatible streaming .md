### 🐛 [#467](https://github.com/ikawrakow/ik_llama.cpp/issues/467) - Bug: Server does not send data: [DONE] for OpenAI-compatible streaming endpoint `/v1/chat/completions`

| **Author** | `cyril23` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-05-28 |
| **Updated** | 2025-06-17 |

---

#### Description

### Description

When using the `/v1/chat/completions` endpoint with `stream: true`, the `ikawrakow/ik_llama.cpp` server does not send the standard `data: [DONE]\n\n` message to terminate the Server-Sent Event stream. This causes issues with clients that strictly adhere to the OpenAI API specification, such as the https://github.com/huggingface/inference-benchmarker/ tool, which reports errors like "Connection closed before completion.", see https://github.com/huggingface/inference-benchmarker/blob/687e477930b387d3c9c787d4953a266f6469f047/src/requests.rs#L165

While clients like `curl` might be more lenient and work by detecting the natural end of the stream, tools designed for benchmarking OpenAI-compatible endpoints rely on this `[DONE]` message for proper stream accounting and termination.

This behavior was confirmed by running `huggingface/inference-benchmarker` against `ikawrakow/ik_llama.cpp` (which failed consistently) and then successfully against the https://github.com/ggml-org/llama.cpp server (which implements the `[DONE]` message, see https://github.com/ggml-org/llama.cpp/blob/26b79b6cb3e7840ff15729350e95907e19f9f480/tools/server/server.cpp#L4309).

### Steps to Reproduce with `curl`

1.  Start the `ikawrakow/ik_llama.cpp` server with any model (e.g. https://huggingface.co/unsloth/phi-4-GGUF/blob/main/phi-4-Q4_K_M.gguf in my case).
2.  Execute the following `curl` command:
    ```bash
    curl -i -N -X POST "http://localhost:8000/v1/chat/completions" \
         -H "Content-Type: application/json" \
         -d '{
               "model": "phi-4",
               "messages": [{"role": "user", "content": "Tell me a short story."}],
               "max_tokens": 50,
               "stream": true
             }'
    ```

### Observed Behavior (with `ikawrakow/ik_llama.cpp`)

The stream provides `data: {...}` events correctly but ends without a final `data: [DONE]\n\n` message.

Example snippet of `curl` output from `ikawrakow/ik_llama.cpp` (full stream ends after the last JSON data chunk):
```
HTTP/1.1 200 OK
Access-Control-Allow-Origin:
Content-Type: text/event-stream
Keep-Alive: timeout=5, max=5
Server: llama.cpp
Transfer-Encoding: chunked

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":"Once"}}],"created":1748421931,"id":"chatcmpl-wgqtIZhAKHJRCj568kAdGfhyDUIj69kZ","model":"phi-4","object":"chat.completion.chunk"}

... 

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":","}}],"created":1748421931,"id":"chatcmpl-wgqtIZhAKHJRCj568kAdGfhyDUIj69kZ","model":"phi-4","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":"length","index":0,"delta":{}}],"created":1748421931,"id":"chatcmpl-wgqtIZhAKHJRCj568kAdGfhyDUIj69kZ","model":"phi-4","object":"chat.completion.chunk","usage":{"completion_tokens":50,"prompt_tokens":14,"total_tokens":64}}
```

### Expected Behavior (and behavior of https://github.com/ggml-org/llama.cpp)

The stream should terminate with a `data: [DONE]\n\n` message after the last data chunk.

Example snippet of `curl` output from https://github.com/ggml-org/llama.cpp for the same request:
```
HTTP/1.1 200 OK
Keep-Alive: timeout=5, max=100
Content-Type: text/event-stream
Server: llama.cpp
Transfer-Encoding: chunked
Access-Control-Allow-Origin:

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"role":"assistant","content":null}}],"created":1748422234,"id":"chatcmpl-51VeqNldSlrUKqMP1Seka7KfXksFbSea","model":"phi-4","system_fingerprint":"b5517-1e8659e6","object":"chat.completion.chunk"}

...

data: {"choices":[{"finish_reason":null,"index":0,"delta":{"content":" climb"}}],"created":1748422235,"id":"chatcmpl-51VeqNldSlrUKqMP1Seka7KfXksFbSea","model":"phi-4","system_fingerprint":"b5517-1e8659e6","object":"chat.completion.chunk"}

data: {"choices":[{"finish_reason":"length","index":0,"delta":{}}],"created":1748422235,"id":"chatcmpl-51VeqNldSlrUKqMP1Seka7KfXksFbSea","model":"phi-4","system_fingerprint":"b5517-1e8659e6","object":"chat.completion.chunk","usage":{"completion_tokens":50,"prompt_tokens":13,"total_tokens":63},"timings":{"prompt_n":13,"prompt_ms":239.618,"prompt_per_token_ms":18.432153846153845,"prompt_per_second":54.25301938919447,"predicted_n":50,"predicted_ms":680.938,"predicted_per_token_ms":13.61876,"predicted_per_second":73.42812414639806}}

data: [DONE]
```

### OpenAI API Documentation Reference

The OpenAI API documentation specifies this termination message for `/v1/completions` (legacy) https://platform.openai.com/docs/api-reference/completions/create#completions-create-stream: "tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a `data: [DONE]` message."

The newer `/v1/chat/completions` spec https://platform.openai.com/docs/api-reference/chat/create (and https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream) does not define the `data: [DONE]` message anymore as far as I can see.

Nevertheless client implementations like https://github.com/huggingface/inference-benchmarker and servers like https://github.com/ggml-org/llama.cpp still expect this for chat completions.

### Discussion and Potential Path Forward

It appears there might be differing interpretations or evolving practices regarding the termination of SSE streams for the `/v1/chat/completions` endpoint. While the  https://github.com/ggml-org/llama.cpp server and tools like `huggingface/inference-benchmarker` operate with the expectation of a `data: [DONE]\n\n` message, `ikawrakow/ik_llama.cpp` currently does not send this, which aligns with a stricter reading of the newer chat completions documentation that omits its explicit mention.

This difference leads to the observed compatibility issues with certain client libraries that were likely built with the original streaming behavior (or the legacy `/v1/completions` behavior) in mind, or that test for it as a general sign of OpenAI compatibility.

To enhance compatibility with a wider range of client tools, including those used for benchmarking like `huggingface/inference-benchmarker`, it might be beneficial for `ikawrakow/ik_llama.cpp` to offer a way to include the `data: [DONE]\n\n` terminator.

### Suggestion

Would it be feasible to introduce an optional server startup flag (e.g., `--openai-strict-stream-end` or `--send-done-event`) that, when enabled, would cause the server to append `data: [DONE]\n\n` to the end of SSE streams for OpenAI-compatible endpoints like `/v1/chat/completions`?

This would allow users who need to interface with clients expecting this specific terminator to do so, while the default behavior could remain as it is if that's preferred or deemed more aligned with the latest interpretation of the chat completions streaming protocol.

This approach could provide flexibility and broader compatibility without necessarily changing the default server behavior if the current implementation is intentional based on the newer spec.

Thank you for your great work on this project and for considering this feedback. 

### Name and Version

```
~/ik_llama.cpp/build/bin# ./llama-cli --version
version: 3715 (09764678)
built with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
```


### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell

```

---

#### 💬 Conversation

👤 **cyril23** commented the **2025-05-28** at **10:58:55**:<br>

(with the help of AI ..) I've made a direct modification to the `handle_chat_completions` function in `examples/server/server.cpp` to force the server to send `data: [DONE]\n\n` at the end of a successful stream.

Disclaimer: This is a minimal, proof-of-concept change intended only to demonstrate the effect of sending the `[DONE]` event and to test compatibility with clients like huggingface/inference-benchmarker. It is not a production-ready solution and doesn't include any configurability (like a command-line flag).

Here is the patch against the current main branch commit [`09764678456f8991f6095118f3727d9d0b17b8c8`](https://github.com/ikawrakow/ik_llama.cpp/commit/09764678456f8991f6095118f3727d9d0b17b8c8):
```diff
diff --git a/examples/server/server.cpp b/examples/server/server.cpp
index 360f571e..c5465846 100644
--- a/examples/server/server.cpp
+++ b/examples/server/server.cpp
@@ -3149,6 +3149,7 @@ int main(int argc, char ** argv) {
             ctx_server.queue_results.remove_waiting_task_id(id_task);
         } else {
             const auto chunked_content_provider = [id_task, &ctx_server, completion_id](size_t, httplib::DataSink & sink) {
+                bool successful_completion = false;
                 while (true) {
                     server_task_result result = ctx_server.queue_results.recv(id_task);
                     if (!result.error) {
@@ -3168,6 +3169,7 @@ int main(int argc, char ** argv) {
                             }
                         }
                         if (result.stop) {
+                            successful_completion = true;
                             break;
                         }
                     } else {
@@ -3183,6 +3185,15 @@ int main(int argc, char ** argv) {
                         break;
                     }
                 }
+                if (successful_completion) {
+                    static const std::string done_message = "data: [DONE]\n\n";
+                    LOG_VERBOSE("data stream", {{"to_send", done_message}});
+                    if (!sink.write(done_message.c_str(), done_message.size())) {
+                        // If writing [DONE] fails, the stream is likely already problematic.
+                        ctx_server.queue_results.remove_waiting_task_id(id_task);
+                        return false; // Signal error to httplib
+                    }
+                }
                 sink.done();
                 ctx_server.queue_results.remove_waiting_task_id(id_task);
                 return true;
```

---

👤 **ikawrakow** commented the **2025-05-28** at **11:30:14**:<br>

@cyril23 

I can try to make a proper PR, but I'm old school and never use such fancy stuff. Are you willing to test?

---

👤 **cyril23** commented the **2025-05-28** at **13:56:52**:<br>

> I can try to make a proper PR, but I'm old school and never use such fancy stuff. Are you willing to test?

Sure, I'll test it

---

👤 **ikawrakow** commented the **2025-05-31** at **05:33:17**:<br>

PR #470 is waiting to be tested.

---

👤 **cyril23** commented the **2025-06-04** at **06:40:24**:<br>

> PR [#470](https://github.com/ikawrakow/ik_llama.cpp/pull/470) is waiting to be tested.

I've tested it successfully in https://github.com/ikawrakow/ik_llama.cpp/pull/470#issuecomment-2938782085, but I'm the wrong guy to review the code

---

👤 **voipmonitor** commented the **2025-06-17** at **07:03:08**:<br>

I have tested it too and it works.

---

👤 **ikawrakow** commented the **2025-06-17** at **07:34:12**:<br>

Closed via #470