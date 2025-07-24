### ✨ [#627](https://github.com/ikawrakow/ik_llama.cpp/issues/627) - Feature Request: Tensor Parallelism

| **Author** | `rankaiyx` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-18 |
| **Updated** | 2025-07-19 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Tensor Parallelism is a model-parallelism technique used in Large Language Model (LLM) inference to distribute the model's tensor computations (e.g., matrix multiplications) across multiple devices (like GPUs or TPUs). This allows different parts of the model's layers to be processed in parallel, improving inference speed and scalability.

**Key Features:**

- **Model Splitting:** Splits model layers (especially large weight matrices) across multiple devices.
- **Distributed Computation:** Performs tensor operations in parallel, reducing computation time.
- **Communication Overhead:** Requires inter-device communication (e.g., using AllReduce) to synchronize results.
- **Efficient Scaling:** Enables inference on larger models that don't fit on a single device.

**Use Case:** Ideal for large-scale LLM inference where model size exceeds a single GPU's memory capacity.

### Motivation

The performance of current methods(--split-mode  row) is much worse than vllm or mlc-llm.

On the 4xP100 platform, using the vLLM or mlc-llm for inference with the Qwen2.5-72B-4bit model achieves a generation speed of approximately 20 tok/s. In contrast, when using the llama.cpp with "--split-mode  row", the generation speed only reaches 10 tok/s, which is merely 50% of the former speed.

mlc-llm development is less active and supports fewer models.
In the upcoming 1.0 version, vllm will abandon a large number of Turing and older hardware.

### Possible Implementation

_No response_

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-07-18** at **08:05:35**:<br>

Have you tried raising the issue with the `llama.cpp` project?

Support for old hardware is not one of the strengths of this project, while exactly this is one of the strengths of mainline `llama.cpp`.

---

👤 **rankaiyx** commented the **2025-07-18** at **08:21:28**:<br>

There is an issue.
But it's expired.
Maybe the mainline llama.cpp focuses on versatility rather than SOTA.
https://github.com/ggml-org/llama.cpp/issues/9086

---

👤 **Ph0rk0z** commented the **2025-07-19** at **10:12:01**:<br>

Originally Cuda Dev was supposed to work on backend agnostic TP. Someone else volunteered and made partial PRs but appears to have abandoned them. Progress is stalled.

My split mode row gives higher T/G but lower PP as of this month in mainline. Since last year, some progress has been made. I tested with command-A. Wanted to compare with IK but then realized command-A isn't supported.

What's interesting is fastllm, who claims to fully utilize numa and supports hybrid inference. I aim to try out qwen-235b and compare speeds at some point. Can use both at 4 bit.

---

👤 **saood06** commented the **2025-07-19** at **10:19:14**:<br>

>Wanted to compare with IK but then realized command-A isn't supported.

I thought it was from #341

---

👤 **Ph0rk0z** commented the **2025-07-19** at **15:01:49**:<br>

Damn.. I missed that. Will give it a go.

_Well.. this is dildos.._

IK: Same prompt processing speed as mainline. In SM row, 17t/s generation. Loads GPU 0 like mainline used to. Unfortunately, command-A outputs what looks like parts of the training data or random text. Without SM it is coherent but only does ~12T/s

Mainline: I unfortunately pulled today. My speed in parallel is only 12t/s. Without it, drops down to 9. Prompt processing for both backends is about half speed when SM is row.

---

👤 **Ph0rk0z** commented the **2025-07-19** at **15:01:49**:<br>

Damn.. I missed that. Will give it a go.

---

👤 **saood06** commented the **2025-07-20** at **01:09:42**:<br>

> IK: Same prompt processing speed as mainline. In SM row, 17t/s generation. Loads GPU 0 like mainline used to. Unfortunately, command-A outputs what looks like parts of the training data or random text. Without SM it is coherent but only does ~12T/s
> 
> Mainline: I unfortunately pulled today. My speed in parallel is only 12t/s. Without it, drops down to 9. Prompt processing for both backends is about half speed when SM is row.

So it looks like this repo gives you the fastest usable generation, I suggest you file an issue for the coherency issues with row enabled (and maybe also for the PP speed dropping by half).