### 🐛 [#325](https://github.com/ikawrakow/ik_llama.cpp/pull/325) - Fix KLD precision

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-12 |
| **Updated** | 2025-04-13 |

---

#### Description

Some people insist that perplexity tells us nothing, and that [Kullback-Leibler Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) (KLD), along with the other statistics computed by `llama-perplexity` with the `--kl-divergence` option, are the one and only one true measure of quantization accuracy. Computing KLD requires 1st running the `llama-perplexity` tool with `--kl-divergence-base` to compute the logits of the base model, which are then used to compute KLD and other token probability statistics in a subsequent run with a quantized (or otherwise approximate) model. The base model logits file is quite large as it stores the log-probabilities for each evaluated token for all tokens in the vocabulary. Hence, when I added KLD capabilities to `llama.cpp` with [this](https://github.com/ggml-org/llama.cpp/pull/5076) and [this](https://github.com/ggml-org/llama.cpp/pull/5081) PRs, I used 16-bit precision to store the logits of the base model, setting the minimum logit to `std::max(min_logit, max_logit - 16). That was adequate for the models available at the time. 

As I'm notoriously short on disk space, I don't keep the large base logits file around. Hence, I find it a hassle to use KLD to evaluate quantization accuracy of some new technique, so I basically never use the `--kl-divergence` option in the `llama-perplexity` tool. But the other day I saw [this post](https://huggingface.co/blog/bartowski/llama4-scout-off) using the statistics produced by `llama-perplexity --kl-divergence` to compare several quantizations of LlaMA-4-Scout, and as I was experimenting with quantization of that model, I decided to run some KLD calculations myself. Hahaha! In all of this time, nobody noticed that my 16-bit approximation for the stored base model logits is not adequate. More specifically,  with much increased vocabulary size, the `std::max(min_logit, max_logit - 16)` lower bound of the log-probabilities stored in the file is too high. The effect is that the perplexity of the base model computed from the stored logits is different from the perplexity computed directly from the float probabilities. I was concerned that other statistics will be influenced as well, but it looks like it is only PPL that becomes wrong.

A lot of talk for this one-liner PR, which fixes the problem.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-04-13** at **15:20:53**:<br>

> I was concerned that other statistics will be influenced as well, but it looks like it is only PPL that becomes wrong.

Just a quick test, fwiw, the PPL computed by `llama-imatrix` before and after this PR seem to give basically the same result for `V3-0324` `q8_0` GGUF. 

* `ik_llama.cpp@2089147a` `Final estimate: PPL = 3.4755 +/- 0.03305`
* `ik_llama.cpp including c01449a` `Final estimate: PPL = 3.4727 +/- 0.03300`

If I understand this PR correctly, I should expect the PPL computed with this PR to be different than the PPL computed without it specifically for models with much increased vocabulary size (e.g. LlaMA-4-Scout)?

Thanks!

---

👤 **ikawrakow** commented the **2025-04-13** at **15:35:00**:<br>

The PR does not affect `imatrix`. It affects `llama-perplexity` when run with `--kl-divergence-base X --kl-divergence`. This computes KL-Divergence and various other token probability statistics between the current model and the token probabilities for the base model stored in `X` and computed in a previous run of `llama-perplexity`.

---

👤 **ikawrakow** commented the **2025-04-13** at **15:38:16**:<br>

Also, I don't know how it affects other models. But for LLaMA-4-Scout I observed a nearly 1% difference without this PR.