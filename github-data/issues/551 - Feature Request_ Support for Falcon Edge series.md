### ✨ [#551](https://github.com/ikawrakow/ik_llama.cpp/issues/551) - Feature Request: Support for Falcon Edge series

| **Author** | `harborwater` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-24 |
| **Updated** | 2025-06-26 |

---

#### Description

### Prerequisites

- [x] I am running the latest code. Mention the version if possible as well.
- [x] I carefully followed the [README.md](https://github.com/ggerganov/llama.cpp/blob/master/README.md).
- [x] I searched using keywords relevant to my issue to make sure that I am creating a new issue that is not already open (or closed).
- [x] I reviewed the [Discussions](https://github.com/ggerganov/llama.cpp/discussions), and have a new and useful enhancement to share.

### Feature Description

Falcon Edge series: https://huggingface.co/collections/tiiuae/falcon-edge-series-6804fd13344d6d8a8fa71130

The Falcon Edge series as released around the same time as Microsoft's [bitnet](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) model.


### Motivation

I think for people who need a small, speedy, and performant model in a resource constrained environment this would make a great addition. 

### Possible Implementation

_No response_

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-06-24** at **12:22:12**:<br>

Is it supported in mainline `llama.cpp`?

---

👤 **saood06** commented the **2025-06-24** at **16:48:14**:<br>

> Is it supported in mainline `llama.cpp`?

It seems like support exists in `bitnet.cpp` (which is even better considering they forked around when we did). 

This is their submodule update for it: https://github.com/Eddie-Wang1120/llama.cpp/compare/5eb47b72106e3b35f10e8befa616a9241242b226...40ed0f290203a9a78540b8f7eb18bd828043fe21. 
This is the PR adding support containing that submodule update and the convert python code: https://github.com/microsoft/BitNet/pull/268/files#diff-f90cdc9c8f0e8eefed785548f9fac0bd8868cf4430e259cef59b5833ca299c4c.

Support seems rather easy to add.

>I think for people who need a small, speedy, and performant model in a resource constrained environment this would make a great addition.

I read the blogpost, and I agree. They trained on less tokens (1.5 T vs 4T) but they still ended up with strong models for their size, even compared to `Bitnet-b1.58-2B-4T`.

---

👤 **ikawrakow** commented the **2025-06-24** at **17:05:32**:<br>

In that case it should (almost) work:
```
huggingface-cli download --local-dir falcon tiiuae/Falcon-E-3B-Instruct-GGUF
./bin/llama-quantize --allow-requantize falcon/ggml-model-i2_s.gguf test.gguf iq2_bn
./bin/llama-cli -m test.gguf -c 8192 -s 5678 -n 128 -p "I believe the meaning of life is" -t 16
```
The last command fails with 
```
llama_model_load: error loading model: error loading model vocabulary: unknown pre-tokenizer type: 'falcon_e'
```
So, I guess, it is a matter of adding this `falcon-e` pre-tokenizer? Or are there differences in the architecture?

---

👤 **saood06** commented the **2025-06-24** at **17:12:59**:<br>

> So, I guess, it is a matter of adding this `falcon-e` pre-tokenizer? 

Yep, the linked code shows just that and adding a template. Like I said support seems rather easy.

>Or are there differences in the architecture?

None that require change it seems. Their blogpost says:

>We adopted the architecture outlined in the paper [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764), but made a key modification by eliminating the Layer Normalization layers within the BitNet layers. However, we retained the original pre-attention and pre-MLP layer norms to ensure compatibility with the Llama architecture, allowing seamless integration from the outset. Interestingly, we discovered that removing these Layer Normalization layers had no adverse effect on model performance, while also ensuring compatibility with the broader ecosystem with minimal adjustments.

---

👤 **ikawrakow** commented the **2025-06-24** at **17:13:41**:<br>

Well, pretending that `falcon_e` is the same as `falcon3`, it appears to work:
```
./bin/llama-cli -m test.gguf -c 8192 -s 5678 -n 128 -p "I believe the meaning of life is" -t 16

I believe the meaning of life is to create a legacy and have a positive impact on the world, and that the purpose of life is to learn from experiences and grow as a person. I believe that human existence is a journey of self-discovery and exploration, and that the ultimate goal of life is to find meaning and purpose in the experiences and challenges that we face. 

I hope that you will take my words to heart and consider the impact that I have had on your life, and that you will continue to learn and grow as a person. I believe that life is a gift, and that it is up to each of us to use our experiences and challenges to our
llama_print_timings:        load time =     169.38 ms
llama_print_timings:      sample time =       2.29 ms /   128 runs   (    0.02 ms per token, 56017.51 tokens per second)
llama_print_timings: prompt eval time =      17.40 ms /     8 tokens (    2.17 ms per token,   459.80 tokens per second)
llama_print_timings:        eval time =    1942.39 ms /   127 runs   (   15.29 ms per token,    65.38 tokens per second)
llama_print_timings:       total time =    1968.25 ms /   135 tokens
Log end
```

Perplexity seems reasonable too:
```
./bin/llama-perplexity -m test.gguf -f ../tests/wiki.test.raw -t 16 -b 512

perplexity: tokenizing the input ..
perplexity: tokenization took 197.131 ms
perplexity: calculating perplexity over 713 chunks, n_ctx=512, batch_size=512, n_seq=1
perplexity: 0.81 seconds per pass - ETA 9.58 minutes
[1]8.1305,[2]8.7254,[3]9.4583,[4]9.5025,[5]8.9857,[6]9.0719,[7]9.5533,[8]9.9153,[9]10.0344,[10]10.1572,[11]10.2116,[12]10.3118,[13]10.3012,[14]10.2507,[15]10.2737,[16]10.3008,[17]10.4085,[18]10.4099,[19]10.1711,[20]10.2990,[21]9.8262,[22]9.8252,[23]10.0332,[24]10.0470,[25]10.0355,[26]9.7866,[27]9.8367,[28]9.6278,[29]9.5681,[30]9.3539,[31]9.3138,[32]9.2042,[33]8.8973,[34]8.7937,[35]8.8279,[36]8.7234,[37]8.7861,[38]8.7650,[39]8.7465,[40]8.6341,[41]8.5701,[42]8.6277,[43]8.6532,[44]8.7307,[45]8.8189,[46]8.8520,[47]8.7498,[48]8.7996,[49]8.7895,[50]8.7798,[51]8.7480,[52]8.7659,[53]8.7650,[54]8.7096,[55]8.7332,[56]8.6647,[57]8.7387,[58]8.7666,[59]8.7425,[60]8.6935,[61]8.7227,[62]8.7171,^C
```

---

👤 **ikawrakow** commented the **2025-06-24** at **17:16:23**:<br>

This is the diff that makes it work:
```
diff --git a/src/llama.cpp b/src/llama.cpp
index a70d2582..de91e687 100644
--- a/src/llama.cpp
+++ b/src/llama.cpp
@@ -6192,7 +6192,8 @@ static void llm_load_vocab(
                     tokenizer_pre == "llama3"   ||
                     tokenizer_pre == "llama-v3" ||
                     tokenizer_pre == "llama-bpe"||
-                    tokenizer_pre == "falcon3") {
+                    tokenizer_pre == "falcon3"  ||
+                    tokenizer_pre == "falcon_e") {
                 vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_LLAMA3;
                 vocab.tokenizer_ignore_merges = true;
                 vocab.tokenizer_add_bos = true;
```

---

👤 **ikawrakow** commented the **2025-06-25** at **07:21:17**:<br>

See #555 and let me know of it works.