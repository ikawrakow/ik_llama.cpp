### 🔀 [#168](https://github.com/ikawrakow/ik_llama.cpp/pull/168) - Falcon3 changes

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-01-10 |
| **Updated** | 2025-01-10 |

---

#### Description

Two changes:
* Add pre-tokenizer for `Falcon3` (same as `llama3`)
* Use integer arithmetic to perform the summation of a row of activations for `Q8_K16`

The second change is required for the `IQ2_BN_R4` 4-row interleaved variant. The existing implementation just sums up the `f32` values. This is fine with the original BitNet models and also with the TriLM ternary models, but with the Falcon3 ternary models I observe too large of a difference between the GPU and the CPU perplexity result. With this change the difference is greatly reduced and `IQ2_BN_R4` actually arrives at a slightly lower PPL than Microsoft's BitNet implementation (which is claimed to be "losless").

---

#### 💬 Conversation

👤 **ikawrakow** commented the **2025-01-10** at **12:56:49**:<br>

Oh, here some performance figures for `IQ2_BN` and Microsoft's [Bitnet](https://github.com/microsoft/BitNet) `I2_S` quants, which claim to be the fastest CPU implementation of ternary transformer models. Tests run on a Ryzen-7950X CPU. 

After following the Bitnet instructions:
```
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp
pip install -r requirements.txt
python setup_env.py --hf-repo tiiuae/Falcon3-7B-Instruct-1.58bit -q i2_s
```
I'm finding that their `e2e_benchmark.py` Python script is not really working. Or, more precisely, it is working but giving a dismal performance. With
```
python3 utils/e2e_benchmark.py -m models/Falcon3-7B-Instruct-1.58bit/ggml-model-i2_s.gguf -n 0 -p 512 -t 16
```
I get this:
| model                          |       size |     params | backend    | threads | n_batch |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------: | ------------: | -------------------: |
| llama 3B I2_S - 2 bpw ternary  |   3.05 GiB |     7.46 B | CPU        |      16 |       1 |         pp512 |         22.15 ± 0.07 |

Hahaha. 22 t/s for PP-512? Fortunately for us, BitNet is just a thin wrapper around `llama.cpp`, so we can run the `llama-bench` tool, which the  `e2e_benchmark.py ` uses under the hood, directly:
```
./build/bin/llama-bench -m models/Falcon3-7B-Instruct-1.58bit/ggml-model-i2_s.gguf -p 512 -n 128 -t 16
```
and we get

| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 3B I2_S - 2 bpw ternary  |   3.05 GiB |     7.46 B | CPU        |      16 |         pp512 |        187.90 ± 0.99 |
| llama 3B I2_S - 2 bpw ternary  |   3.05 GiB |     7.46 B | CPU        |       8 |         tg128 |         23.39 ± 0.05 |

In comparison, here is what we get with `IQ2_BN` (using `-rtr 1` to interleave 4 rows when loading the model:
| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| llama ?B IQ2_BN - 2.00 bpw Bitnet |   2.07 GiB |     7.46 B | CPU        |      16 |         pp512 |    465.85 ± 1.91 |
| llama ?B IQ2_BN - 2.00 bpw Bitnet |   2.07 GiB |     7.46 B | CPU        |       8 |         tg128 |     28.03 ± 0.04 |

So, 2.5X for PP-512, and ~20% better for TG-128 (both achieve maximum performance at 8 threads). TG-128 is of course memory bound and the BitNet authors make claims about energy efficiency, so let's look at TG with fewer threads:

| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 3B I2_S - 2 bpw ternary  |   3.05 GiB |     7.46 B | CPU        |       1 |         tg128 |          9.64 ± 0.05 |
| llama 3B I2_S - 2 bpw ternary  |   3.05 GiB |     7.46 B | CPU        |       2 |         tg128 |         15.45 ± 0.04 |
| llama 3B I2_S - 2 bpw ternary  |   3.05 GiB |     7.46 B | CPU        |       4 |         tg128 |         22.21 ± 0.20 |

vs

| model                          |       size |     params | backend    | threads |          test |              t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | ---------------: |
| llama ?B IQ2_BN - 2.00 bpw Bitnet |   2.07 GiB |     7.46 B | CPU        |       1 |         tg128 |     12.83 ± 0.24 |
| llama ?B IQ2_BN - 2.00 bpw Bitnet |   2.07 GiB |     7.46 B | CPU        |       2 |         tg128 |     22.46 ± 0.03 |
| llama ?B IQ2_BN - 2.00 bpw Bitnet |   2.07 GiB |     7.46 B | CPU        |       4 |         tg128 |     27.62 ± 0.05 |

OK. Now I can claim that `IQ2_BN` is almost 4X more energy efficient than BitNet as we get (almost) the same performance at 2 threads as their maximum performance at 8 threads.