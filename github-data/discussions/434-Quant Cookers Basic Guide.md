### 🗣️ [#434](https://github.com/ikawrakow/ik_llama.cpp/discussions/434) - Quant Cookers Basic Guide

| **Author** | `ubergarm` |
| :--- | :--- |
| **Created** | 2025-05-18 |
| **Updated** | 2025-05-21 |

---

#### Description

Quant Cooking Basic Guide
===
Example workflow for cooking custom quants with ik_llama.cpp that I used to generate [ubergarm/Qwen3-14B-GGUF](https://huggingface.co/ubergarm/Qwen3-14B-GGUF).

## Goal
The goal is to provide a specific example of methodology that can be adapted for future LLMs and quant types in general.

In this guide we will download and quantize the dense model [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) on a gaming rig with a single 3090TI FE 24GB VRAM GPU.

We will use the latest [ik_llama.cpp quants](https://github.com/ikawrakow/ik_llama.cpp/pull/422) to target running this 14B model in GGUF format fully offloaded on <=16GB VRAM systems with 32k context.

This guide does *not* get into more complex things like MLA methodology e.g. converting fp8 to bf16 on older GPU hardware.

## Dependencies
This is all run on a Linux rig, but feel free to use WSL for a similar experience if you're limited to a windows based OS.

Install any build essentials, git, etc. We will use `uv` for python virtual environment management to keep everything clean.

```bash
# Setup folder to do your work and hold the models etc
mkdir /mnt/llms
cd /mnt/llms

# Install uv and python packages
# https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ./venv --python 3.12 --python-preference=only-managed
source ./venv/bin/activate
uv pip install huggingface_hub[hf-xet]

# Start downloading the bf16 safetensors from huggingface
mkdir -p Qwen/Qwen3-14B
cd Qwen/Qwen3-14B
huggingface-cli download --local-dir ./ Qwen/Qwen3-14B

# Make a target directory to hold your finished quants for uploading to huggingface
mkdir -p ubergarm/Qwen3-14B-GGUF # use your name obviously

# Install mainline or evshiron llama.cpp forks just for the python scripts.
cd /mnt/llms
git clone git@github.com:ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1
cmake --build build --config Release -j $(nproc)

# Install and build ik_llama.cpp for the heavy lifting and SOTA GGUF quants.
cd /mnt/llms
git clone git@github.com:ikawrakow/ik_llama.cpp.git
cd ik_llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1
cmake --build build --config Release -j $(nproc)

# Download your imatrix corpus and wiki.test.raw test corpus.
wget https://gist.githubusercontent.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/raw/571fda718462de863e5a0171078c175420c7649a/calibration_data_v5_rc.txt

wget https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/main/wiki.test.raw.gz
gunzip wiki.test.raw.gz

# Okay, now your folders should look something like this, and you are ready to begin cooking!
cd /mnt/llms
tree

.
├── venv
├── ik_llama.cpp
├── llama.cpp
├── Qwen
│  └── Qwen3-14B
└── ubergarm
   └── Qwen3-14B-GGUF
```

## Convert bf16 safetensors to bf16 gguf
I generally use mainline llama.cpp or evshiron's fork for doing conversion with python script.
```bash
# This took less than 12GiB RAM and about 30 seconds
cd /mnt/llms
uv pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt --prerelease=allow --index-strategy unsafe-best-match

python \
    llama.cpp/convert_hf_to_gguf.py \
    --outtype bf16 \
    --split-max-size 50G \
    --outfile ./ubergarm/Qwen3-14B-GGUF/ \
    ./Qwen/Qwen3-14B/

du -hc ./ubergarm/Qwen3-14B-GGUF/*.gguf
28G ./ubergarm/Qwen3-14B-GGUF/Qwen3-14B-BF16.gguf
```

## Generate imatrix
Notes:

1. This took just over 5 minutes on my high end gaming rig.
2. If you can't run the bf16 you could make a q8_0 without imatrix and then use that as "baseline" instead
3. I could offload 32 layers naievly with `-ngl 32` but do whatever you need to run inferencing e.g. `-ngl 99 -ot ...` etc.
4. I don't bother with fancy calibration corpus nor extra context length as it isn't clearly proven to always improve results afaict.
5. Assuming you're offloading some to CPU, adjust threads as needed or set to exactly 1 if you are fully offloading to VRAM.

```bash
cd ik_llama.cpp
./build/bin/llama-imatrix \
    --verbosity 1 \
    -m /mnt/llms/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-BF16.gguf \
    -f calibration_data_v5_rc.txt \
    -o ./Qwen3-14B-BF16-imatrix.dat \
    -ngl 32 \
    --layer-similarity \
    --ctx-size 512 \
    --threads 16

mv ./Qwen3-14B-BF16-imatrix.dat ../ubergarm/Qwen3-14B-GGUF/
```

## Create Quant Recipe
I personally like to make a bash script for each quant recipe. You can explore different mixes using layer-similarity or [other imatrix statistics tools](https://github.com/ggml-org/llama.cpp/pull/12718). Keep log files around with `./blah 2>&1 | tee -a logs/version-blah.log`.

I often like to off with a pure q8_0 for benchmarking and then tweak as desired for target VRAM breakpoints.

```bash
#!/usr/bin/env bash

# token_embd.weight,         torch.bfloat16 --> BF16, shape = {5120, 151936}
#
# blk.28.ffn_down.weight,    torch.bfloat16 --> BF16, shape = {17408, 5120}
# blk.28.ffn_gate.weight,    torch.bfloat16 --> BF16, shape = {5120, 17408}
# blk.28.ffn_up.weight,      torch.bfloat16 --> BF16, shape = {5120, 17408}
#
# blk.28.attn_output.weight, torch.bfloat16 --> BF16, shape = {5120, 5120}
# blk.28.attn_q.weight,      torch.bfloat16 --> BF16, shape = {5120, 5120}
# blk.28.attn_k.weight,      torch.bfloat16 --> BF16, shape = {5120, 1024}
# blk.28.attn_v.weight,      torch.bfloat16 --> BF16, shape = {5120, 1024}
#
# blk.28.attn_norm.weight,   torch.bfloat16 --> F32, shape = {5120}
# blk.28.ffn_norm.weight,    torch.bfloat16 --> F32, shape = {5120}
# blk.28.attn_k_norm.weight, torch.bfloat16 --> F32, shape = {128}
# blk.28.attn_q_norm.weight, torch.bfloat16 --> F32, shape = {128}
#
# output_norm.weight,        torch.bfloat16 --> F32, shape = {5120}
# output.weight,             torch.bfloat16 --> BF16, shape = {5120, 151936}

custom="
# Attention
blk\.[0-9]\.attn_.*\.weight=iq5_ks
blk\.[1-3][0-9]\.attn_.*\.weight=iq5_ks

# FFN
blk\.[0-9]\.ffn_down\.weight=iq5_ks
blk\.[1-3][0-9]\.ffn_down\.weight=iq5_ks

blk\.[0-9]\.ffn_(gate|up)\.weight=iq4_ks
blk\.[1-3][0-9]\.ffn_(gate|up)\.weight=iq4_ks

# Token embedding/output
token_embd\.weight=iq6_k
output\.weight=iq6_k
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

./build/bin/llama-quantize \
    --imatrix /mnt/llms/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-BF16-imatrix.dat \
    --custom-q "$custom" \
    /mnt/llms/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-BF16.gguf \
    /mnt/llms/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-IQ4_KS.gguf \
    IQ4_KS \
    16
```

## Perplexity
Run some benchmarks to compare your various quant recipes.

```bash
model=/mnt/llms/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-Q8_0.gguf

./build/bin/llama-perplexity \
    -m "$model" \
    --ctx-size 512 \
    --ubatch-size 512 \
    -f wiki.test.raw \
    -fa \
    -ngl 99 \
    --seed 1337 \
    --threads 1
```

* BF16
    - `Final estimate: PPL = 9.0128 +/- 0.07114`
* Q8_0
    - `Final estimate: PPL = 9.0281 +/- 0.07136`
* [ubergarm/IQ4_KS](https://huggingface.co/ubergarm/Qwen3-14B-GGUF#qwen3-14b-iq4_ks)
    - `Final estimate: PPL = 9.0505 +/- 0.07133`
* [unsloth/UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3-14B-GGUF?show_file_info=Qwen3-14B-UD-Q4_K_XL.gguf)
    - `Final estimate: PPL = 9.1034 +/- 0.07189`
* [bartowski/Q4_K_L](https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF?show_file_info=Qwen_Qwen3-14B-Q4_K_L.gguf)
    - `Final estimate: PPL = 9.1395 +/- 0.07236`

## KL-Divergence
You can run KLD if you want to measure how much smaller quants diverge from the unquantized model's outputs.

I have a custom ~1.6MiB `ubergarm-kld-test-corpus.txt` made from whisper-large-v3 transcriptions in plain text format from some recent episodes of [Buddha at the Gas Pump BATGAP YT Channel](https://www.youtube.com/c/batgap/videos).

#### Pass 1 Generate KLD Baseline File
The output kld base file can be quite large, this case it is ~55GiB. If
you can't run BF16, you could use Q8_0 as your baseline if necessary.

```bash
model=/mnt/llms/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-BF16.gguf
CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-perplexity \
    -m "$model" \
    --kl-divergence-base /mnt/llms/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-BF16-ubergarm-kld-test-corpus-base.dat \
    -f ubergarm-kld-test-corpus.txt \
    -fa \
    -ngl 32 \
    --seed 1337 \
    --threads 16
```

#### Pass 2 Measure KLD
This uses the above kld base file as input baseline.
```bash
model=/mnt/llms/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-IQ4_KS.gguf
CUDA_VISIBLE_DEVICES="0" \
./build/bin/llama-perplexity \
    -m "$model" \
    --kl-divergence-base /mnt/llms/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-BF16-ubergarm-kld-test-corpus-base.dat \
    --kl-divergence \
    -f ubergarm-kld-test-corpus.txt \
    -fa \
    -ngl 99 \
    --seed 1337 \
    --threads 1
```

This will report Perplexity on this corpus as well as various other statistics.

* BF16
    - `Final estimate: PPL = 14.8587 +/- 0.09987`
* Q8_0
    - `Mean PPL(Q)        :  14.846724 ± 0.099745`
    - `Median KLD: 0.000834`
    - `99.0% KLD: 0.004789`
    - `RMS Δp: 0.920 ± 0.006 %`
    - `99.0% Δp: 2.761%`
* [ubergarm/IQ4_KS](https://huggingface.co/ubergarm/Qwen3-14B-GGUF#qwen3-14b-iq4_ks)
    - `Mean PPL(Q)        :  14.881428 ± 0.099779`
    - `Median KLD: 0.004756`
    - `99.0% KLD: 0.041509`
    - `RMS Δp: 2.267 ± 0.013 %`
    - `99.0% Δp: 6.493%`
* [unsloth/UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3-14B-GGUF?show_file_info=Qwen3-14B-UD-Q4_K_XL.gguf)
    - `Mean PPL(Q)        :  14.934694 ± 0.100320`
    - `Median KLD: 0.006275`
    - `99.0% KLD: 0.060005`
    - `RMS Δp: 2.545 ± 0.015 %`
    - `99.0% Δp: 7.203%`
* [bartowski/Q4_K_L](https://huggingface.co/bartowski/Qwen_Qwen3-14B-GGUF?show_file_info=Qwen_Qwen3-14B-Q4_K_L.gguf)
    - `Mean PPL(Q)        :  14.922353 ± 0.100054`
    - `Median KLD: 0.006195`
    - `99.0% KLD: 0.063428`
    - `RMS Δp: 2.581 ± 0.015 %`
    - `99.0% Δp:  7.155%`

## Speed Benchmarks
Run some `llama-sweep-bench` to see how fast your quants are over various context lengths.

```bash
model=/mnt/llms/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-IQ4_KS.gguf
./build/bin/llama-sweep-bench \
  --model "$model" \
  -fa \
  -c 32768 \
  -ngl 99 \
  --warmup-batch \
  --threads 1
```
![sweep-bench-qwen3-14b-gguf-more-q4](https://github.com/user-attachments/assets/2ba1f817-c1b9-4648-9cab-5b759f56e4a2)

## Vibe Check
Always remember to actually *run* your model to confirm it is working properly and generating valid responses.

```bash
#!/usr/bin/env bash

model=/mnt/llms/ubergarm/Qwen3-14B-GGUF/Qwen3-14B-IQ4_KS.gguf

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

## References
* [ik_llama.cpp old getting started guide](https://github.com/ikawrakow/ik_llama.cpp/discussions/258)
* [gist with some benchmarking gist methodology](https://gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38#methodology)
* [ubergarm/Qwen3-14B-GGUF](https://huggingface.co/ubergarm/Qwen3-14B-GGUF)

---

#### 🗣️ Discussion

👤 **VinnyG9** replied the **2025-05-19** at **14:48:32**:<br>

thanks for this, can you point me where can i read a description of:
-DGGML_RPC=OFF
--seed 1337

> 👤 **ubergarm** replied the **2025-05-19** at **15:07:31**:<br>
> > -DGGML_RPC=OFF
> > --seed 1337
> 
> The had turned off the RPC backend building at some point becuase in the past I had enabled it to test some things, you can probably ignore it for the purposes of this guide. If you're interested the RPC "remote procedure call" allows you to run [a client and server(s) distributed across multiple machines or processes](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) for distributing inferencing. However, it is very basic and lacking a variety of features which make it less than useful in most of my testing and purposes.
> 
> > --seed 1337
> 
> I set the same random seed, just for fun, across all of my measurements in a hopeful attempt to reduce differences due to entropy. Not sure if it really matters. [1337](https://www.urbandictionary.com/define.php?term=1337) is leet speek for [leet](https://www.urbandictionary.com/define.php?term=leet).
> 
> 👤 **VinnyG9** replied the **2025-05-21** at **03:42:57**:<br>
> > > -DGGML_RPC=OFF
> > > --seed 1337
> > 
> > The had turned off the RPC backend building at some point becuase in the past I had enabled it to test some things, you can probably ignore it for the purposes of this guide. If you're interested the RPC "remote procedure call" allows you to run [a client and server(s) distributed across multiple machines or processes](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) for distributing inferencing. However, it is very basic and lacking a variety of features which make it less than useful in most of my testing and purposes.
> > 
> > > --seed 1337
> > 
> > I set the same random seed, just for fun, across all of my measurements in a hopeful attempt to reduce differences due to entropy. Not sure if it really matters. [1337](https://www.urbandictionary.com/define.php?term=1337) is leet speek for [leet](https://www.urbandictionary.com/define.php?term=leet).
> 
> you nerds speak like i know what you're talking about xD
> what is it "seeding"?
> i thought it was a reference to the universe's "fine-structure constant"