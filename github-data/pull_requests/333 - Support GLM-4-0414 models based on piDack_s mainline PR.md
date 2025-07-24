### 🔀 [#333](https://github.com/ikawrakow/ik_llama.cpp/pull/333) - Support GLM-4-0414 models based on piDack's mainline PR

| **Author** | `ubergarm` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-04-17 |
| **Updated** | 2025-04-21 |

---

#### Description

## tl;dr;
I got stuck on this PR and figured I'd push it anyway, no pressure to look at it.

## Status
This PR needs some more love. It is *not* working on CUDA backend, but *might* be working on CPU backend for `THUDM/GLM-Z1-Rumination-32B-0414` `bf16` GGUF converted using piDack's mainline branch.

## Purpose
The goal of this PR is to incorporate changes made by [piDack on maline llama.cpp PR#12957](https://github.com/ggml-org/llama.cpp/pull/12957) in order to support the recently updated [THUDM/glm-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e) models.

Specifically I was attempting to imatrix and quantize [THUDM/GLM-Z1-Rumination-32B-0414](https://huggingface.co/THUDM/GLM-Z1-Rumination-32B-0414/tree/main) hoping to use the new cosine similarity layer importance scoring to design a lower PPL quant.

## Details

<details>

<summary>Download and convert using piDack's mainline branch (*NOTE*: I didn't include python changes to this PR)</summary>

#### 1. Download Model
```
$ uv venv ./venv --python 3.12 --python-preference=only-managed
$ source ./venv/bin/activate
$ uv pip install huggingface-hub hf_transfer huggingface-cli
$ HF_HUB_ENABLE_HF_TRANSFER=1 \
  huggingface-cli \
    download \
    --resume-download \
    --local-dir ./ \
    THUDM/GLM-Z1-Rumination-32B-0414
```

#### 2. Quantize with mainline llama.cpp piDack branch
```
# Pull and build https://github.com/ggml-org/llama.cpp/pull/12957
$ git remote add piDack git@github.com:piDack/llama.cpp.git
$ git fetch piDack
$ git checkout piDack/update_glm4z
$ git rev-parse --short HEAD
5592c081

# build it then use to convert (dumps gguf into same dir as input files)

$ python \
    convert_hf_to_gguf.py \
    --outtype bf16 \
    --split-max-size 35G \
    /mnt/raid/models/THUDM/GLM-Z1-Rumination-32B-0414/

INFO:hf-to-gguf:Loading model: GLM-Z1-Rumination-32B-0414
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
INFO:hf-to-gguf:Exporting model...
INFO:hf-to-gguf:gguf: loading model weight map from 'model.safetensors.index.json'
INFO:hf-to-gguf:gguf: loading model part 'model-00001-of-00014.safetensors'
INFO:hf-to-gguf:token_embd.weight,                 torch.bfloat16 --> BF16, shape = {6144, 151552}
INFO:hf-to-gguf:blk.0.attn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.0.ffn_down.weight,             torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.0.ffn_up.weight,               torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.0.ffn_norm.weight,             torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.0.post_ffw_norm.weight,        torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.0.post_attention_norm.weight,  torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.0.attn_k.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.0.attn_output.weight,          torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.0.attn_q.weight,               torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.0.attn_v.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.1.attn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.1.ffn_down.weight,             torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.1.ffn_up.weight,               torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.1.ffn_norm.weight,             torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.1.post_ffw_norm.weight,        torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.1.post_attention_norm.weight,  torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.1.attn_k.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.1.attn_output.weight,          torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.1.attn_q.weight,               torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.1.attn_v.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.2.attn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.2.ffn_down.weight,             torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.2.ffn_up.weight,               torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.2.ffn_norm.weight,             torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.2.post_ffw_norm.weight,        torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.2.post_attention_norm.weight,  torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.2.attn_k.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.2.attn_output.weight,          torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.2.attn_q.weight,               torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.2.attn_v.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00002-of-00014.safetensors'
INFO:hf-to-gguf:blk.3.attn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.3.ffn_down.weight,             torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.3.ffn_up.weight,               torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.3.ffn_norm.weight,             torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.3.post_ffw_norm.weight,        torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.3.post_attention_norm.weight,  torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.3.attn_k.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.3.attn_output.weight,          torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.3.attn_q.weight,               torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.3.attn_v.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.4.attn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.4.ffn_down.weight,             torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.4.ffn_up.weight,               torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.4.ffn_norm.weight,             torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.4.post_ffw_norm.weight,        torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.4.post_attention_norm.weight,  torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.4.attn_k.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.4.attn_output.weight,          torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.4.attn_q.weight,               torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.4.attn_v.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.5.attn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.5.ffn_down.weight,             torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.5.ffn_up.weight,               torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.5.ffn_norm.weight,             torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.5.post_ffw_norm.weight,        torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.5.post_attention_norm.weight,  torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.5.attn_k.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.5.attn_output.weight,          torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.5.attn_q.weight,               torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.5.attn_v.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.6.attn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.6.ffn_down.weight,             torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.6.ffn_up.weight,               torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.6.ffn_norm.weight,             torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.6.post_ffw_norm.weight,        torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.6.post_attention_norm.weight,  torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.6.attn_k.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.6.attn_output.weight,          torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.6.attn_q.weight,               torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.6.attn_v.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.7.ffn_up.weight,               torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.7.attn_k.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.7.attn_output.weight,          torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.7.attn_q.weight,               torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.7.attn_v.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00003-of-00014.safetensors'
INFO:hf-to-gguf:blk.10.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.10.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.10.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.10.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.10.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.10.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.10.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.10.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.10.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.10.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.11.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.11.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.11.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.11.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.11.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.11.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.11.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.11.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.11.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.11.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.12.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.12.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.12.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.12.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.7.attn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.7.ffn_down.weight,             torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.7.ffn_norm.weight,             torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.7.post_ffw_norm.weight,        torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.7.post_attention_norm.weight,  torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.8.attn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.8.ffn_down.weight,             torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.8.ffn_up.weight,               torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.8.ffn_norm.weight,             torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.8.post_ffw_norm.weight,        torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.8.post_attention_norm.weight,  torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.8.attn_k.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.8.attn_output.weight,          torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.8.attn_q.weight,               torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.8.attn_v.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.9.attn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.9.ffn_down.weight,             torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.9.ffn_up.weight,               torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.9.ffn_norm.weight,             torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.9.post_ffw_norm.weight,        torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.9.post_attention_norm.weight,  torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.9.attn_k.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.9.attn_output.weight,          torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.9.attn_q.weight,               torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.9.attn_v.weight,               torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00004-of-00014.safetensors'
INFO:hf-to-gguf:blk.12.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.12.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.12.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.12.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.12.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.12.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.13.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.13.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.13.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.13.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.13.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.13.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.13.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.13.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.13.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.13.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.14.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.14.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.14.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.14.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.14.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.14.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.14.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.14.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.14.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.14.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.15.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.15.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.15.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.15.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.15.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.15.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.15.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.15.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.15.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.15.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.16.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.16.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.16.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.16.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.16.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.16.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.16.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.16.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.16.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.16.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00005-of-00014.safetensors'
INFO:hf-to-gguf:blk.17.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.17.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.17.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.17.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.17.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.17.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.17.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.17.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.17.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.17.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.18.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.18.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.18.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.18.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.18.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.18.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.18.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.18.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.18.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.18.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.19.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.19.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.19.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.19.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.19.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.19.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.19.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.19.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.19.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.19.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.20.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.20.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.20.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.20.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.20.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.20.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.20.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.20.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.20.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.20.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.21.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.21.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.21.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.21.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.21.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00006-of-00014.safetensors'
INFO:hf-to-gguf:blk.21.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.21.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.21.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.21.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.21.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.22.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.22.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.22.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.22.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.22.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.22.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.22.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.22.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.22.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.22.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.23.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.23.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.23.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.23.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.23.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.23.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.23.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.23.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.23.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.23.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.24.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.24.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.24.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.24.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.24.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.24.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.24.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.24.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.24.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.24.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.25.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.25.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.25.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.25.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.25.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.25.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.25.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.25.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.25.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.25.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.26.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.26.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.26.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.26.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00007-of-00014.safetensors'
INFO:hf-to-gguf:blk.26.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.26.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.26.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.26.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.26.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.26.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.27.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.27.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.27.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.27.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.27.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.27.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.27.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.27.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.27.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.27.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.28.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.28.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.28.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.28.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.28.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.28.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.28.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.28.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.28.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.28.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.29.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.29.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.29.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.29.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.29.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.29.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.29.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.29.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.29.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.29.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.30.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.30.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.30.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.30.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.30.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.30.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.30.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.30.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.30.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.30.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00008-of-00014.safetensors'
INFO:hf-to-gguf:blk.31.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.31.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.31.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.31.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.31.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.31.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.31.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.31.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.31.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.31.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.32.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.32.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.32.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.32.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.32.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.32.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.32.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.32.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.32.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.32.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.33.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.33.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.33.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.33.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.33.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.33.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.33.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.33.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.33.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.33.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.34.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.34.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.34.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.34.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.34.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.34.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.34.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.34.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.34.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.34.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.35.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.35.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.35.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.35.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.35.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00009-of-00014.safetensors'
INFO:hf-to-gguf:blk.35.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.35.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.35.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.35.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.35.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.36.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.36.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.36.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.36.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.36.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.36.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.36.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.36.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.36.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.36.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.37.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.37.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.37.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.37.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.37.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.37.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.37.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.37.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.37.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.37.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.38.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.38.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.38.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.38.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.38.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.38.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.38.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.38.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.38.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.38.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.39.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.39.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.39.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.39.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.39.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.39.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.39.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.39.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.39.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.39.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.40.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.40.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.40.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.40.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00010-of-00014.safetensors'
INFO:hf-to-gguf:blk.40.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.40.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.40.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.40.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.40.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.40.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.41.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.41.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.41.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.41.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.41.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.41.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.41.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.41.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.41.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.41.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.42.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.42.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.42.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.42.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.42.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.42.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.42.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.42.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.42.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.42.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.43.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.43.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.43.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.43.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.43.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.43.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.43.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.43.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.43.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.43.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.44.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.44.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.44.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.44.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.44.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.44.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.44.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.44.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.44.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.44.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00011-of-00014.safetensors'
INFO:hf-to-gguf:blk.45.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.45.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.45.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.45.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.45.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.45.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.45.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.45.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.45.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.45.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.46.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.46.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.46.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.46.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.46.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.46.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.46.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.46.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.46.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.46.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.47.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.47.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.47.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.47.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.47.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.47.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.47.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.47.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.47.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.47.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.48.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.48.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.48.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.48.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.48.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.48.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.48.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.48.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.48.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.48.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.49.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.49.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.49.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.49.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.49.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00012-of-00014.safetensors'
INFO:hf-to-gguf:blk.49.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.49.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.49.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.49.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.49.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.50.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.50.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.50.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.50.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.50.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.50.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.50.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.50.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.50.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.50.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.51.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.51.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.51.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.51.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.51.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.51.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.51.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.51.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.51.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.51.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.52.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.52.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.52.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.52.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.52.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.52.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.52.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.52.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.52.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.52.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.53.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.53.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.53.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.53.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.53.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.53.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.53.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.53.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.53.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.53.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.54.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.54.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.54.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.54.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00013-of-00014.safetensors'
INFO:hf-to-gguf:blk.54.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.54.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.54.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.54.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.54.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.54.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.55.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.55.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.55.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.55.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.55.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.55.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.55.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.55.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.55.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.55.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.56.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.56.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.56.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.56.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.56.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.56.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.56.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.56.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.56.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.56.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.57.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.57.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.57.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.57.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.57.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.57.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.57.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.57.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.57.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.57.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.58.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.58.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.58.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.58.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.58.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.58.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.58.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.58.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.58.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.58.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:gguf: loading model part 'model-00014-of-00014.safetensors'
INFO:hf-to-gguf:output.weight,                     torch.bfloat16 --> BF16, shape = {6144, 151552}
INFO:hf-to-gguf:blk.59.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.59.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.59.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.59.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.59.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.59.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.59.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.59.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.59.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.59.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.60.attn_norm.weight,           torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.60.ffn_down.weight,            torch.bfloat16 --> BF16, shape = {23040, 6144}
INFO:hf-to-gguf:blk.60.ffn_up.weight,              torch.bfloat16 --> BF16, shape = {6144, 46080}
INFO:hf-to-gguf:blk.60.ffn_norm.weight,            torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.60.post_ffw_norm.weight,       torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.60.post_attention_norm.weight, torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:blk.60.attn_k.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:blk.60.attn_output.weight,         torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.60.attn_q.weight,              torch.bfloat16 --> BF16, shape = {6144, 6144}
INFO:hf-to-gguf:blk.60.attn_v.weight,              torch.bfloat16 --> BF16, shape = {6144, 1024}
INFO:hf-to-gguf:output_norm.weight,                torch.bfloat16 --> F32, shape = {6144}
INFO:hf-to-gguf:Set meta model
INFO:hf-to-gguf:Set model parameters
INFO:hf-to-gguf:Set model tokenizer
INFO:gguf.vocab:Adding 318088 merge(s).
INFO:gguf.vocab:Setting special token type eos to 151329
INFO:gguf.vocab:Setting special token type pad to 151329
INFO:gguf.vocab:Setting special token type eot to 151336
INFO:gguf.vocab:Setting special token type unk to 151329
INFO:gguf.vocab:Setting special token type bos to 151331
INFO:gguf.vocab:Setting chat_template to [gMASK]<sop><|system|>
你是一个专业的深度研究助手，通过提供的工具与模拟浏览器交互，来帮助用户完成深度信息调研和报告撰写任务。今年是 2025 年。

<核心要求>
- 首先分解用户请求，得到包含多个子要求的列表
- 制定初始研究计划
- 进行多轮迭代搜索和页面浏览（at least 10 function calls）：
    * 根据已获得的信息调整研究计划和关键词
    * 打开页面阅读，从发现的内容中识别新的关键概念/名词
    * 从搜索结果中提取新的关键词继续搜索
    * 访问并仔细阅读相关页面，识别新的关键概念/名词

<重要配置>
- 采用语言
    * 搜索关键词：英语
    * 思考：英语

<可调用的工具列表>

[{"name": "search", "description": "Execute a search query and return search results. Use this function when you need to find information about a specific topic.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query string, use English words unless it is a proper name in Chinese"}}, "required": ["query"], "additionalProperties": false}}, {"name": "click", "description": "Click a link in the search results and navigate to the corresponding page. Use this function when you need to view detailed content of a specific search result.", "parameters": {"type": "object", "properties": {"link_id": {"type": "integer", "description": "The link ID to click (from the sequence number in search results)"}}, "required": ["link_id"], "additionalProperties": false}}, {"name": "open", "description": "Open a specific website. Get content from any website with its URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The target website URL or domain"}}, "required": ["url"], "additionalProperties": false}}, {"name": "finish", "description": "Finish the task. Use this function when you have found the information you need.", "parameters": {"type": "object", "properties": {}, "additionalProperties": false}}]

{%- for message in messages if message.role != 'system' %}{%- set role = message['role'] %}{%- set content = message['content'] %}{%- set visible = content.split('</think>')[-1].strip() %}{%- set meta = message.get("metadata", "") %}{%- if role == 'user' %}<|user|>
{{ visible }}{%- elif role == 'assistant' and not meta %}<|assistant|>
{{ visible }}{%- elif role == 'assistant' and meta %}<|assistant|>{{ meta }} 
{{ visible }}{%- elif role == 'observation' %}<|observation|>
{{ visible }}{%- endif %}{%- endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}
INFO:hf-to-gguf:Set model quantization version
INFO:gguf.gguf_writer:Writing the following files:
INFO:gguf.gguf_writer:/mnt/raid/models/THUDM/GLM-Z1-Rumination-32B-0414/GLM-Z1-Rumination-32B-0414-BF16-00001-of-00002.gguf: n_tensors = 323, total_size = 35.0G
INFO:gguf.gguf_writer:/mnt/raid/models/THUDM/GLM-Z1-Rumination-32B-0414/GLM-Z1-Rumination-32B-0414-BF16-00002-of-00002.gguf: n_tensors = 290, total_size = 31.3G

Shard (0/2): 0.00byte [00:00, ?byte/s]

Writing:   0%|          | 0.00/66.3G [00:00<?, ?byte/s][A
Shard (1/2): : 0.00byte [00:00, ?byte/s]
Shard (1/2):   0%|          | 0.00/35.0G [00:00<?, ?byte/s]
Writing: 100%|██████████| 66.3G/66.3G [01:14<00:00, 892Mbyte/s]
INFO:hf-to-gguf:Model successfully exported to /mnt/raid/models/THUDM/GLM-Z1-Rumination-32B-0414/

```

</details>

<details>

<summary>CUDA fails: This PR with `ik_llama.cpp` fork to calculate imatrix on the bf16</summary>

```
# compile with CUDA support
$ ./build/bin/llama-imatrix \
    --verbosity 1 \
    --layer-similarity \
    -m /mnt/raid/models/ubergarm/GLM-Z1-Rumination-32B-0414-GGUF/GLM-Z1-Rumination-32B-0414-BF16-00001-of-00002.gguf \
    -f calibration_data_v5_rc.txt \
    -o /mnt/raid/models/ubergarm/GLM-Z1-Rumination-32B-0414-GGUF/imatrix-GLM-Z1-Rumination-32B-0414.dat \
    --ctx-size 512 \
    --n-gpu-layers 99 \
    --threads 24

llama_model_loader: additional 1 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 35 key-value pairs and 613 tensors from /mnt/raid/models/ubergarm/GLM-Z1-Rumination-32B-0414-GG
UF/GLM-Z1-Rumination-32B-0414-BF16-00001-of-00002.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = chatglm
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = GLM Z1 Rumination 32B 0414
llama_model_loader: - kv   3:                            general.version str              = 0414
llama_model_loader: - kv   4:                           general.basename str              = GLM-Z1-Rumination
llama_model_loader: - kv   5:                         general.size_label str              = 32B
llama_model_loader: - kv   6:                            general.license str              = mit
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                          general.languages arr[str,2]       = ["zh", "en"]
llama_model_loader: - kv   9:                     chatglm.context_length u32              = 131072
llama_model_loader: - kv  10:                   chatglm.embedding_length u32              = 6144
llama_model_loader: - kv  11:                chatglm.feed_forward_length u32              = 23040
llama_model_loader: - kv  12:                        chatglm.block_count u32              = 61
llama_model_loader: - kv  13:               chatglm.attention.head_count u32              = 48
llama_model_loader: - kv  14:            chatglm.attention.head_count_kv u32              = 8
llama_model_loader: - kv  15:   chatglm.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  16:                          general.file_type u32              = 32
llama_model_loader: - kv  17:               chatglm.rope.dimension_count u32              = 64
llama_model_loader: - kv  18:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  19:                     chatglm.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = chatglm-bpe
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151552]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151552]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,318088]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151329
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151329
llama_model_loader: - kv  27:                tokenizer.ggml.eot_token_id u32              = 151336
llama_model_loader: - kv  28:            tokenizer.ggml.unknown_token_id u32              = 151329
llama_model_loader: - kv  29:                tokenizer.ggml.bos_token_id u32              = 151331
llama_model_loader: - kv  30:                    tokenizer.chat_template str              = [gMASK]<sop><|system|>\n你是一个...
llama_model_loader: - kv  31:               general.quantization_version u32              = 2
llama_model_loader: - kv  32:                                   split.no u16              = 0
llama_model_loader: - kv  33:                                split.count u16              = 2
llama_model_loader: - kv  34:                        split.tensors.count i32              = 613
llama_model_loader: - type  f32:  245 tensors
llama_model_loader: - type bf16:  368 tensors
llm_load_vocab: special tokens cache size = 14
llm_load_vocab: token to piece cache size = 0.9710 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = chatglm
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151552
llm_load_print_meta: n_merges         = 318088
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 131072
llm_load_print_meta: n_embd           = 6144
llm_load_print_meta: n_layer          = 61
llm_load_print_meta: n_head           = 48
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_rot            = 64
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 6
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 23040
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 131072
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 32B
llm_load_print_meta: model ftype      = BF16
llm_load_print_meta: model params     = 33.142 B
llm_load_print_meta: model size       = 61.734 GiB (16.001 BPW)
llm_load_print_meta: repeating layers = 58.265 GiB (16.001 BPW, 31.279 B parameters)
llm_load_print_meta: general.name     = GLM Z1 Rumination 32B 0414
llm_load_print_meta: BOS token        = 151331 '[gMASK]'
llm_load_print_meta: EOS token        = 151329 '<|endoftext|>'
llm_load_print_meta: UNK token        = 151329 '<|endoftext|>'
llm_load_print_meta: PAD token        = 151329 '<|endoftext|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 151336 '<|user|>'
llm_load_print_meta: max token length = 1024
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
  Device 1: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.28 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/62 layers to GPU
llm_load_tensors:        CPU buffer size = 33345.02 MiB
llm_load_tensors:        CPU buffer size = 29870.72 MiB
.................................................................................................
llama_new_context_with_model: n_ctx      = 512
llama_new_context_with_model: n_batch    = 512
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:  CUDA_Host KV buffer size =   122.00 MiB
llama_new_context_with_model: KV self size  =  122.00 MiB, K (f16):   61.00 MiB, V (f16):   61.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
ggml_gallocr_reserve_n: reallocating CUDA0 buffer from size 0.00 MiB to 2084.02 MiB
ggml_gallocr_reserve_n: reallocating CUDA1 buffer from size 0.00 MiB to 0.00 MiB
ggml_gallocr_reserve_n: reallocating CUDA_Host buffer from size 0.00 MiB to 13.01 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  2084.02 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    13.01 MiB
llama_new_context_with_model: graph nodes  = 1835
llama_new_context_with_model: graph splits = 735
system_info: n_threads = 24 / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 |
 FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL
_INT8 = 0 | LLAMAFILE = 1 |
compute_imatrix: tokenizing the input ..
compute_imatrix: tokenization took 1271.86 ms
compute_imatrix: computing over 220 chunks with batch_size 512
llama_output_reserve: reallocating output buffer from size 0.58 MiB to 296.00 MiB
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: CUDA graph update failed
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to too many consecutive updates
ggml_backend_cuda_graph_compute: CUDA graph update failed
nan detected in blk.1.attn_output.weight
```

</details>

<details>

<summary>CPU seems to work: This PR with `ik_llama.cpp` fork to calculate imatrix on the bf16</summary>

```bash
# compile with CPU only support
$ ./build/bin/llama-imatrix \
    --verbosity 1 \
    --layer-similarity \
    -m /mnt/raid/models/ubergarm/GLM-Z1-Rumination-32B-0414-GGUF/GLM-Z1-Rumination-32B-0414-BF16-00001-of-00002.gguf \
    -f calibration_data_v5_rc.txt \
    -o /mnt/raid/models/ubergarm/GLM-Z1-Rumination-32B-0414-GGUF/imatrix-GLM-Z1-Rumination-32B-0414.dat \
    --ctx-size 512 \
    --n-gpu-layers 99 \
    --threads 24

.
.
.
llama_kv_cache_init:        CPU KV buffer size =   122.00 MiB
llama_new_context_with_model: KV self size  =  122.00 MiB, K (f16):   61.00 MiB, V (f16):   61.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.58 MiB
ggml_gallocr_reserve_n: reallocating CPU buffer from size 0.00 MiB to 308.00 MiB
llama_new_context_with_model:        CPU compute buffer size =   308.00 MiB
llama_new_context_with_model: graph nodes  = 1835
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 24 / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
compute_imatrix: tokenizing the input ..
compute_imatrix: tokenization took 1093.25 ms
compute_imatrix: computing over 220 chunks with batch_size 512
llama_output_reserve: reallocating output buffer from size 0.58 MiB to 296.00 MiB
compute_imatrix: 176.75 seconds per pass - ETA 10 hours 48.07 minutes
[1]22.1807,[2]8.6827,[3]5.8279,^C

# takes too long at bf16 on this rig so i stopped it...
```

</details>

I'll skip ahead and try to quantize it without imatrix for now and see if it actually runs or not.

---

#### 💬 Conversation

👤 **ubergarm** commented the **2025-04-17** at **22:30:45**:<br>

Okay, after some more testing it seems to be working with CPU backend, but not with CUDA.

<details>

<summary>Quick Q4_0 quantization success</summary>

```bash
custom="
# Token embedding and output tensors
token_embd\.weight=q4_0
output\.weight=q4_0
output_norm\.weight=q4_0

# TODO customize layers based on cosine similarity layer importance scores
"

custom=$(
  echo "$custom" | grep -v '^#' | \
  sed -Ez 's:\n+:,:g;s:,$::;s:^,::'
)

# wtf is: --ignore-imatrix-rules  ?? doesn't exist?
./build/bin/llama-quantize \
    --token-embedding-type q4_0 \
    --output-tensor-type q4_0 \
    --custom-q "$custom" \
    /mnt/raid/models/ubergarm/GLM-Z1-Rumination-32B-0414-GGUF/GLM-Z1-Rumination-32B-0414-BF16-00001-of-00002.gguf \
    /mnt/raid/models/ubergarm/GLM-Z1-Rumination-32B-0414-GGUF/GLM-Z1-Rumination-32B-0414-Q4_0.gguf \
    Q4_0 \
    24

.
.
.
[  52/ 613]               blk.5.attn_norm.weight - [ 6144,     1,     1,     1], type =    f32, size =    0.023 MB
[  53/ 613]                blk.5.ffn_down.weight - [23040,  6144,     1,     1], type =   bf16, converting to q4_0 .. size =   270.00 MiB ->    75.94 MiB
[  54/ 613]                  blk.5.ffn_up.weight - [ 6144, 46080,     1,     1], type =   bf16, converting to q4_0 .. size =   540.00 MiB ->   151.88 MiB
[  55/ 613]                blk.5.ffn_norm.weight - [ 6144,     1,     1,     1], type =    f32, size =    0.023 MB
[  56/ 613]           blk.5.post_ffw_norm.weight - [ 6144,     1,     1,     1], type =    f32, size =    0.023 MB
[  57/ 613]     blk.5.post_attention_norm.weight - [ 6144,     1,     1,     1], type =    f32, size =    0.023 MB
[  58/ 613]                  blk.5.attn_k.weight - [ 6144,  1024,     1,     1], type =   bf16, converting to q4_0 .. size =    12.00 MiB ->     3.38 MiB
[  59/ 613]             blk.5.attn_output.weight - [ 6144,  6144,     1,     1], type =   bf16, Using custom type q4_0 for tensor blk.5.attn_output.weight
converting to q4_0 .. size =    72.00 MiB ->    20.25 MiB
[  60/ 613]                  blk.5.attn_q.weight - [ 6144,  6144,     1,     1], type =   bf16, converting to q4_0 .. size =    72.00 MiB ->    20.25 MiB
[  61/ 613]                  blk.5.attn_v.weight - [ 6144,  1024,     1,     1], type =   bf16, converting to q4_0 .. size =    12.00 MiB ->     3.38 MiB
[  62/ 613]               blk.6.attn_norm.weight - [ 6144,     1,     1,     1], type =    f32, size =    0.023 MB
[  63/ 613]                blk.6.ffn_down.weight - [23040,  6144,     1,     1], type =   bf16, converting to q4_0 .. size =   270.00 MiB ->    75.94 MiB
[  64/ 613]                  blk.6.ffn_up.weight - [ 6144, 46080,     1,     1], type =   bf16, converting to q4_0 .. size =   540.00 MiB ->   151.88 MiB
[  65/ 613]                blk.6.ffn_norm.weight - [ 6144,     1,     1,     1], type =    f32, size =    0.023 MB
[  66/ 613]           blk.6.post_ffw_norm.weight - [ 6144,     1,     1,     1], type =    f32, size =    0.023 MB
[  67/ 613]     blk.6.post_attention_norm.weight - [ 6144,     1,     1,     1], type =    f32, size =    0.023 MB
[  68/ 613]                  blk.6.attn_k.weight - [ 6144,  1024,     1,     1], type =   bf16, converting to q4_0 .. size =    12.00 MiB ->     3.38 MiB
[  69/ 613]             blk.6.attn_output.weight - [ 6144,  6144,     1,     1], type =   bf16, Using custom type q4_0 for tensor blk.6.attn_output.weight
converting to q4_0 .. size =    72.00 MiB ->    20.25 MiB
[  70/ 613]                  blk.6.attn_q.weight - [ 6144,  6144,     1,     1], type =   bf16, converting to q4_0 .. size =    72.00 MiB ->    20.25 MiB
[  71/ 613]                  blk.6.attn_v.weight - [ 6144,  1024,     1,     1], type =   bf16, converting to q4_0 .. size =    12.00 MiB ->     3.38 MiB
.
.
.
[ 613/ 613]                   output_norm.weight - [ 6144,     1,     1,     1], type =    f32, size =    0.023 MB
llama_model_quantize_internal: model size  = 63215.74 MB
llama_model_quantize_internal: quant size  = 17783.55 MB
```

</details>

<details>

<summary>CUDA test fails</summary>

```bash
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA RTX A6000, compute capability 8.6, VMM: yes
llm_load_tensors: ggml ctx size =    0.56 MiB
llm_load_tensors: offloading 61 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 62/62 layers to GPU
llm_load_tensors:        CPU buffer size =   499.50 MiB
llm_load_tensors:      CUDA0 buffer size = 17284.05 MiB
.................................................................................................
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  1952.00 MiB
llama_new_context_with_model: KV self size  = 1952.00 MiB, K (f16):  976.00 MiB, V (f16):  976.00 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     0.58 MiB
ggml_gallocr_reserve_n: reallocating CUDA0 buffer from size 0.00 MiB to 832.00 MiB
ggml_gallocr_reserve_n: reallocating CUDA_Host buffer from size 0.00 MiB to 28.01 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =   832.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =    28.01 MiB
llama_new_context_with_model: graph nodes  = 1835
llama_new_context_with_model: graph splits = 2
system_info: n_threads = 24 / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE =
 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order:
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature
generate: n_ctx = 8192, n_batch = 2048, n_predict = -1, n_keep = 0


The meaning of life is
ggml_backend_cuda_graph_compute: disabling CUDA graphs due to batch size > 1 [ffn_inp-0] [6144 5 1 1]
GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG

llama_print_timings:        load time =    1278.26 ms
llama_print_timings:      sample time =      17.28 ms /    51 runs   (    0.34 ms per token,  2951.56 tokens per second)
llama_print_timings: prompt eval time =      44.63 ms /     5 tokens (    8.93 ms per token,   112.04 tokens per second)
llama_print_timings:        eval time =    1545.17 ms /    50 runs   (   30.90 ms per token,    32.36 tokens per second)
llama_print_timings:       total time =    1630.87 ms /    55 tokens
```

</details>

<details>

<summary>CPU test seems okay in quick test</summary>

```bash
$ ./build/bin/llama-cli \
    --alias ubergarm/GLM-Z1-Rumination-32B-0414-Q4_0 \
    --model /mnt/raid/models/ubergarm/GLM-Z1-Rumination-32B-0414-GGUF/GLM-Z1-Rumination-32B-0414-Q4_0.gguf \
    --ctx-size 8192 \
    --parallel 1 \
    --prompt "The meaning of life is" \
    --threads 24

.
.
.
llm_load_print_meta: model size       = 17.367 GiB (4.501 BPW)
llm_load_print_meta: repeating layers = 16.391 GiB (4.501 BPW, 31.279 B parameters)
llm_load_print_meta: general.name     = GLM Z1 Rumination 32B 0414
llm_load_print_meta: BOS token        = 151331 '[gMASK]'
llm_load_print_meta: EOS token        = 151329 '<|endoftext|>'
llm_load_print_meta: UNK token        = 151329 '<|endoftext|>'
llm_load_print_meta: PAD token        = 151329 '<|endoftext|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 151336 '<|user|>'
llm_load_print_meta: max token length = 1024
llm_load_tensors: ggml ctx size =    0.28 MiB
llm_load_tensors:        CPU buffer size = 17783.55 MiB
.................................................................................................
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =  1952.00 MiB
llama_new_context_with_model: KV self size  = 1952.00 MiB, K (f16):  976.00 MiB, V (f16):  976.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.58 MiB
ggml_gallocr_reserve_n: reallocating CPU buffer from size 0.00 MiB to 832.01 MiB
llama_new_context_with_model:        CPU compute buffer size =   832.01 MiB
llama_new_context_with_model: graph nodes  = 1835
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 24 / 48 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE =
 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |
sampling:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order:
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature
generate: n_ctx = 8192, n_batch = 2048, n_predict = -1, n_keep = 0


The meaning of life is to find your gift. The

llama_print_timings:        load time =    1421.56 ms
llama_print_timings:      sample time =       2.23 ms /     6 runs   (    0.37 ms per token,  2696.63 tokens per second)
llama_print_timings: prompt eval time =    3502.11 ms /     5 tokens (  700.42 ms per token,     1.43 tokens per second)
llama_print_timings:        eval time =    5874.86 ms /     5 runs   ( 1174.97 ms per token,     0.85 tokens per second)
llama_print_timings:       total time =    9967.31 ms /    10 tokens
```

</details>


Not exactly sure, but a few possible issues given I'm not too familiar with the code-base and mainline has diverged for some of this code:

1. `batch` vs `ubatch`
2. loading contexts

---

👤 **pwilkin** commented the **2025-04-17** at **22:46:57**:<br>

Took a quick look and I think you're missing the `convert_hf_to_gguf.py` changes from this commit: https://github.com/ggml-org/llama.cpp/pull/12957/commits/b928f8ca24b1f5f4e781b57f70e375bee07a9763, those were the ones that fixed the interleaved RoPE problems with the converted / quantified models.

---

👤 **ubergarm** commented the **2025-04-17** at **23:13:50**:<br>

> Took a quick look and I think you're missing the `convert_hf_to_gguf.py` changes.

Oh wow, thanks for taking a look! Right, I was being lazy and used your branch to do the `convert_hf_to_gguf.py` and only attempted to include changes to cpp code in this PR.

It made me think to try the `Q4_0` gguf I quantized with this `ik_llama.cpp` fork back over on your mainline PR and it works with CUDA and wow yeah does this thing ruminate with the default system prompt given it is not hooked up to actual tool use deep-research stuff.

<details>

<summary>Testing this `Q4_0` on </summary>

```bash
$ git branch | grep '*'
* (HEAD detached at piDack/update_glm4z)

$ git rev-parse --short HEAD
5592c081

$ CUDA_VISIBLE_DEVICES="0," \
./build/bin/llama-cli \
    --model /mnt/raid/models/ubergarm/GLM-Z1-Rumination-32B-0414-GGUF/GLM-Z1-Rumination-32B-0414-Q4_0.gguf \
    --ctx-size 8192 \
    --parallel 1 \
    --n-gpu-layers 62 \
    --prompt "The meaning of life is" \
    --threads 24

你是一个专业的深度研究助手，通过提供的工具与模拟浏览器交互，来帮助用户完成深度信息调研和报告撰写任务。今年是 2025 年。

<核心要求>
- 首先分解用户请求，得到包含多个子要求的列表
- 制定初始研究计划
- 进行多轮迭代搜索和页面浏览（at least 10 function calls）：
    * 根据已获得的信息调整研究计划和关键词
    * 打开页面阅读，从发现的内容中识别新的关键概念/名词
    * 从搜索结果中提取新的关键词继续搜索
    * 访问并仔细阅读相关页面，识别新的关键概念/名词

<重要配置>
- 采用语言
    * 搜索关键词：英语
    * 思考：英语

<可调用的工具列表>

[{"name": "search", "description": "Execute a search query and return search results. Use this function when you need to find information about a specific topic.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query string, use English words unless it is a proper name in Chinese"}}, "required": ["query"], "additionalProperties": false}}, {"name": "click", "description": "Click a link in the search results and navigate to the corresponding page. Use this function when you need to view detailed content of a specific search result.", "parameters": {"type": "object", "properties": {"link_id": {"type": "integer", "description": "The link ID to click (from the sequence number in search results)"}}, "required": ["link_id"], "additionalProperties": false}}, {"name": "open", "description": "Open a specific website. Get content from any website with its URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The target website URL or domain"}}, "required": ["url"], "additionalProperties": false}}, {"name": "finish", "description": "Finish the task. Use this function when you have found the information you need.", "parameters": {"type": "object", "properties": {}, "additionalProperties": false}}]
The meaning of life is
<think>
So, the user's question is "The meaning of life is". Hmm, that's a pretty broad and deep question. It seems like it's expecting some kind of philosophical or existential answer. But I need to be careful because as an AI, I don't have personal beliefs or experiences; I just gather information from the web.

First, I should understand what the user is asking. Is it "What is the meaning of life?" or is it "What does 'the meaning of life' mean?" Wait, the way it's phrased, "The meaning of life is", it looks like the user is starting to state something, but it's incomplete. Maybe it's a typo or perhaps it's meant to be a question like "What is the meaning of life?".

I think it's safe to assume that the user is asking for the meaning of life. So, I need to find out what that is according to various sources on the web.

Alright, let's start with a web search. I can use the "search" action to look for information about the meaning of life.

So, my first function call would be:
</think>{"name": "search", "arguments": {"query": "meaning of life"}}
<observation>
【0†The Meaning of Life | The Official David Bowie Website†https://bowie.la/meaning-of-life/】
The Meaning Of Life is an album by David Bowie released in 1983.

【1†What is the meaning of life? - Quora†https://www.quora.com/What-is-the-meaning-of-life】
To answer this question accurately, I must first define what life is, or at least, what is not life. One way to define life is as a collection of molecules that are self-organized, and self-replicating. The question then becomes: what is the meaning of this self-organized, self-replicating, collection of molecules? In other words, what is the purpose of life? What is the function of life? Why does life exist? The answer to this question depends on whether life has a purpose. If life has a purpose, then life has meaning. If life does not have a purpose, then life has no meaning. So, does life have a purpose? This is where the debate begins. Some people believe that life has no purpose. They believe that life is a result of chance, and that there is no reason for life to exist.

.
.
.
```

</details>

---

👤 **ikawrakow** commented the **2025-04-20** at **06:15:30**:<br>

Did you see https://github.com/ggml-org/llama.cpp/pull/13021 ?

---

👤 **ubergarm** commented the **2025-04-21** at **15:36:34**:<br>

I see, the PR that actually got merged was mainline `PR#12867`. I'll close this for now and hope to get a chance to try again using that PR to guide me instead. Low priority, just having fun trying to learn a little more. Thanks!