### 📝 [#514](https://github.com/ikawrakow/ik_llama.cpp/issues/514) - CUDA Kernel Error on RTX 5090 (Compute Capability 12.0): \"no kernel image is available for execution on the device\"

| **Author** | `mtcl` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-10 |
| **Updated** | 2025-06-14 |

---

#### Description

**Description:**  

Disclaimer: I used Qwen3 to generate this message for clarity. 

When running `ik_llama.cpp` on an **RTX 5090 GPU (compute capability 12.0)**, the server crashes with the error:  
```bash
ggml_cuda_compute_forward: FUSED_RMS_NORM failed
CUDA error: no kernel image is available for execution on the device
```  
The same model works fine on an **RTX 4090 (compute capability 8.9)**. The error suggests missing CUDA kernel support for the 5090's architecture.

---

**Steps to Reproduce:**  
1. Run the server on the 5090 (device 1):  
   ```bash
   CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-server --model [MODEL_PATH] --n-gpu-layers 100 ...
   ```  
2. Observe the crash during initialization at `ggml-cuda.cu:2963`.  

---

**System Info:**  
- **GPUs**:  
  - RTX 5090 (compute cap 12.0) ❌  
  - RTX 4090 (compute cap 8.9) ✅  
- **CUDA**: Likely incompatible version (user should confirm with `nvcc --version`).  
- **Model**: `Qwen3-235B-A22B-mix-IQ3_K` (fused MoE with `flash_attn` and `fused_moe` enabled).  

---

**Root Cause:**  
The `ggml` CUDA kernels are not compiled for compute capability `12.0`. The 5090 requires CUDA 12.4+ and `-gencode arch=compute_120,code=sm_120` flags. The current build only includes support for older architectures (e.g., `sm_89` for 4090).  

---

**Request for Maintainer Action:**  
- Update the build system to detect/support newer compute capabilities.  
- Document GPU compatibility requirements (e.g., CUDA 12.4+ for RTX 5090).  

---


Startup command
```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-server \ 
    --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
    --ctx-size 40960 \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    -b 4096 -ub 4096 \
    -fmoe \
    --n-gpu-layers 100 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 56 \
    --host 0.0.0.0 \
    --port 10002
```
Full error:
```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
INFO [                    main] build info | tid="127741998542848" timestamp=1749530316 build=3738 commit="fa90a986"
INFO [                    main] system info | tid="127741998542848" timestamp=1749530316 n_threads=56 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 139
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  37:                                   split.no u16              = 0
llama_model_loader: - kv  38:                                split.count u16              = 3
llama_model_loader: - kv  39:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type iq3_k:  188 tensors
llama_model_loader: - type iq4_k:   94 tensors
llama_model_loader: - type iq6_k:  376 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 106.830 GiB (3.903 BPW) 
llm_load_print_meta: repeating layers = 105.598 GiB (3.879 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 36422.69 MiB
llm_load_tensors:        CPU buffer size = 37141.03 MiB
llm_load_tensors:        CPU buffer size = 35082.59 MiB
llm_load_tensors:        CPU buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size =  6115.01 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 40960
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3995.05 MiB
llama_new_context_with_model: KV self size  = 3995.00 MiB, K (q8_0): 1997.50 MiB, V (q8_0): 1997.50 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  2502.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   704.05 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 190
ggml_cuda_compute_forward: FUSED_RMS_NORM failed
CUDA error: no kernel image is available for execution on the device
  current device: 0, in function ggml_cuda_compute_forward at /home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:2963
  err
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
```

---

#### 💬 Conversation

👤 **mtcl** commented the **2025-06-10** at **04:56:51**:<br>

@ikawrakow  or @ubergarm  is there an easy fix? 

after installing 5090 i purged and updated the nvidia drivers etc, and rebuilt the ik_llama using this:

### pulled latest
```bash
git pull
```

### Configure CUDA+CPU Backend
```bash
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1
```
### Build
```bash
cmake --build ./build --config Release -j $(nproc)
```

---

👤 **ikawrakow** commented the **2025-06-10** at **05:19:51**:<br>

So, the default is to make a native build for the GPU you have. This works fine in most cases. I assume it gets built for the 4090 (compute 89). But it seems the 5090 is a different compute architecture, so it does not work. I have no experience with 5090s, and I'm not finding anything related to that in mainline `llama.cpp`. Can you build and run successfully with mainline?

---

👤 **mtcl** commented the **2025-06-10** at **13:54:58**:<br>

Trying with llama.cpp, pulled latest and configured like this:

```
(base) mukul@jarvis:~/dev-ai$ cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is GNU 13.3.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- GGML_SYSTEM_ARCH: x86
-- Including CPU backend
-- Found OpenMP: TRUE (found version "4.5")  
-- x86 detected
-- Adding CPU backend variant ggml-cpu: -march=native 
-- Found CUDAToolkit: /usr/local/cuda/targets/x86_64-linux/include (found version "12.9.86") 
-- CUDA Toolkit found
-- Using CUDA architectures: native
-- The CUDA compiler identification is NVIDIA 12.9.86
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- CUDA host compiler is GNU 13.3.0
-- Including CUDA backend
-- Found CURL: /usr/lib/x86_64-linux-gnu/libcurl.so (found version "8.5.0")  
-- Configuring done (7.2s)
-- Generating done (0.2s)
-- Build files have been written to: /home/mukul/dev-ai/llama.cpp/build
(base) mukul@jarvis:~/dev-ai$ 
```

build is in progress. 

Update: build succeeded

```
(base) mukul@jarvis:~/dev-ai$ cmake --build llama.cpp/build --config Release -j 100 --clean-first --target llama-cli llama-gguf-split
cp llama.cpp/build/bin/llama-* llama.cpp
[  0%] Generating build details from Git
-- Found Git: /usr/bin/git (found version "2.43.0") 
[  0%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml.cpp.o
[  2%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml-alloc.c.o
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/gguf.cpp.o
[  4%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-threading.cpp.o
[  4%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-opt.cpp.o
[  4%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.o
[  4%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml-quants.c.o
[  4%] Building CXX object common/CMakeFiles/build_info.dir/build-info.cpp.o
[  4%] Built target build_info
[  6%] Linking CXX static library libggml-base.a
[  6%] Built target ggml-base
[  6%] Building C object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/quants.c.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/repack.cpp.o
[  8%] Building C object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.c.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.cpp.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/hbm.cpp.o
[  8%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/acc.cu.o
[ 10%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/argmax.cu.o
[ 10%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/arange.cu.o
[ 12%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/traits.cpp.o
[ 14%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/diagmask.cu.o
[ 16%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/count-equal.cu.o
[ 16%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/binbcast.cu.o
[ 16%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/argsort.cu.o
[ 16%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/amx/mmq.cpp.o
[ 16%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/amx/amx.cpp.o
[ 16%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/clamp.cu.o
[ 16%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/binary-ops.cpp.o
[ 18%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/unary-ops.cpp.o
[ 22%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/llamafile/sgemm.cpp.o
[ 22%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/fattn-wmma-f16.cu.o
[ 22%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/vec.cpp.o
[ 22%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/concat.cu.o
[ 22%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/fattn-tile-f32.cu.o
[ 22%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ops.cpp.o
[ 22%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/conv-transpose-1d.cu.o
[ 22%] Building C object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/quants.c.o
[ 22%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/convert.cu.o
[ 22%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/repack.cpp.o
[ 22%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/cpy.cu.o
[ 22%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/cross-entropy-loss.cu.o
[ 22%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/fattn-tile-f16.cu.o
[ 25%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/fattn.cu.o
[ 25%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ggml-cuda.cu.o
[ 25%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmq.cu.o
[ 27%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/gla.cu.o
[ 27%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/getrows.cu.o
[ 29%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmv.cu.o
[ 29%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvq.cu.o
[ 29%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/im2col.cu.o
[ 29%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/out-prod.cu.o
[ 29%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/pad.cu.o
[ 31%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/opt-step-adamw.cu.o
[ 31%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/norm.cu.o
[ 31%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/pool2d.cu.o
[ 33%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/quantize.cu.o
[ 33%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/rope.cu.o
[ 35%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/softmax.cu.o
[ 35%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/scale.cu.o
[ 35%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ssm-conv.cu.o
[ 35%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ssm-scan.cu.o
[ 35%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/sumrows.cu.o
[ 35%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/tsembd.cu.o
[ 37%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/sum.cu.o
[ 39%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/upscale.cu.o
[ 39%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/unary.cu.o
[ 39%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/wkv.cu.o
[ 39%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_16.cu.o
[ 41%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu.o
[ 41%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_1.cu.o
[ 41%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_2.cu.o
[ 43%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu.o
[ 43%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_16.cu.o
[ 43%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_4.cu.o
[ 45%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_8.cu.o
[ 45%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2.cu.o
[ 45%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_1.cu.o
[ 47%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_2.cu.o
[ 47%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_8.cu.o
[ 47%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_16.cu.o
[ 47%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_4.cu.o
[ 50%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_64-ncols2_1.cu.o
[ 50%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_1.cu.o
[ 50%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_2.cu.o
[ 52%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_4.cu.o
[ 52%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu.o
[ 52%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-iq1_s.cu.o
[ 52%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-iq2_xs.cu.o
[ 52%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-iq3_s.cu.o
[ 52%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-iq2_xxs.cu.o
[ 54%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-iq3_xxs.cu.o
[ 56%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-q2_k.cu.o
[ 56%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-q3_k.cu.o
[ 56%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-iq4_nl.cu.o
[ 56%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-iq4_xs.cu.o
[ 56%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-q4_0.cu.o
[ 58%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-q4_1.cu.o
[ 58%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-q5_1.cu.o
[ 58%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-q5_0.cu.o
[ 58%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-q4_k.cu.o
[ 60%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-q5_k.cu.o
[ 60%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-q6_k.cu.o
[ 60%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-q8_0.cu.o
[ 60%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-vec-f32-instance-hs128-q4_0-q4_0.cu.o
[ 62%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-vec-f16-instance-hs128-q4_0-q4_0.cu.o
[ 64%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-vec-f16-instance-hs128-q8_0-q8_0.cu.o
[ 64%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-vec-f32-instance-hs128-q8_0-q8_0.cu.o
[ 64%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-vec-f16-instance-hs128-f16-f16.cu.o
[ 64%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-vec-f16-instance-hs256-f16-f16.cu.o
[ 66%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-vec-f16-instance-hs64-f16-f16.cu.o
[ 66%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-vec-f32-instance-hs128-f16-f16.cu.o
[ 68%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/mmq-instance-iq2_s.cu.o
[ 68%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-vec-f32-instance-hs256-f16-f16.cu.o
[ 68%] Building CUDA object ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/template-instances/fattn-vec-f32-instance-hs64-f16-f16.cu.o
[ 70%] Linking CXX static library libggml-cpu.a
[ 70%] Built target ggml-cpu
[ 72%] Linking CXX static library libggml-cuda.a
[ 72%] Built target ggml-cuda
[ 75%] Building CXX object ggml/src/CMakeFiles/ggml.dir/ggml-backend-reg.cpp.o
[ 75%] Linking CXX static library libggml.a
[ 75%] Built target ggml
[ 75%] Building CXX object src/CMakeFiles/llama.dir/llama.cpp.o
[ 75%] Building CXX object src/CMakeFiles/llama.dir/llama-adapter.cpp.o
[ 77%] Building CXX object src/CMakeFiles/llama.dir/llama-arch.cpp.o
[ 77%] Building CXX object src/CMakeFiles/llama.dir/llama-batch.cpp.o
[ 77%] Building CXX object src/CMakeFiles/llama.dir/llama-chat.cpp.o
[ 79%] Building CXX object src/CMakeFiles/llama.dir/llama-context.cpp.o
[ 79%] Building CXX object src/CMakeFiles/llama.dir/llama-cparams.cpp.o
[ 79%] Building CXX object src/CMakeFiles/llama.dir/llama-grammar.cpp.o
[ 81%] Building CXX object src/CMakeFiles/llama.dir/llama-hparams.cpp.o
[ 81%] Building CXX object src/CMakeFiles/llama.dir/llama-graph.cpp.o
[ 81%] Building CXX object src/CMakeFiles/llama.dir/llama-impl.cpp.o
[ 81%] Building CXX object src/CMakeFiles/llama.dir/llama-model-loader.cpp.o
[ 81%] Building CXX object src/CMakeFiles/llama.dir/llama-kv-cache-unified-iswa.cpp.o
[ 81%] Building CXX object src/CMakeFiles/llama.dir/llama-io.cpp.o
[ 83%] Building CXX object src/CMakeFiles/llama.dir/llama-kv-cache-unified.cpp.o
[ 83%] Building CXX object src/CMakeFiles/llama.dir/llama-kv-cache-recurrent.cpp.o
[ 85%] Building CXX object src/CMakeFiles/llama.dir/llama-memory.cpp.o
[ 85%] Building CXX object src/CMakeFiles/llama.dir/llama-mmap.cpp.o
[ 85%] Building CXX object src/CMakeFiles/llama.dir/llama-sampling.cpp.o
[ 87%] Building CXX object src/CMakeFiles/llama.dir/llama-model-saver.cpp.o
[ 87%] Building CXX object src/CMakeFiles/llama.dir/llama-model.cpp.o
[ 87%] Building CXX object src/CMakeFiles/llama.dir/unicode-data.cpp.o
[ 87%] Building CXX object src/CMakeFiles/llama.dir/llama-quant.cpp.o
[ 89%] Building CXX object src/CMakeFiles/llama.dir/llama-vocab.cpp.o
[ 89%] Building CXX object src/CMakeFiles/llama.dir/unicode.cpp.o
[ 91%] Linking CXX static library libllama.a
[ 91%] Built target llama
[ 91%] Building CXX object common/CMakeFiles/common.dir/arg.cpp.o
[ 93%] Building CXX object common/CMakeFiles/common.dir/chat.cpp.o
[ 93%] Building CXX object common/CMakeFiles/common.dir/chat-parser.cpp.o
[ 93%] Building CXX object common/CMakeFiles/common.dir/common.cpp.o
[ 95%] Building CXX object common/CMakeFiles/common.dir/console.cpp.o
[ 95%] Building CXX object common/CMakeFiles/common.dir/ngram-cache.cpp.o
[ 95%] Building CXX object common/CMakeFiles/common.dir/json-partial.cpp.o
[ 95%] Building CXX object common/CMakeFiles/common.dir/speculative.cpp.o
[ 97%] Building CXX object common/CMakeFiles/common.dir/sampling.cpp.o
[ 97%] Building CXX object common/CMakeFiles/common.dir/json-schema-to-grammar.cpp.o
[100%] Building CXX object common/CMakeFiles/common.dir/llguidance.cpp.o
[100%] Building CXX object common/CMakeFiles/common.dir/log.cpp.o
[100%] Building CXX object common/CMakeFiles/common.dir/regex-partial.cpp.o
[100%] Linking CXX static library libcommon.a
[100%] Built target common
[100%] Building CXX object tools/main/CMakeFiles/llama-cli.dir/main.cpp.o
[100%] Linking CXX executable ../../bin/llama-cli
[100%] Built target llama-cli
[  0%] Built target build_info
[  6%] Built target ggml-base
[ 16%] Built target ggml-cpu
[ 71%] Built target ggml-cuda
[ 73%] Built target ggml
[ 89%] Built target llama
[ 97%] Built target common
[100%] Building CXX object tools/gguf-split/CMakeFiles/llama-gguf-split.dir/gguf-split.cpp.o
[100%] Linking CXX executable ../../bin/llama-gguf-split
[100%] Built target llama-gguf-split
(base) mukul@jarvis:~/dev-ai$ 
```

---

👤 **mtcl** commented the **2025-06-10** at **14:09:37**:<br>

ok it indeed works with mainline, i validated that it indeed got loaded on 5090. 
This is the guide I used by the way:
https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune#running-qwen3-235b-a22b

![Image](https://github.com/user-attachments/assets/1bdc9f66-1da7-4c55-b3d0-d48459f682bd)

```
(base) mukul@jarvis:~/dev-ai$ ./llama.cpp/llama-cli \
    --model /media/mukul/data/models/unsloth/Qwen3-30B-A3B-GGUF/Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf \
    --threads 32 \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    -ot ".ffn_.*_exps.=CPU" \
    --seed 3407 \
    --prio 3 \
    --temp 0.6 \
    --min-p 0.0 \
    --top-p 0.95 \
    --top-k 20 \
    -no-cnv \
    --prompt "<|im_start|>user\nhey, how are you?<|im_end|>\n<|im_start|>assistant\n"
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 2 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
  Device 1: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
build: 5622 (97340b4c) with cc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 for x86_64-linux-gnu
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 4090) - 9862 MiB free
llama_model_load_from_file_impl: using device CUDA1 (NVIDIA GeForce RTX 5090) - 31518 MiB free
llama_model_loader: loaded meta data with 35 key-value pairs and 579 tensors from /media/mukul/data/models/unsloth/Qwen3-30B-A3B-GGUF/Q4_K_M/Qwen3-30B-A3B-Q4_K_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3-30B-A3B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3-30B-A3B
llama_model_loader: - kv   4:                       general.quantized_by str              = Unsloth
llama_model_loader: - kv   5:                         general.size_label str              = 30B-A3B
llama_model_loader: - kv   6:                           general.repo_url str              = https://huggingface.co/unsloth
llama_model_loader: - kv   7:                       qwen3moe.block_count u32              = 48
llama_model_loader: - kv   8:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv   9:                  qwen3moe.embedding_length u32              = 2048
llama_model_loader: - kv  10:               qwen3moe.feed_forward_length u32              = 6144
llama_model_loader: - kv  11:              qwen3moe.attention.head_count u32              = 32
llama_model_loader: - kv  12:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  13:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  14:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  15:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  16:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  17:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  18:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  19:        qwen3moe.expert_feed_forward_length u32              = 768
llama_model_loader: - kv  20:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  21:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  22:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  23:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  24:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  25:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  26:            tokenizer.ggml.padding_token_id u32              = 151654
llama_model_loader: - kv  27:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  28:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  29:               general.quantization_version u32              = 2
llama_model_loader: - kv  30:                          general.file_type u32              = 15
llama_model_loader: - kv  31:                      quantize.imatrix.file str              = Qwen3-30B-A3B-GGUF/imatrix_unsloth.dat
llama_model_loader: - kv  32:                   quantize.imatrix.dataset str              = unsloth_calibration_Qwen3-30B-A3B.txt
llama_model_loader: - kv  33:             quantize.imatrix.entries_count i32              = 384
llama_model_loader: - kv  34:              quantize.imatrix.chunks_count i32              = 32
llama_model_loader: - type  f32:  241 tensors
llama_model_loader: - type q4_K:  289 tensors
llama_model_loader: - type q6_K:   49 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_K - Medium
print_info: file size   = 17.28 GiB (4.86 BPW) 
load: special tokens cache size = 26
load: token to piece cache size = 0.9311 MB
print_info: arch             = qwen3moe
print_info: vocab_only       = 0
print_info: n_ctx_train      = 40960
print_info: n_embd           = 2048
print_info: n_layer          = 48
print_info: n_head           = 32
print_info: n_head_kv        = 4
print_info: n_rot            = 128
print_info: n_swa            = 0
print_info: is_swa_any       = 0
print_info: n_embd_head_k    = 128
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 8
print_info: n_embd_k_gqa     = 512
print_info: n_embd_v_gqa     = 512
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: f_attn_scale     = 0.0e+00
print_info: n_ff             = 6144
print_info: n_expert         = 128
print_info: n_expert_used    = 8
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 2
print_info: rope scaling     = linear
print_info: freq_base_train  = 1000000.0
print_info: freq_scale_train = 1
print_info: n_ctx_orig_yarn  = 40960
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 30B.A3B
print_info: model params     = 30.53 B
print_info: general.name     = Qwen3-30B-A3B
print_info: n_ff_exp         = 768
print_info: vocab type       = BPE
print_info: n_vocab          = 151936
print_info: n_merges         = 151387
print_info: BOS token        = 11 ','
print_info: EOS token        = 151645 '<|im_end|>'
print_info: EOT token        = 151645 '<|im_end|>'
print_info: PAD token        = 151654 '<|vision_pad|>'
print_info: LF token         = 198 'Ċ'
print_info: FIM PRE token    = 151659 '<|fim_prefix|>'
print_info: FIM SUF token    = 151661 '<|fim_suffix|>'
print_info: FIM MID token    = 151660 '<|fim_middle|>'
print_info: FIM PAD token    = 151662 '<|fim_pad|>'
print_info: FIM REP token    = 151663 '<|repo_name|>'
print_info: FIM SEP token    = 151664 '<|file_sep|>'
print_info: EOG token        = 151643 '<|endoftext|>'
print_info: EOG token        = 151645 '<|im_end|>'
print_info: EOG token        = 151662 '<|fim_pad|>'
print_info: EOG token        = 151663 '<|repo_name|>'
print_info: EOG token        = 151664 '<|file_sep|>'
print_info: max token length = 256
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 48 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 49/49 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 17447.91 MiB
load_tensors:        CUDA0 model buffer size =   135.76 MiB
load_tensors:        CUDA1 model buffer size =   648.66 MiB
....................................................................................................
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 16384
llama_context: n_ctx_per_seq = 16384
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = 0
llama_context: freq_base     = 1000000.0
llama_context: freq_scale    = 1
llama_context: n_ctx_per_seq (16384) < n_ctx_train (40960) -- the full capacity of the model will not be utilized
llama_context:  CUDA_Host  output buffer size =     0.58 MiB
llama_kv_cache_unified:      CUDA0 KV buffer size =   384.00 MiB
llama_kv_cache_unified:      CUDA1 KV buffer size =  1152.00 MiB
llama_kv_cache_unified: size = 1536.00 MiB ( 16384 cells,  48 layers,  1 seqs), K (f16):  768.00 MiB, V (f16):  768.00 MiB
llama_context:      CUDA0 compute buffer size =  1080.00 MiB
llama_context:      CUDA1 compute buffer size =  1080.00 MiB
llama_context:  CUDA_Host compute buffer size =    36.01 MiB
llama_context: graph nodes  = 3222
llama_context: graph splits = 183 (with bs=512), 98 (with bs=1)
common_init_from_params: setting dry_penalty_last_n to ctx_size = 16384
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: llama threadpool init, n_threads = 32

system_info: n_threads = 32 (n_threads_batch = 32) / 112 | CUDA : ARCHS = 890,1200 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | AMX_INT8 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | 

sampler seed: 3407
sampler params: 
	repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
	dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 16384
	top_k = 20, top_p = 0.950, min_p = 0.000, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, top_n_sigma = -1.000, temp = 0.600
	mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> dry -> top-n-sigma -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist 
generate: n_ctx = 16384, n_batch = 2048, n_predict = -1, n_keep = 0

user
hey, how are you?
assistant
<think>
Okay, the user asked "hey, how are you?" I need to respond appropriately. First, I should acknowledge their greeting. Since I'm an AI, I can't have feelings, but I can express that I'm here and ready to help. I should keep it friendly and open-ended. Maybe say something like, "Hi there! I'm just a virtual assistant, so I don't have feelings, but I'm here and ready to help with whatever you need!" That sounds good. It's polite, clear, and invites them to ask for assistance. I should check if there's anything else needed, but the user hasn't asked a specific question yet. So, just a standard friendly response should be fine.
</think>

Hi there! I'm just a virtual assistant, so I don't have feelings, but I'm here and ready to help with whatever you need! 😊 How can I assist you today? [end of text]


llama_perf_sampler_print:    sampling time =      22.49 ms /   204 runs   (    0.11 ms per token,  9069.49 tokens per second)
llama_perf_context_print:        load time =    1524.84 ms
llama_perf_context_print: prompt eval time =     149.75 ms /    14 tokens (   10.70 ms per token,    93.49 tokens per second)
llama_perf_context_print:        eval time =    4173.56 ms /   189 runs   (   22.08 ms per token,    45.29 tokens per second)
llama_perf_context_print:       total time =    4409.74 ms /   203 tokens
(base) mukul@jarvis:~/dev-ai$ 

```

---

👤 **ikawrakow** commented the **2025-06-10** at **14:15:24**:<br>

In the folder where you build mainline `llama.cpp` there must be a file called `compile_commands.json`. Can you attach it here? Thanks.

---

👤 **ubergarm** commented the **2025-06-10** at **14:24:05**:<br>

@mtcl 

I've had reports of folks with 5090's successfully using ik_llama.cpp e.g. 

>  2x5090, 2x4090, A6000, 3090
> @panchovix [discussion here](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/discussions/7)

I don't know if they are compiling differently for NVIDIA GeForce RTX 5090, compute capability 12.0 or forcing older compute capability e.g. 8.9 or the lowest for the GPU set etc.

Also I'm not sure if they are removing `-fmoe` as the error you saw says `ggml_cuda_compute_forward: FUSED_RMS_NORM failed` so possibly removing `-fmoe` might temporarily alleviate the issue but likely at a cost to performance until this is figured out better.

Something to try while you get more info for ik anyway and maybe @panchovix will have seen this before.

---

👤 **mtcl** commented the **2025-06-10** at **14:37:12**:<br>

> In the folder where you build mainline `llama.cpp` there must be a file called `compile_commands.json`. Can you attach it here? Thanks.

[compile_commands.json](https://github.com/user-attachments/files/20674801/compile_commands.json)

---

👤 **Panchovix** commented the **2025-06-10** at **14:45:03**:<br>

I have at the moment 2x5090+2x4090+2x3090+A6000 and ikllamacpp works fine.

I explicitly set the compute architecture on the compile command, but before doing this it worked without issues as well (I did it because the 3090s or A6000 could disconnect randomly and then not built with it using native)

```
cmake -B lenux \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_BLAS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
    -DGGML_IQK_FA_ALL_QUANTS=1 \
    -DGGML_SCHED_MAX_COPIES=1 \
    -DGGML_CUDA_IQK_FORCE_BF16=1 \
```

CUDA 12.8 and 12.9 worked fine to compile.

What is your OS by the way? If it's Fedora 42, since it has GCC15, it is a bit different to build it.

---

👤 **mtcl** commented the **2025-06-10** at **15:04:17**:<br>

> 
> CUDA 12.8 and 12.9 worked fine to compile.
>
```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
```

> What is your OS by the way? If it's Fedora 42, since it has GCC15, it is a bit different to build it.
```
I have ubuntu 24.04 LTS
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ uname -a
Linux jarvis 6.11.0-26-generic #26~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Apr 17 19:20:47 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ 
```


> I explicitly set the compute architecture on the compile command, but before doing this it worked without issues as well (I did it because the 3090s or A6000 could disconnect randomly and then not built with it using native)
> 
> ```
> cmake -B lenux \
>     -DGGML_CUDA=ON \
>     -DGGML_CUDA_FA_ALL_QUANTS=ON \
>     -DGGML_BLAS=OFF \
>     -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
>     -DGGML_IQK_FA_ALL_QUANTS=1 \
>     -DGGML_SCHED_MAX_COPIES=1 \
>     -DGGML_CUDA_IQK_FORCE_BF16=1 \
> ```

I tried this but it didnt work for me, detailed logs are below:


```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ cmake -B lenux \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_BLAS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
    -DGGML_IQK_FA_ALL_QUANTS=1 \
    -DGGML_SCHED_MAX_COPIES=1 \
    -DGGML_CUDA_IQK_FORCE_BF16=1
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is GNU 13.3.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: /usr/bin/git (found version "2.43.0") 
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE  
-- Found OpenMP_C: -fopenmp (found version "4.5") 
-- Found OpenMP_CXX: -fopenmp (found version "4.5") 
-- Found OpenMP: TRUE (found version "4.5")  
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Enabling IQK Flash Attention kernels
-- Including all IQK FA kernels
-- Using llamafile
-- Found CUDAToolkit: /usr/local/cuda/targets/x86_64-linux/include (found version "12.9.86") 
-- CUDA found
-- Using CUDA architectures: 86;89;120
-- The CUDA compiler identification is NVIDIA 12.9.86
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- CUDA host compiler is GNU 13.3.0

-- ccache found, compilation results will be cached. Disable with GGML_CCACHE=OFF.
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- x86 detected
-- ARCH_FLAGS = -march=native
-- Configuring done (3.6s)
-- Generating done (0.1s)
-- Build files have been written to: /home/mukul/dev-ai/ik_llama.cpp/lenux
```

And then the Build
```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ cmake --build ./build --config Release -j $(nproc)
[  0%] Generating build details from Git
[  1%] Built target xxhash
[  1%] Built target sha256
[  1%] Built target sha1
-- Found Git: /usr/bin/git (found version "2.43.0") 
[  1%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o
[  1%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_576_512.cpp.o
[  2%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_96_96.cpp.o
[  2%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_64_64.cpp.o
[  3%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_iqk_quants.cpp.o
[  3%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_256_256.cpp.o
[  4%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_192_128.cpp.o
[  4%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_128_128.cpp.o
[  4%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_floats.cpp.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_kquants.cpp.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_ktquants.cpp.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_iquants.cpp.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_1bit.cpp.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_legacy_quants.cpp.o
[  6%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o
[  6%] Building CXX object common/CMakeFiles/build_info.dir/build-info.cpp.o
[  6%] Built target build_info
[  7%] Linking CXX shared library libggml.so
[ 48%] Built target ggml
[ 48%] Linking CXX executable ../../bin/llama-gguf
[ 48%] Linking CXX executable ../../bin/llama-gguf-hash
[ 48%] Linking CXX shared library libllama.so
[ 49%] Built target llama-gguf
[ 50%] Built target llama-gguf-hash
[ 52%] Built target llama
[ 52%] Linking CXX static library libcommon.a
[ 53%] Linking CXX executable ../../bin/llama-bench-matmult
[ 53%] Linking C executable ../bin/test-c
[ 54%] Built target llava
[ 54%] Linking CXX executable ../../bin/llama-quantize-stats
[ 55%] Built target llava_static
[ 55%] Linking CXX shared library libllava_shared.so
[ 55%] Built target test-c
[ 58%] Built target common
[ 58%] Linking CXX executable ../bin/test-tokenizer-1-spm
[ 58%] Linking CXX executable ../bin/test-quantize-perf
[ 58%] Linking CXX executable ../bin/test-grad0
[ 59%] Linking CXX executable ../bin/test-tokenizer-0
[ 60%] Linking CXX executable ../bin/test-sampling
[ 60%] Linking CXX executable ../bin/test-chat-template
[ 60%] Linking CXX executable ../bin/test-quantize-fns
[ 60%] Linking CXX executable ../bin/test-tokenizer-1-bpe
[ 61%] Linking CXX executable ../bin/test-model-load-cancel
[ 62%] Linking CXX executable ../../bin/llama-baby-llama
[ 63%] Linking CXX executable ../bin/test-grammar-integration
[ 64%] Linking CXX executable ../bin/test-grammar-parser
[ 65%] Linking CXX executable ../bin/test-autorelease
[ 66%] Linking CXX executable ../bin/test-llama-grammar
[ 67%] Linking CXX executable ../bin/test-json-schema-to-grammar
[ 68%] Linking CXX executable ../../bin/llama-cvector-generator
[ 68%] Linking CXX executable ../../bin/llama-batched-bench
[ 68%] Linking CXX executable ../bin/test-rope
[ 69%] Linking CXX executable ../bin/test-backend-ops
[ 69%] Linking CXX executable ../../bin/llama-convert-llama2c-to-ggml
[ 70%] Linking CXX executable ../../bin/llama-export-lora
[ 70%] Linking CXX executable ../../bin/llama-imatrix
[ 70%] Linking CXX executable ../../bin/llama-embedding
[ 71%] Linking CXX executable ../../bin/llama-infill
[ 71%] Linking CXX executable ../../bin/llama-gritlm
[ 71%] Linking CXX executable ../../bin/llama-lookup
[ 72%] Linking CXX executable ../../bin/llama-lookup-create
[ 73%] Linking CXX executable ../../bin/llama-bench
[ 74%] Linking CXX executable ../../bin/llama-batched
[ 74%] Linking CXX executable ../../bin/llama-llava-cli
[ 74%] Linking CXX executable ../../bin/llama-minicpmv-cli
[ 74%] Linking CXX executable ../../bin/llama-lookup-stats
[ 74%] Linking CXX executable ../../bin/llama-gbnf-validator
[ 74%] Linking CXX executable ../../bin/llama-eval-callback
[ 74%] Linking CXX executable ../../bin/llama-lookahead
[ 75%] Linking CXX executable ../../bin/llama-gguf-split
[ 75%] Linking CXX executable ../../bin/llama-lookup-merge
[ 75%] Linking CXX executable ../../bin/llama-parallel
[ 75%] Linking CXX executable ../../bin/llama-passkey
[ 75%] Linking CXX executable ../../bin/llama-cli
[ 77%] Linking CXX executable ../../bin/llama-perplexity
[ 77%] Linking CXX executable ../../bin/llama-retrieval
[ 77%] Linking CXX executable ../../bin/llama-quantize
[ 78%] Linking CXX executable ../../bin/llama-save-load-state
[ 78%] Linking CXX executable ../../bin/llama-simple
[ 78%] Linking CXX executable ../../bin/llama-speculative
[ 79%] Linking CXX executable ../../bin/llama-tokenize
[ 79%] Linking CXX executable ../../bin/llama-vdot
[ 79%] Linking CXX executable ../../bin/llama-sweep-bench
[ 80%] Linking CXX executable ../../bin/llama-q8dot
[ 80%] Built target llama-bench-matmult
[ 81%] Linking CXX executable ../../bin/llama-server
[ 82%] Built target llama-quantize-stats
[ 82%] Built target llava_shared
[ 82%] Built target test-grammar-parser
[ 83%] Built target test-grad0
[ 83%] Built target test-model-load-cancel
[ 87%] Built target test-rope
[ 87%] Built target llama-convert-llama2c-to-ggml
[ 87%] Built target test-quantize-perf
[ 87%] Built target test-autorelease
[ 87%] Built target test-quantize-fns
[ 87%] Built target test-llama-grammar
[ 87%] Built target llama-q8dot
[ 87%] Built target llama-lookup-merge
[ 87%] Built target llama-gbnf-validator
[ 88%] Built target test-sampling
[ 88%] Built target llama-gguf-split
[ 88%] Built target test-backend-ops
[ 88%] Built target llama-vdot
[ 89%] Built target test-grammar-integration
[ 89%] Built target test-json-schema-to-grammar
[ 90%] Built target test-tokenizer-1-spm
[ 90%] Built target test-tokenizer-0
[ 90%] Built target llama-baby-llama
[ 91%] Built target llama-batched-bench
[ 91%] Built target llama-llava-cli
[ 91%] Built target llama-gritlm
[ 91%] Built target llama-infill
[ 91%] Built target test-tokenizer-1-bpe
[ 91%] Built target llama-embedding
[ 92%] Built target llama-lookup
[ 92%] Built target llama-cvector-generator
[ 93%] Built target llama-imatrix
[ 93%] Built target llama-lookup-create
[ 94%] Built target llama-lookahead
[ 94%] Built target llama-export-lora
[ 94%] Built target llama-batched
[ 94%] Built target llama-minicpmv-cli
[ 95%] Built target llama-lookup-stats
[ 96%] Built target test-chat-template
[ 96%] Built target llama-bench
[ 97%] Built target llama-cli
[ 97%] Built target llama-eval-callback
[ 98%] Built target llama-passkey
[ 98%] Built target llama-parallel
[ 98%] Built target llama-perplexity
[ 98%] Built target llama-quantize
[ 98%] Built target llama-retrieval
[ 99%] Built target llama-save-load-state
[ 99%] Built target llama-sweep-bench
[ 99%] Built target llama-simple
[ 99%] Built target llama-tokenize
[ 99%] Built target llama-speculative
[100%] Built target llama-server
```

Server Start
```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-server     --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf   --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K     --ctx-size 40960     -ctk q8_0 -ctv q8_0     -fa     -b 4096 -ub 4096     -fmoe     --n-gpu-layers 100     --override-tensor exps=CPU     --parallel 1     --threads 56     --host 0.0.0.0     --port 10002
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
INFO [                    main] build info | tid="134548031524864" timestamp=1749567558 build=3739 commit="3c1f2c68"
INFO [                    main] system info | tid="134548031524864" timestamp=1749567558 n_threads=56 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 139
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  37:                                   split.no u16              = 0
llama_model_loader: - kv  38:                                split.count u16              = 3
llama_model_loader: - kv  39:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type iq3_k:  188 tensors
llama_model_loader: - type iq4_k:   94 tensors
llama_model_loader: - type iq6_k:  376 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 106.830 GiB (3.903 BPW) 
llm_load_print_meta: repeating layers = 105.598 GiB (3.879 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 36422.69 MiB
llm_load_tensors:        CPU buffer size = 37141.03 MiB
llm_load_tensors:        CPU buffer size = 35082.59 MiB
llm_load_tensors:        CPU buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size =  6115.01 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 40960
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3995.05 MiB
llama_new_context_with_model: KV self size  = 3995.00 MiB, K (q8_0): 1997.50 MiB, V (q8_0): 1997.50 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  2502.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   704.05 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 190
ggml_cuda_compute_forward: FUSED_RMS_NORM failed
CUDA error: no kernel image is available for execution on the device
  current device: 0, in function ggml_cuda_compute_forward at /home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:2963
  err
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ 
```

---

👤 **mtcl** commented the **2025-06-10** at **15:07:37**:<br>

> [@mtcl](https://github.com/mtcl)
> 
> I've had reports of folks with 5090's successfully using ik_llama.cpp e.g.
> 
> > 2x5090, 2x4090, A6000, 3090
> > [@Panchovix](https://github.com/Panchovix) [discussion here](https://huggingface.co/ubergarm/DeepSeek-R1-0528-GGUF/discussions/7)
> 
> I don't know if they are compiling differently for NVIDIA GeForce RTX 5090, compute capability 12.0 or forcing older compute capability e.g. 8.9 or the lowest for the GPU set etc.
> 
> Also I'm not sure if they are removing `-fmoe` as the error you saw says `ggml_cuda_compute_forward: FUSED_RMS_NORM failed` so possibly removing `-fmoe` might temporarily alleviate the issue but likely at a cost to performance until this is figured out better.
> 
> Something to try while you get more info for ik anyway and maybe [@Panchovix](https://github.com/Panchovix) will have seen this before.

I removed -fmoe but i got the same error:

```
llama_new_context_with_model: n_ctx      = 40960
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 0
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3995.05 MiB
llama_new_context_with_model: KV self size  = 3995.00 MiB, K (q8_0): 1997.50 MiB, V (q8_0): 1997.50 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  2502.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   704.05 MiB
llama_new_context_with_model: graph nodes  = 3860
llama_new_context_with_model: graph splits = 284
ggml_cuda_compute_forward: FUSED_RMS_NORM failed
CUDA error: no kernel image is available for execution on the device
  current device: 0, in function ggml_cuda_compute_forward at /home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:2963
  err
/home/mukul/dev-ai/ik_llama.cpp/ggml/src/ggml-cuda.cu:110: CUDA error
Could not attach to process.  If your uid matches the uid of the target
process, check the setting of /proc/sys/kernel/yama/ptrace_scope, or try
again as the root user.  For more details, see /etc/sysctl.d/10-ptrace.conf
ptrace: Operation not permitted.
No stack.
The program is not being run.
Aborted (core dumped)
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ 
```

---

👤 **ikawrakow** commented the **2025-06-10** at **15:08:00**:<br>

I think `ccache` maybe the issue. Try building in a new folder.

---

👤 **mtcl** commented the **2025-06-10** at **15:09:19**:<br>

> I think `ccache` maybe the issue. Try building in a new folder.

i will delete the whole folder, reclone and rebuild. one moment please.

---

👤 **ikawrakow** commented the **2025-06-10** at **15:29:42**:<br>

@Panchovix IIRC, you were getting over 200 t/s prefill for DeepSeek-R1/V3, but I think your setup has improved since then. What is your current performance?

---

👤 **mtcl** commented the **2025-06-10** at **15:35:52**:<br>

OK this worked! This is what I had to do. 

Deleted the folder. 
Cloned again

used below command 
```bash
cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DCMAKE_CUDA_ARCHITECTURES="86;89;120"
```

```bash
cmake --build ./build --config Release -j $(nproc)
```

Below is the full log:

```
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ cd ..
(base) mukul@jarvis:~/dev-ai$ rm -rf ik_llama.cpp/
(base) mukul@jarvis:~/dev-ai$ git clone https://github.com/ikawrakow/ik_llama.cpp
Cloning into 'ik_llama.cpp'...
remote: Enumerating objects: 30315, done.
remote: Counting objects: 100% (227/227), done.
remote: Compressing objects: 100% (99/99), done.
remote: Total 30315 (delta 164), reused 151 (delta 128), pack-reused 30088 (from 3)
Receiving objects: 100% (30315/30315), 38.80 MiB | 4.59 MiB/s, done.
Resolving deltas: 100% (22926/22926), done.
(base) mukul@jarvis:~/dev-ai$ cd ik_llama.cpp
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1
-- The C compiler identification is GNU 13.3.0
-- The CXX compiler identification is GNU 13.3.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found Git: /usr/bin/git (found version "2.43.0") 
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE  
-- Found OpenMP_C: -fopenmp (found version "4.5") 
-- Found OpenMP_CXX: -fopenmp (found version "4.5") 
-- Found OpenMP: TRUE (found version "4.5")  
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Enabling IQK Flash Attention kernels
-- Using llamafile
-- Found CUDAToolkit: /usr/local/cuda/targets/x86_64-linux/include (found version "12.9.86") 
-- CUDA found
-- Using CUDA architectures: native
-- The CUDA compiler identification is NVIDIA 12.9.86
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: /usr/local/cuda/bin/nvcc - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- CUDA host compiler is GNU 13.3.0

-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- x86 detected
-- ARCH_FLAGS = -march=native
-- Configuring done (7.5s)
-- Generating done (0.1s)
-- Build files have been written to: /home/mukul/dev-ai/ik_llama.cpp/build
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ cmake -B ./build -DGGML_CUDA=ON -DGGML_BLAS=OFF -DGGML_SCHED_MAX_COPIES=1 -DCMAKE_CUDA_ARCHITECTURES="86;89;120"
-- OpenMP found
-- Using optimized iqk matrix multiplications
-- Enabling IQK Flash Attention kernels
-- Using llamafile
-- CUDA found
-- Using CUDA architectures: 86;89;120
-- CUDA host compiler is GNU 13.3.0

-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: x86_64
-- x86 detected
-- ARCH_FLAGS = -march=native
-- Configuring done (0.3s)
-- Generating done (0.1s)
-- Build files have been written to: /home/mukul/dev-ai/ik_llama.cpp/build
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ cmake --build ./build --config Release -j $(nproc)
[  0%] Generating build details from Git
[  0%] Building C object examples/gguf-hash/CMakeFiles/sha256.dir/deps/sha256/sha256.c.o
[  1%] Building C object examples/gguf-hash/CMakeFiles/xxhash.dir/deps/xxhash/xxhash.c.o
-- Found Git: /usr/bin/git (found version "2.43.0") 
[  1%] Building C object examples/gguf-hash/CMakeFiles/sha1.dir/deps/sha1/sha1.c.o
[  2%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-alloc.c.o
[  2%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml.c.o
[  3%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/argsort.cu.o
[  3%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-quants.c.o
[  3%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/arange.cu.o
[  3%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-backend.c.o
[  3%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/fattn-tile-f16.cu.o
[  4%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/acc.cu.o
[  4%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/binbcast.cu.o
[  4%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/clamp.cu.o
[  5%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/fattn-tile-f32.cu.o
[  5%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/concat.cu.o
[  5%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/conv-transpose-1d.cu.o
[  5%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/convert.cu.o
[  5%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/dmmv.cu.o
[  6%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/cpy.cu.o
[  7%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/fattn.cu.o
[  8%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/fattn-new-mma.cu.o
[  8%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/diagmask.cu.o
[  9%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/mmvq.cu.o
[  9%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/norm.cu.o
[ 10%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/pool2d.cu.o
[ 10%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/pad.cu.o
[ 10%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/mmq.cu.o
[ 10%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/rope.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/iqk_mmvq.cu.o
[ 11%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/im2col.cu.o
[ 12%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/scale.cu.o
[ 12%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/getrows.cu.o
[ 12%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/softmax.cu.o
[ 12%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/softcap.cu.o
[ 13%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/sumrows.cu.o
[ 13%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/unary.cu.o
[ 14%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/upscale.cu.o
[ 14%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/quantize.cu.o
[ 14%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/tsembd.cu.o
[ 14%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda.cu.o
[ 15%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqfloat-cpb16.cu.o
[ 15%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqhalf-cpb16.cu.o
[ 15%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqfloat-cpb32.cu.o
[ 16%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqhalf-cpb32.cu.o
[ 16%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-wmma-f16-instance-kqhalf-cpb8.cu.o
[ 16%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_8.cu.o
[ 17%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_1.cu.o
[ 17%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_2.cu.o
[ 17%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_16-ncols2_4.cu.o
[ 18%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_4.cu.o
[ 18%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_8.cu.o
[ 18%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_1.cu.o
[ 19%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_32-ncols2_2.cu.o
[ 19%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_4.cu.o
[ 19%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_2.cu.o
[ 20%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_8.cu.o
[ 21%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_1.cu.o
[ 21%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_2.cu.o
[ 21%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_64-ncols2_1.cu.o
[ 22%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_8.cu.o
[ 22%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq1_s_r4.cu.o
[ 22%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_8-ncols2_4.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_k.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq1_s.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_ks.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq3_k.cu.o
[ 23%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_s.cu.o
[ 24%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_xs.cu.o
[ 24%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq2_xxs.cu.o
[ 25%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_k.cu.o
[ 25%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq3_xxs.cu.o
[ 25%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_ks_r4.cu.o
[ 26%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq3_s.cu.o
[ 26%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_xs.cu.o
[ 27%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_nl.cu.o
[ 28%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq5_ks.cu.o
[ 28%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq5_k.cu.o
[ 28%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq5_ks_r4.cu.o
[ 28%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq4_ks.cu.o
[ 29%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-iq6_k.cu.o
[ 29%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q2_k.cu.o
[ 29%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q3_k.cu.o
[ 29%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q4_k.cu.o
[ 30%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q4_1.cu.o
[ 30%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q5_0.cu.o
[ 31%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q5_1.cu.o
[ 31%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q6_k.cu.o
[ 31%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q5_k.cu.o
[ 32%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q6_0.cu.o
[ 33%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q8_0.cu.o
[ 33%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q4_0-q4_0.cu.o
[ 33%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q4_0-q4_0.cu.o
[ 33%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q8_0-q8_0.cu.o
[ 34%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs192-q8_0-q8_0.cu.o
[ 34%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q8_0-q8_0.cu.o
[ 34%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs256-q8_0-q8_0.cu.o
[ 35%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs192-q8_0-q8_0.cu.o
[ 35%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs256-q8_0-q8_0.cu.o
[ 37%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-f16-f16.cu.o
[ 37%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-f16-f16.cu.o
[ 37%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs192-f16-f16.cu.o
[ 37%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs64-f16-f16.cu.o
[ 37%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs256-f16-f16.cu.o
[ 37%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs192-f16-f16.cu.o
[ 39%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/mmq-instance-q4_0.cu.o
[ 39%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q8_0-iq4_nl.cu.o
[ 39%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-iq4_nl-iq4_nl.cu.o
[ 39%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs256-f16-f16.cu.o
[ 38%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-iq4_nl-iq4_nl.cu.o
[ 38%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs64-f16-f16.cu.o
[ 38%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q8_0-iq4_nl.cu.o
[ 39%] Building CXX object common/CMakeFiles/build_info.dir/build-info.cpp.o
[ 39%] Built target build_info
[ 40%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q6_0-q5_0.cu.o
In function ‘SHA1Update’,
    inlined from ‘SHA1Final’ at /home/mukul/dev-ai/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:265:5:
/home/mukul/dev-ai/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: warning: ‘SHA1Transform’ reading 64 bytes from a region of size 0 [-Wstringop-overread]
  219 |             SHA1Transform(context->state, &data[i]);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/mukul/dev-ai/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: note: referencing argument 2 of type ‘const unsigned char[64]’
/home/mukul/dev-ai/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c: In function ‘SHA1Final’:
/home/mukul/dev-ai/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:54:6: note: in a call to function ‘SHA1Transform’
   54 | void SHA1Transform(
      |      ^~~~~~~~~~~~~
In function ‘SHA1Update’,
    inlined from ‘SHA1Final’ at /home/mukul/dev-ai/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:269:9:
/home/mukul/dev-ai/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: warning: ‘SHA1Transform’ reading 64 bytes from a region of size 0 [-Wstringop-overread]
  219 |             SHA1Transform(context->state, &data[i]);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/mukul/dev-ai/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:219:13: note: referencing argument 2 of type ‘const unsigned char[64]’
/home/mukul/dev-ai/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c: In function ‘SHA1Final’:
/home/mukul/dev-ai/ik_llama.cpp/examples/gguf-hash/deps/sha1/sha1.c:54:6: note: in a call to function ‘SHA1Transform’
   54 | void SHA1Transform(
      |      ^~~~~~~~~~~~~
[ 40%] Built target sha1
[ 40%] Built target sha256
[ 40%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q6_0-q5_0.cu.o
[ 40%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f16-instance-hs128-q8_0-q6_0.cu.o
[ 41%] Building CUDA object ggml/src/CMakeFiles/ggml.dir/ggml-cuda/template-instances/fattn-vec-f32-instance-hs128-q8_0-q6_0.cu.o
[ 41%] Building CXX object ggml/src/CMakeFiles/ggml.dir/llamafile/sgemm.cpp.o
[ 41%] Built target xxhash
[ 41%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_mul_mat.cpp.o
[ 42%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_flash_attn.cpp.o
[ 42%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_576_512.cpp.o
[ 43%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_192_128.cpp.o
[ 43%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_256_256.cpp.o
[ 43%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_128_128.cpp.o
[ 44%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_96_96.cpp.o
[ 44%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/fa/iqk_fa_64_64.cpp.o
[ 44%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_floats.cpp.o
[ 45%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_kquants.cpp.o
[ 45%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_iquants.cpp.o
[ 46%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_iqk_quants.cpp.o
[ 46%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_ktquants.cpp.o
[ 46%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_1bit.cpp.o
[ 46%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_gemm_legacy_quants.cpp.o
[ 47%] Building CXX object ggml/src/CMakeFiles/ggml.dir/iqk/iqk_quantize.cpp.o
[ 47%] Building C object ggml/src/CMakeFiles/ggml.dir/ggml-aarch64.c.o
[ 48%] Linking CXX shared library libggml.so
[ 48%] Built target ggml
[ 49%] Building CXX object examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/gguf-hash.cpp.o
[ 50%] Building CXX object src/CMakeFiles/llama.dir/llama-grammar.cpp.o
[ 50%] Building CXX object src/CMakeFiles/llama.dir/llama.cpp.o
[ 51%] Building CXX object examples/gguf/CMakeFiles/llama-gguf.dir/gguf.cpp.o
[ 51%] Building CXX object src/CMakeFiles/llama.dir/llama-vocab.cpp.o
[ 51%] Building CXX object src/CMakeFiles/llama.dir/llama-sampling.cpp.o
[ 51%] Building CXX object src/CMakeFiles/llama.dir/unicode.cpp.o
[ 52%] Building CXX object src/CMakeFiles/llama.dir/unicode-data.cpp.o
[ 52%] Linking CXX executable ../../bin/llama-gguf
[ 52%] Built target llama-gguf
[ 52%] Linking CXX executable ../../bin/llama-gguf-hash
[ 52%] Built target llama-gguf-hash
[ 52%] Linking CXX shared library libllama.so
[ 52%] Built target llama
[ 53%] Building CXX object common/CMakeFiles/common.dir/common.cpp.o
[ 53%] Building C object tests/CMakeFiles/test-c.dir/test-c.c.o
[ 54%] Building CXX object examples/quantize-stats/CMakeFiles/llama-quantize-stats.dir/quantize-stats.cpp.o
[ 54%] Building CXX object common/CMakeFiles/common.dir/sampling.cpp.o
[ 54%] Building CXX object common/CMakeFiles/common.dir/json-schema-to-grammar.cpp.o
[ 54%] Building CXX object common/CMakeFiles/common.dir/console.cpp.o
[ 56%] Building CXX object common/CMakeFiles/common.dir/ngram-cache.cpp.o
[ 56%] Building CXX object common/CMakeFiles/common.dir/grammar-parser.cpp.o
[ 56%] Building CXX object common/CMakeFiles/common.dir/train.cpp.o
[ 56%] Building CXX object examples/benchmark/CMakeFiles/llama-bench-matmult.dir/benchmark-matmult.cpp.o
[ 57%] Building CXX object examples/llava/CMakeFiles/llava.dir/llava.cpp.o
[ 57%] Building CXX object examples/llava/CMakeFiles/llava.dir/clip.cpp.o
[ 57%] Linking C executable ../bin/test-c
[ 57%] Built target test-c
[ 58%] Linking CXX executable ../../bin/llama-bench-matmult
[ 58%] Built target llama-bench-matmult
[ 58%] Linking CXX executable ../../bin/llama-quantize-stats
[ 58%] Built target llama-quantize-stats
[ 58%] Built target llava
[ 58%] Linking CXX shared library libllava_shared.so
[ 59%] Linking CXX static library libllava_static.a
[ 59%] Built target llava_static
[ 59%] Built target llava_shared
[ 59%] Linking CXX static library libcommon.a
[ 59%] Built target common
[ 59%] Building CXX object tests/CMakeFiles/test-tokenizer-0.dir/test-tokenizer-0.cpp.o
[ 60%] Building CXX object tests/CMakeFiles/test-tokenizer-1-spm.dir/test-tokenizer-1-spm.cpp.o
[ 60%] Building CXX object tests/CMakeFiles/test-quantize-fns.dir/get-model.cpp.o
[ 60%] Building CXX object tests/CMakeFiles/test-quantize-fns.dir/test-quantize-fns.cpp.o
[ 60%] Building CXX object tests/CMakeFiles/test-sampling.dir/test-sampling.cpp.o
[ 61%] Building CXX object tests/CMakeFiles/test-sampling.dir/get-model.cpp.o
[ 61%] Building CXX object tests/CMakeFiles/test-tokenizer-1-bpe.dir/test-tokenizer-1-bpe.cpp.o
[ 62%] Building CXX object tests/CMakeFiles/test-chat-template.dir/test-chat-template.cpp.o
[ 62%] Building CXX object tests/CMakeFiles/test-chat-template.dir/get-model.cpp.o
[ 62%] Generating loading.html.hpp
[ 63%] Generating index.html.gz.hpp
[ 63%] Building CXX object tests/CMakeFiles/test-grammar-integration.dir/get-model.cpp.o
[ 64%] Building CXX object tests/CMakeFiles/test-grammar-integration.dir/test-grammar-integration.cpp.o
[ 64%] Building CXX object tests/CMakeFiles/test-quantize-perf.dir/test-quantize-perf.cpp.o
[ 65%] Building CXX object tests/CMakeFiles/test-quantize-perf.dir/get-model.cpp.o
[ 65%] Building CXX object tests/CMakeFiles/test-json-schema-to-grammar.dir/test-json-schema-to-grammar.cpp.o
[ 65%] Building CXX object tests/CMakeFiles/test-grammar-parser.dir/get-model.cpp.o
[ 65%] Building CXX object tests/CMakeFiles/test-backend-ops.dir/test-backend-ops.cpp.o
[ 65%] Building CXX object tests/CMakeFiles/test-llama-grammar.dir/test-llama-grammar.cpp.o
[ 65%] Building CXX object tests/CMakeFiles/test-json-schema-to-grammar.dir/get-model.cpp.o
[ 65%] Building CXX object examples/cvector-generator/CMakeFiles/llama-cvector-generator.dir/cvector-generator.cpp.o
[ 66%] Building CXX object tests/CMakeFiles/test-grad0.dir/test-grad0.cpp.o
[ 66%] Building CXX object examples/baby-llama/CMakeFiles/llama-baby-llama.dir/baby-llama.cpp.o
[ 66%] Building CXX object tests/CMakeFiles/test-autorelease.dir/get-model.cpp.o
[ 66%] Building CXX object tests/CMakeFiles/test-model-load-cancel.dir/get-model.cpp.o
[ 66%] Building CXX object tests/CMakeFiles/test-rope.dir/test-rope.cpp.o
[ 66%] Building CXX object tests/CMakeFiles/test-llama-grammar.dir/get-model.cpp.o
[ 66%] Building CXX object tests/CMakeFiles/test-model-load-cancel.dir/test-model-load-cancel.cpp.o
[ 66%] Building CXX object tests/CMakeFiles/test-grad0.dir/get-model.cpp.o
[ 67%] Building CXX object tests/CMakeFiles/test-autorelease.dir/test-autorelease.cpp.o
[ 68%] Building CXX object examples/embedding/CMakeFiles/llama-embedding.dir/embedding.cpp.o
[ 68%] Building CXX object tests/CMakeFiles/test-rope.dir/get-model.cpp.o
[ 68%] Building CXX object tests/CMakeFiles/test-grammar-parser.dir/test-grammar-parser.cpp.o
[ 68%] Building CXX object tests/CMakeFiles/test-backend-ops.dir/get-model.cpp.o
[ 69%] Building CXX object examples/eval-callback/CMakeFiles/llama-eval-callback.dir/eval-callback.cpp.o
[ 70%] Building CXX object examples/convert-llama2c-to-ggml/CMakeFiles/llama-convert-llama2c-to-ggml.dir/convert-llama2c-to-ggml.cpp.o
[ 70%] Building CXX object examples/gbnf-validator/CMakeFiles/llama-gbnf-validator.dir/gbnf-validator.cpp.o
[ 70%] Building CXX object examples/export-lora/CMakeFiles/llama-export-lora.dir/export-lora.cpp.o
[ 70%] Building CXX object examples/gguf-split/CMakeFiles/llama-gguf-split.dir/gguf-split.cpp.o
[ 71%] Building CXX object examples/gritlm/CMakeFiles/llama-gritlm.dir/gritlm.cpp.o
[ 71%] Building CXX object examples/batched-bench/CMakeFiles/llama-batched-bench.dir/batched-bench.cpp.o
[ 72%] Building CXX object examples/infill/CMakeFiles/llama-infill.dir/infill.cpp.o
[ 72%] Building CXX object examples/imatrix/CMakeFiles/llama-imatrix.dir/imatrix.cpp.o
[ 72%] Building CXX object examples/llama-bench/CMakeFiles/llama-bench.dir/llama-bench.cpp.o
[ 72%] Building CXX object examples/batched/CMakeFiles/llama-batched.dir/batched.cpp.o
[ 72%] Building CXX object examples/llava/CMakeFiles/llama-minicpmv-cli.dir/minicpmv-cli.cpp.o
[ 73%] Building CXX object examples/lookahead/CMakeFiles/llama-lookahead.dir/lookahead.cpp.o
[ 73%] Building CXX object examples/llava/CMakeFiles/llama-llava-cli.dir/llava-cli.cpp.o
[ 74%] Building CXX object examples/lookup/CMakeFiles/llama-lookup.dir/lookup.cpp.o
[ 74%] Building CXX object examples/lookup/CMakeFiles/llama-lookup-merge.dir/lookup-merge.cpp.o
[ 74%] Building CXX object examples/main/CMakeFiles/llama-cli.dir/main.cpp.o
[ 75%] Building CXX object examples/lookup/CMakeFiles/llama-lookup-stats.dir/lookup-stats.cpp.o
[ 75%] Building CXX object examples/quantize/CMakeFiles/llama-quantize.dir/quantize.cpp.o
[ 75%] Building CXX object examples/parallel/CMakeFiles/llama-parallel.dir/parallel.cpp.o
[ 75%] Building CXX object examples/lookup/CMakeFiles/llama-lookup-create.dir/lookup-create.cpp.o
[ 75%] Building CXX object examples/retrieval/CMakeFiles/llama-retrieval.dir/retrieval.cpp.o
[ 75%] Building CXX object examples/simple/CMakeFiles/llama-simple.dir/simple.cpp.o
[ 75%] Building CXX object examples/speculative/CMakeFiles/llama-speculative.dir/speculative.cpp.o
[ 75%] Building CXX object examples/tokenize/CMakeFiles/llama-tokenize.dir/tokenize.cpp.o
[ 75%] Building CXX object examples/perplexity/CMakeFiles/llama-perplexity.dir/perplexity.cpp.o
[ 76%] Building CXX object examples/passkey/CMakeFiles/llama-passkey.dir/passkey.cpp.o
[ 76%] Building CXX object examples/save-load-state/CMakeFiles/llama-save-load-state.dir/save-load-state.cpp.o
[ 77%] Building CXX object pocs/vdot/CMakeFiles/llama-vdot.dir/vdot.cpp.o
[ 77%] Building CXX object examples/sweep-bench/CMakeFiles/llama-sweep-bench.dir/sweep-bench.cpp.o
[ 77%] Building CXX object pocs/vdot/CMakeFiles/llama-q8dot.dir/q8dot.cpp.o
[ 78%] Linking CXX executable ../bin/test-model-load-cancel
[ 78%] Built target test-model-load-cancel
[ 79%] Linking CXX executable ../bin/test-autorelease
[ 79%] Linking CXX executable ../bin/test-rope
[ 79%] Built target test-autorelease
[ 79%] Built target test-rope
[ 80%] Linking CXX executable ../bin/test-quantize-fns
[ 80%] Built target test-quantize-fns
[ 81%] Linking CXX executable ../../bin/llama-baby-llama
[ 81%] Linking CXX executable ../../bin/llama-lookup-merge
[ 81%] Linking CXX executable ../bin/test-sampling
[ 81%] Linking CXX executable ../bin/test-tokenizer-1-spm
[ 83%] Linking CXX executable ../../bin/llama-q8dot
[ 83%] Linking CXX executable ../bin/test-grammar-parser
[ 83%] Linking CXX executable ../../bin/llama-gbnf-validator
[ 83%] Built target llama-lookup-merge
[ 83%] Built target test-sampling
[ 83%] Built target llama-baby-llama
[ 84%] Linking CXX executable ../../bin/llama-tokenize
[ 84%] Built target test-grammar-parser
[ 84%] Built target llama-q8dot
[ 84%] Linking CXX executable ../../bin/llama-vdot
[ 84%] Linking CXX executable ../bin/test-chat-template
[ 84%] Built target llama-gbnf-validator
[ 84%] Built target test-tokenizer-1-spm
[ 85%] Linking CXX executable ../../bin/llama-lookup-create
[ 85%] Built target llama-tokenize
[ 85%] Built target llama-vdot
[ 85%] Built target test-chat-template
[ 85%] Linking CXX executable ../../bin/llama-eval-callback
[ 85%] Linking CXX executable ../bin/test-grad0
[ 86%] Linking CXX executable ../../bin/llama-gguf-split
[ 86%] Built target llama-lookup-create
[ 86%] Built target test-grad0
[ 87%] Linking CXX executable ../bin/test-llama-grammar
[ 87%] Built target llama-gguf-split
[ 87%] Built target llama-eval-callback
[ 88%] Linking CXX executable ../../bin/llama-simple
[ 88%] Linking CXX executable ../../bin/llama-batched-bench
[ 88%] Linking CXX executable ../../bin/llama-gritlm
[ 88%] Linking CXX executable ../../bin/llama-sweep-bench
[ 88%] Linking CXX executable ../../bin/llama-embedding
[ 89%] Linking CXX executable ../bin/test-tokenizer-0
[ 89%] Built target test-llama-grammar
[ 89%] Linking CXX executable ../../bin/llama-lookup-stats
[ 89%] Linking CXX executable ../../bin/llama-batched
[ 89%] Linking CXX executable ../../bin/llama-save-load-state
[ 89%] Built target llama-simple
[ 89%] Built target llama-batched-bench
[ 89%] Built target llama-gritlm
[ 89%] Built target llama-sweep-bench
[ 89%] Built target llama-embedding
[ 89%] Built target test-tokenizer-0
[ 89%] Building CXX object examples/server/CMakeFiles/llama-server.dir/server.cpp.o
[ 89%] Built target llama-lookup-stats
[ 89%] Linking CXX executable ../../bin/llama-passkey
[ 89%] Built target llama-batched
[ 89%] Built target llama-save-load-state
[ 89%] Linking CXX executable ../bin/test-quantize-perf
[ 90%] Linking CXX executable ../../bin/llama-minicpmv-cli
[ 90%] Linking CXX executable ../../bin/llama-lookup
[ 90%] Built target llama-passkey
[ 90%] Built target test-quantize-perf
[ 90%] Linking CXX executable ../../bin/llama-lookahead
[ 90%] Linking CXX executable ../../bin/llama-parallel
[ 90%] Built target llama-minicpmv-cli
[ 90%] Built target llama-lookup
[ 90%] Linking CXX executable ../../bin/llama-llava-cli
[ 90%] Built target llama-lookahead
[ 90%] Linking CXX executable ../../bin/llama-convert-llama2c-to-ggml
[ 90%] Built target llama-parallel
[ 90%] Built target llama-llava-cli
[ 90%] Built target llama-convert-llama2c-to-ggml
[ 91%] Linking CXX executable ../../bin/llama-retrieval
[ 92%] Linking CXX executable ../../bin/llama-export-lora
[ 92%] Linking CXX executable ../../bin/llama-quantize
[ 92%] Built target llama-retrieval
[ 93%] Linking CXX executable ../../bin/llama-cvector-generator
[ 94%] Linking CXX executable ../../bin/llama-infill
[ 94%] Built target llama-export-lora
[ 94%] Built target llama-quantize
[ 94%] Built target llama-cvector-generator
[ 94%] Built target llama-infill
[ 94%] Linking CXX executable ../../bin/llama-speculative
[ 94%] Built target llama-speculative
[ 94%] Linking CXX executable ../bin/test-tokenizer-1-bpe
[ 94%] Linking CXX executable ../../bin/llama-cli
[ 94%] Built target test-tokenizer-1-bpe
[ 94%] Built target llama-cli
[ 94%] Linking CXX executable ../../bin/llama-imatrix
[ 94%] Built target llama-imatrix
[ 95%] Linking CXX executable ../../bin/llama-perplexity
[ 96%] Linking CXX executable ../bin/test-backend-ops
[ 96%] Built target llama-perplexity
[ 96%] Built target test-backend-ops
[ 97%] Linking CXX executable ../bin/test-grammar-integration
[ 97%] Built target test-grammar-integration
[ 98%] Linking CXX executable ../bin/test-json-schema-to-grammar
[ 98%] Built target test-json-schema-to-grammar
[ 99%] Linking CXX executable ../../bin/llama-bench
[ 99%] Built target llama-bench
[100%] Linking CXX executable ../../bin/llama-server
[100%] Built target llama-server
(base) mukul@jarvis:~/dev-ai/ik_llama.cpp$ CUDA_VISIBLE_DEVICES="1" ./build/bin/llama-server \ 
    --model /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf \
  --alias ubergarm/Qwen3-235B-A22B-mix-IQ3_K \
    --ctx-size 40960 \
    -ctk q8_0 -ctv q8_0 \
    -fa \
    -b 4096 -ub 4096 \
    -fmoe \
    --n-gpu-layers 100 \
    --override-tensor exps=CPU \
    --parallel 1 \
    --threads 56 \
    --host 0.0.0.0 \
    --port 10002
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 5090, compute capability 12.0, VMM: yes
INFO [                    main] build info | tid="132980309143552" timestamp=1749569259 build=3739 commit="3c1f2c68"
INFO [                    main] system info | tid="132980309143552" timestamp=1749569259 n_threads=56 n_threads_batch=-1 total_threads=112 system_info="AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 1 | AVX512_VNNI = 1 | AVX512_BF16 = 1 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | "
llama_model_loader: additional 2 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 40 key-value pairs and 1131 tensors from /media/mukul/backup/models/ubergarm/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-mix-IQ3_K-00001-of-00003.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = qwen3moe
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = Qwen3 235B A22B
llama_model_loader: - kv   3:                           general.basename str              = Qwen3
llama_model_loader: - kv   4:                         general.size_label str              = 235B-A22B
llama_model_loader: - kv   5:                            general.license str              = apache-2.0
llama_model_loader: - kv   6:                       general.license.link str              = https://huggingface.co/Qwen/Qwen3-235...
llama_model_loader: - kv   7:                               general.tags arr[str,1]       = ["text-generation"]
llama_model_loader: - kv   8:                       qwen3moe.block_count u32              = 94
llama_model_loader: - kv   9:                    qwen3moe.context_length u32              = 40960
llama_model_loader: - kv  10:                  qwen3moe.embedding_length u32              = 4096
llama_model_loader: - kv  11:               qwen3moe.feed_forward_length u32              = 12288
llama_model_loader: - kv  12:              qwen3moe.attention.head_count u32              = 64
llama_model_loader: - kv  13:           qwen3moe.attention.head_count_kv u32              = 4
llama_model_loader: - kv  14:                    qwen3moe.rope.freq_base f32              = 1000000.000000
llama_model_loader: - kv  15:  qwen3moe.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  16:                 qwen3moe.expert_used_count u32              = 8
llama_model_loader: - kv  17:              qwen3moe.attention.key_length u32              = 128
llama_model_loader: - kv  18:            qwen3moe.attention.value_length u32              = 128
llama_model_loader: - kv  19:                          general.file_type u32              = 139
llama_model_loader: - kv  20:                      qwen3moe.expert_count u32              = 128
llama_model_loader: - kv  21:        qwen3moe.expert_feed_forward_length u32              = 1536
llama_model_loader: - kv  22:               general.quantization_version u32              = 2
llama_model_loader: - kv  23:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  24:                         tokenizer.ggml.pre str              = qwen2
llama_model_loader: - kv  25:                      tokenizer.ggml.tokens arr[str,151936]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  26:                  tokenizer.ggml.token_type arr[i32,151936]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  27:                      tokenizer.ggml.merges arr[str,151387]  = ["Ġ Ġ", "ĠĠ ĠĠ", "i n", "Ġ t",...
llama_model_loader: - kv  28:                tokenizer.ggml.eos_token_id u32              = 151645
llama_model_loader: - kv  29:            tokenizer.ggml.padding_token_id u32              = 151643
llama_model_loader: - kv  30:                tokenizer.ggml.bos_token_id u32              = 151643
llama_model_loader: - kv  31:               tokenizer.ggml.add_bos_token bool             = false
llama_model_loader: - kv  32:                    tokenizer.chat_template str              = {%- if tools %}\n    {{- '<|im_start|>...
llama_model_loader: - kv  33:                      quantize.imatrix.file str              = /mnt/raid/models/ubergarm/Qwen3-235B-...
llama_model_loader: - kv  34:                   quantize.imatrix.dataset str              = calibration_data_v5_rc.txt
llama_model_loader: - kv  35:             quantize.imatrix.entries_count i32              = 753
llama_model_loader: - kv  36:              quantize.imatrix.chunks_count i32              = 225
llama_model_loader: - kv  37:                                   split.no u16              = 0
llama_model_loader: - kv  38:                                split.count u16              = 3
llama_model_loader: - kv  39:                        split.tensors.count i32              = 1131
llama_model_loader: - type  f32:  471 tensors
llama_model_loader: - type q8_0:    2 tensors
llama_model_loader: - type iq3_k:  188 tensors
llama_model_loader: - type iq4_k:   94 tensors
llama_model_loader: - type iq6_k:  376 tensors
llm_load_vocab: special tokens cache size = 26
llm_load_vocab: token to piece cache size = 0.9311 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = qwen3moe
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 151936
llm_load_print_meta: n_merges         = 151387
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 40960
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_layer          = 94
llm_load_print_meta: n_head           = 64
llm_load_print_meta: n_head_kv        = 4
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_swa_pattern    = 1
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 16
llm_load_print_meta: n_embd_k_gqa     = 512
llm_load_print_meta: n_embd_v_gqa     = 512
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 12288
llm_load_print_meta: n_expert         = 128
llm_load_print_meta: n_expert_used    = 8
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 1000000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 40960
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = IQ3_K - 3.4325 bpw
llm_load_print_meta: model params     = 235.094 B
llm_load_print_meta: model size       = 106.830 GiB (3.903 BPW) 
llm_load_print_meta: repeating layers = 105.598 GiB (3.879 BPW, 233.849 B parameters)
llm_load_print_meta: general.name     = Qwen3 235B A22B
llm_load_print_meta: BOS token        = 151643 '<|endoftext|>'
llm_load_print_meta: EOS token        = 151645 '<|im_end|>'
llm_load_print_meta: PAD token        = 151643 '<|endoftext|>'
llm_load_print_meta: LF token         = 148848 'ÄĬ'
llm_load_print_meta: EOT token        = 151645 '<|im_end|>'
llm_load_print_meta: max token length = 256
llm_load_print_meta: n_ff_exp         = 1536
llm_load_tensors: ggml ctx size =    0.99 MiB
Tensor blk.0.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.0.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.1.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.2.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.3.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.4.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.5.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.6.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.7.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.8.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.9.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.10.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.11.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.12.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.13.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.14.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.15.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.16.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.17.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.18.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.19.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.20.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.21.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.22.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.23.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.24.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.25.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.26.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.27.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.28.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.29.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.30.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.31.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.32.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.33.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.34.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.35.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.36.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.37.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.38.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.39.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.40.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.41.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.42.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.43.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.44.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.45.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.46.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.47.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.48.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.49.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.50.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.51.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.52.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.53.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.54.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.55.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.56.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.57.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.58.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.59.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.60.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.61.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.62.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.63.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.64.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.65.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.66.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.67.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.68.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.69.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.70.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.71.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.72.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.73.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.74.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.75.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.76.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.77.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.78.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.79.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.80.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.81.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.82.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.83.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.84.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.85.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.86.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.87.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.88.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.89.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.90.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.91.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.92.ffn_up_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_gate_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_down_exps.weight buffer type overriden to CPU
Tensor blk.93.ffn_up_exps.weight buffer type overriden to CPU
llm_load_tensors: offloading 94 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 95/95 layers to GPU
llm_load_tensors:        CPU buffer size = 36422.69 MiB
llm_load_tensors:        CPU buffer size = 37141.03 MiB
llm_load_tensors:        CPU buffer size = 35082.59 MiB
llm_load_tensors:        CPU buffer size =   630.59 MiB
llm_load_tensors:      CUDA0 buffer size =  6115.01 MiB
....................................................................................................
llama_new_context_with_model: n_ctx      = 40960
llama_new_context_with_model: n_batch    = 4096
llama_new_context_with_model: n_ubatch   = 4096
llama_new_context_with_model: flash_attn = 1
llama_new_context_with_model: mla_attn   = 0
llama_new_context_with_model: attn_max_b = 0
llama_new_context_with_model: fused_moe  = 1
llama_new_context_with_model: ser        = -1, 0
llama_new_context_with_model: freq_base  = 1000000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:      CUDA0 KV buffer size =  3995.05 MiB
llama_new_context_with_model: KV self size  = 3995.00 MiB, K (q8_0): 1997.50 MiB, V (q8_0): 1997.50 MiB
llama_new_context_with_model:  CUDA_Host  output buffer size =     1.16 MiB
llama_new_context_with_model:      CUDA0 compute buffer size =  2502.00 MiB
llama_new_context_with_model:  CUDA_Host compute buffer size =   704.05 MiB
llama_new_context_with_model: graph nodes  = 3672
llama_new_context_with_model: graph splits = 190
INFO [                    init] initializing slots | tid="132980309143552" timestamp=1749569286 n_slots=1
INFO [                    init] new slot | tid="132980309143552" timestamp=1749569286 id_slot=0 n_ctx_slot=40960
INFO [                    main] model loaded | tid="132980309143552" timestamp=1749569286
INFO [                    main] chat template | tid="132980309143552" timestamp=1749569286 chat_example="<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n" built_in=true
INFO [                    main] HTTP server listening | tid="132980309143552" timestamp=1749569286 n_threads_http="111" port="10002" hostname="0.0.0.0"
INFO [            update_slots] all slots are idle | tid="132980309143552" timestamp=1749569286
INFO [      log_server_request] request | tid="132970105794560" timestamp=1749569300 remote_addr="172.17.0.3" remote_port=46260 status=200 method="GET" path="/v1/models" params={}
INFO [      log_server_request] request | tid="132970097401856" timestamp=1749569327 remote_addr="172.17.0.3" remote_port=41930 status=200 method="GET" path="/v1/models" params={}
INFO [   launch_slot_with_task] slot is processing task | tid="132980309143552" timestamp=1749569330 id_slot=0 id_task=0
INFO [            update_slots] kv cache rm [p0, end) | tid="132980309143552" timestamp=1749569330 id_slot=0 id_task=0 p0=0
INFO [            update_slots] kv cache rm [p0, end) | tid="132980309143552" timestamp=1749569344 id_slot=0 id_task=0 p0=4096
INFO [            update_slots] kv cache rm [p0, end) | tid="132980309143552" timestamp=1749569358 id_slot=0 id_task=0 p0=8192
INFO [            update_slots] kv cache rm [p0, end) | tid="132980309143552" timestamp=1749569371 id_slot=0 id_task=0 p0=12288
INFO [            update_slots] kv cache rm [p0, end) | tid="132980309143552" timestamp=1749569385 id_slot=0 id_task=0 p0=16384
INFO [            update_slots] kv cache rm [p0, end) | tid="132980309143552" timestamp=1749569399 id_slot=0 id_task=0 p0=20480
INFO [            update_slots] kv cache rm [p0, end) | tid="132980309143552" timestamp=1749569414 id_slot=0 id_task=0 p0=24576
INFO [      log_server_request] request | tid="132970080616448" timestamp=1749569420 remote_addr="172.17.0.3" remote_port=48978 status=200 method="GET" path="/v1/models" params={}
INFO [            update_slots] kv cache rm [p0, end) | tid="132980309143552" timestamp=1749569429 id_slot=0 id_task=0 p0=28672
INFO [           print_timings] prompt eval time     =  113306.35 ms / 31923 tokens (    3.55 ms per token,   281.74 tokens per second) | tid="132980309143552" timestamp=1749569476 id_slot=0 id_task=0 t_prompt_processing=113306.351 n_prompt_tokens_processed=31923 t_token=3.549364126178617 n_tokens_second=281.74060604952325
INFO [           print_timings] generation eval time =   32427.96 ms /   389 runs   (   83.36 ms per token,    12.00 tokens per second) | tid="132980309143552" timestamp=1749569476 id_slot=0 id_task=0 t_token_generation=32427.96 n_decoded=389 t_token=83.36236503856041 n_tokens_second=11.99582089036745
INFO [           print_timings]           total time =  145734.31 ms | tid="132980309143552" timestamp=1749569476 id_slot=0 id_task=0 t_prompt_processing=113306.351 t_token_generation=32427.96 t_total=145734.311
INFO [            update_slots] slot released | tid="132980309143552" timestamp=1749569476 id_slot=0 id_task=0 n_ctx=40960 n_past=32311 n_system_tokens=0 n_cache_tokens=32311 truncated=false
INFO [            update_slots] all slots are idle | tid="132980309143552" timestamp=1749569476
INFO [      log_server_request] request | tid="132970089009152" timestamp=1749569476 remote_addr="172.17.0.3" remote_port=41946 status=200 method="POST" path="/v1/chat/completions" params={}
INFO [            update_slots] all slots are idle | tid="132980309143552" timestamp=1749569476

```

---

👤 **ikawrakow** commented the **2025-06-10** at **15:44:24**:<br>

I think `ccache` was the issue. It is very useful when not making significant changes to the setup. But it does get confused and does not rebuild correctly what needs to be rebuild. So, in the future, if you fetch a new version of `ik_llama.cpp`, update CUDA, change computer setup, etc., it is best to just delete the existing folder and rebuild from scratch.

---

👤 **Panchovix** commented the **2025-06-10** at **16:13:42**:<br>

> [@Panchovix](https://github.com/Panchovix) IIRC, you were getting over 200 t/s prefill for DeepSeek-R1/V3, but I think your setup has improved since then. What is your current performance?

@ikawrakow I was on 5090+4090x2+3090+A6000, but then:

I got another 5090 for cheap (1800USD or so)

The A6000 and the 3090 stopped working from one day to another.

Got another 3090 that worked.

Then I was with 5090x2+4090x2+3090,

Then I re soldered the PCIe power connector on the 3090 and the EPS cable on the A6000 (and on the latter used a direct EPS cable) and they have revived, since 2 days ago.

So I haven't tested much recently, but I think on Q3_K_KL I went to a higher batch size, and testing -b/ub 4096 I was getting above 300 t/s PP IIRC, but I think I tested on chats with less than 5K ctx so real speed could be higher.

---

👤 **mtcl** commented the **2025-06-10** at **16:16:45**:<br>

I currently have 1*5090, 2*4090, 1*3090 and I'll be getting another 5090 tomorrow.

I was originally going to sell everything else and keep only 2*5090s, is there any reason to keep more cards? Two 5090s are so sleek and small that the looks to performance ratio may not be worth it 😂

---

👤 **Panchovix** commented the **2025-06-10** at **20:25:07**:<br>

@mtcl just no self control, and being able to run Q3 Deepseek 685B models without much issues. Also can *kinda* run IQ4_XS quant with about 20GB RAM left or so.

---

👤 **Panchovix** commented the **2025-06-10** at **20:25:07**:<br>

@mtcl just no self control, and being bale to run Q3 Deepseek 685B models without much issues. Also can *kinda* run IQ4_NL quant, but just barely.

---

👤 **RodriMora** commented the **2025-06-11** at **16:15:46**:<br>

I do have 2x5090 and 4x3090 and had no problem building at all. I have ccache installed too. How I usually do it:

```
rm -rf build
cmake -B build -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_IQK_FORCE_BF16=1
cmake --build build --config Release --clean-first -j$(nproc)
```

for mainline I use
`cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON`


200t/s pp, 13t/s tg

---

👤 **ikawrakow** commented the **2025-06-11** at **16:31:19**:<br>

> 200t/s pp, 13t/s tg

With `ik_llama.cpp` or with `llama.cpp`?

---

👤 **RodriMora** commented the **2025-06-11** at **16:32:54**:<br>

> > 200t/s pp, 13t/s tg
> 
> With `ik_llama.cpp` or with `llama.cpp`?

`ik_llama.cpp` with ubergarm's quants at IQ2_K_R4

Edit: did a quick sweep bench now

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
|  4096 |   1024 |      0 |   15.523 |   263.87 |   79.492 |    12.88 |
|  4096 |   1024 |   4096 |   15.698 |   260.93 |   81.500 |    12.56 |
|  4096 |   1024 |   8192 |   17.091 |   239.65 |   84.523 |    12.11 |
|  4096 |   1024 |  12288 |   19.241 |   212.87 |   86.913 |    11.78 |

---

👤 **Panchovix** commented the **2025-06-11** at **16:46:36**:<br>

@RodriMora Can you tell me the command to run this bench please? Maybe I can try with Q3_K_XL and IQ3_K_R4. I guess you're using a quite big ubatch size?

---

👤 **Panchovix** commented the **2025-06-11** at **16:46:36**:<br>

@RodriMora Can you tell me the command to run this bench? Maybe I can try with Q3_K_XL and IQ3_K_R4. I guess you're using a quite big ubatch size?

---

👤 **RodriMora** commented the **2025-06-11** at **16:56:48**:<br>

> [@RodriMora](https://github.com/RodriMora) Can you tell me the command to run this bench please? Maybe I can try with Q3_K_XL and IQ3_K_R4. I guess you're using a quite big ubatch size?

The -ot are specific for my setup, the CUDA2 and CUDA4 are the 5090s. 0,1,3,5 are the 3090s
```

CUDA_VISIBLE_DEVICES="2,4,0,1,3,5" \
                                             ./build/bin/llama-sweep-bench \
                                              --model /mnt/llms/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ2_K_R4/DeepSeek-V3-0324-IQ2_K_R4-00001-of-00005.gguf \
                                              --alias ubergarm/DeepSeek-V3-0324-IQ2_K_R4 -mla 3 -fa \
                                              -amb 512 \
                                              -fmoe \
                                              -ctk f16 \
                                              -c 16384 \
                                              -ngl 99 \
                                              -ot "blk\.(3|4|5|6|7)\.ffn_.*=CUDA0" \
                                              -ot "blk\.(9|10|11|12|13)\.ffn_.*=CUDA1" \
                                              -ot "blk\.(15|16|17)\.ffn_.*=CUDA2" \
                                              -ot "blk\.(20|21|22)\.ffn_.*=CUDA3" \
                                              -ot "blk\.(25|26|27)\.ffn_.*=CUDA4" \
                                              -ot "blk\.(30|31|32)\.ffn_.*=CUDA5" \
                                              -ot exps=CPU \
                                              -b 4096 -ub 4096 \
                                              --no-mmap \
                                              --threads 24
```

Edit: There are some layers missing as I deleted the last one (8,14,18,23,28) from each card as i'm playing around with the context size and i was having OOM errors

---

👤 **RodriMora** commented the **2025-06-11** at **16:56:48**:<br>

> [@RodriMora](https://github.com/RodriMora) Can you tell me the command to run this bench please? Maybe I can try with Q3_K_XL and IQ3_K_R4. I guess you're using a quite big ubatch size?

The -ot are specific for my setup, the CUDA2 and CUDA4 are the 5090s. 0,1,3,5 are the 3090s
```

CUDA_VISIBLE_DEVICES="2,4,0,1,3,5" \
                                             ./build/bin/llama-sweep-bench \
                                              --model /mnt/llms/models/ubergarm/DeepSeek-V3-0324-GGUF/DeepSeek-V3-0324-IQ2_K_R4/DeepSeek-V3-0324-IQ2_K_R4-00001-of-00005.gguf \
                                              --alias ubergarm/DeepSeek-V3-0324-IQ2_K_R4 -mla 3 -fa \
                                              -amb 512 \
                                              -fmoe \
                                              -ctk f16 \
                                              -c 16384 \
                                              -ngl 99 \
                                              -ot "blk\.(3|4|5|6|7)\.ffn_.*=CUDA0" \
                                              -ot "blk\.(9|10|11|12|13)\.ffn_.*=CUDA1" \
                                              -ot "blk\.(15|16|17)\.ffn_.*=CUDA2" \
                                              -ot "blk\.(20|21|22)\.ffn_.*=CUDA3" \
                                              -ot "blk\.(25|26|27)\.ffn_.*=CUDA4" \
                                              -ot "blk\.(30|31|32)\.ffn_.*=CUDA5" \
                                              -ot exps=CPU \
                                              -b 4096 -ub 4096 \
                                              --no-mmap \
                                              --threads 24
```

---

👤 **Panchovix** commented the **2025-06-11** at **19:54:21**:<br>

Okay I noticed something on ikllamacpp vs llamacpp

When offloading semi layers on ikllamacpp, (for example:)
```
-ot "blk.31.ffn_(norm|gate_inp|gate_shexp|down_shexp|up_shexp).weight=CUDA1" \
-ot "blk.31.ffn_gate_exps.weight=CUDA1" \
-ot "blk.31.ffn_down_exps.weight=CUDA2" \
-ot "blk.32.ffn_(norm|gate_inp|gate_shexp|down_shexp|up_shexp).weight=CUDA0" \
-ot "blk.32.ffn_gate_exps.weight=CUDA0" \
```
TG t/s tanks (1.5 t/s vs 7.5 t/s). This doesn't seem to happen with normal llamacpp. 

PP t/s are similar. I have created a new issue https://github.com/ikawrakow/ik_llama.cpp/issues/521

@RodriMora thanks for the command! It helped me to confirm this and also see perf in general.

---

👤 **mtcl** commented the **2025-06-12** at **05:54:13**:<br>

I got 2x5090s and they got perfectly in my system.  nowi just need to sell my 2x4090 and 1x3090. 😂

---

👤 **ikawrakow** commented the **2025-06-14** at **12:00:38**:<br>

I think we can close this.