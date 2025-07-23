### 🐛 [#499](https://github.com/ikawrakow/ik_llama.cpp/issues/499) - Bug: cache quantization crash with IQK_FORCE_BF16

| **Author** | `randoentity` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2025-06-06 |
| **Updated** | 2025-06-07 |

---

#### Description

### What happened?

Using `DGGML_CUDA_IQK_FORCE_BF16=1` in combination with `--cache-type-k q8_0` results in the error below.
Turning either off does not raise an error.
`--cache-type-v` doesn't seem to do anything for this model.

```sh
cmake -B ./${BUILD_DIR} -DGGML_CUDA=ON -DGGML_RPC=OFF -DGGML_SCHED_MAX_COPIES=1 -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA_IQK_FORCE_BF16=1  -DGGML_BLAS=OFF
```

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,0,1 ./build_bf16/bin/llama-sweep-bench \
--attention-max-batch 64 \
--batch-size 4096 \
--ubatch-size 4096 \
--cache-type-k q8_0 \
--cache-type-v q8_0 \
--ctx-size 32768 \
--flash-attn \
--fused-moe \
--mla-use 3 \
--model /mnt/x/models/ubergarm/dsr1-0528-iq1-s4/DeepSeek-R1-0528-IQ1_S_R4-00001-of-00003.gguf \
--n-gpu-layers 99 \
--override-tensor "blk\.(16|17|18|19|20|21|22|23|24)\.ffn_.*=CUDA1" \
--override-tensor "blk\.(3|4|5|6)\.ffn_.*=CUDA0" \
--override-tensor "blk\.(7|8|9|10|11|12|13|14|15)\.ffn_.*=CUDA2" \
--override-tensor exps=CPU,attn_kv_b=CPU \
--tensor-split 100,1,1 \
--threads 6 \
--threads-batch 12 \
--min_p 0.01 \
--temp 0.6 \
--top_p 0.95 \
--warmup-batch
```

### Name and Version

build_bf16/bin/llama-sweep-bench --version
version: 3730 (ffd87f28)
built with cc (Gentoo 14.2.1_p20241221 p7) 14.2.1 20241221 for x86_64-pc-linux-gnu

### What operating system are you seeing the problem on?

Linux

### Relevant log output

```shell
main: n_kv_max = 32768, n_batch = 4096, n_ubatch = 4096, flash_attn = 1, n_gpu_layers = 99, n_threads = 6, n_threads_batch = 12

|    PP |     TG |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |
|-------|--------|--------|----------|----------|----------|----------|
/mnt/x/ik_llama.cpp/ggml/src/ggml-cuda.cu:1286: GGML_ASSERT(to_bf16_cuda != nullptr) failed
[New LWP 8409]
[New LWP 8408]
[New LWP 8407]
[New LWP 8406]
[New LWP 8332]
[New LWP 8331]
[New LWP 7938]
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/usr/lib64/libthread_db.so.1".
0x00007fae703158a7 in wait4 () from /usr/lib64/libc.so.6
#0  0x00007fae703158a7 in wait4 () from /usr/lib64/libc.so.6
#1  0x0000564ac2e60592 in ggml_abort ()
#2  0x0000564ac2f166fa in ggml_cuda_op_mul_mat_cublas(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char
 const*, float*, long, long, long, long, CUstream_st*) ()
#3  0x0000564ac2f09c8b in ggml_cuda_op_mul_mat(ggml_backend_cuda_context&, ggml_tensor const*, ggml_tensor const*, ggml_tensor*, void (*)(ggml_backend_cuda_context&, g
gml_tensor const*, ggml_tensor const*, ggml_tensor*, char const*, float const*, char const*, float*, long, long, long, long, CUstream_st*), void (*)(float const*, void
*, long, long, long, long, ggml_type, CUstream_st*)) [clone .constprop.0] ()
#4  0x0000564ac2f1e58b in ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) ()
#5  0x0000564ac2ebce93 in ggml_backend_sched_compute_splits ()
#6  0x0000564ac2d79e5a in llama_decode ()
#7  0x0000564ac2cd6920 in main::{lambda(llama_context*, llama_batch&, int)#1}::operator()(llama_context*, llama_batch&, int) const [clone .isra.0] ()
#8  0x0000564ac2c792ec in main ()
[Inferior 1 (process 7937) detached]
```

---

#### 💬 Conversation

👤 **Thireus** commented the **2025-06-06** at **15:04:29**:<br>

I can confirm the same issue occurs on q4_0 as well.

---

👤 **ikawrakow** commented the **2025-06-06** at **16:32:03**:<br>

Does #501 fix it?