### 🗣️ [#619](https://github.com/ikawrakow/ik_llama.cpp/discussions/619) - gpu p2p utilization

| **Author** | `magikRUKKOLA` |
| :--- | :--- |
| **Created** | 2025-07-16 |
| **Updated** | 2025-07-17 |

---

#### Description

Is there any mode of the llm inference in ik_llama.cpp that utilizes the p2p functionality between the GPUs?  That would include the NVLINKs and, most importantly, the regular p2p master-slave functionality as enabled by the opensource nvidia drivers (see https://github.com/aikitoria/open-gpu-kernel-modules ).

[EDIT]:

with and without p2p functionality:

```bash
/usr/share/doc/nvidia-cuda-toolkit/examples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest

Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1      2
     0 839.83  14.54  16.64
     1  14.53 839.83  16.67
     2  16.72  16.67 840.26
Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1      2
     0 839.15  52.04  52.04
     1  52.04 839.83  52.03
     2  51.94  52.03 839.83
```

So there is about 35 GB/s free bandwidth available for the nvidia gpu users.

[EDIT]:
If I am reading the code correctly, the p2p functionality is used only at: ggml_backend_sycl_graph_compute and the ggml_sycl_set_peer_access is allowing it only if n_tokens is less than 128?  Can anyone provide more info?

[EDIT2]:
Uh oh?

```
4415 //todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
4416 #if 0
4417         SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(
4418             (char *)dst->data, (const char *)src->data, size).wait()));
4419
4420         /*
4421         DPCT1009:201: SYCL uses exceptions to report errors and does not use the
4422         error codes. The original code was commented out and a warning string
4423         was inserted. You need to rewrite this code.
4424         */
4425         SYCL_CHECK(CHECK_TRY_ERROR(
4426             dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
4427 #endif
```