//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "reduce.cuh"

#include <chrono>

template <typename T, int block_size>
static __global__ void k_add(int nelem, const T * src, T * dst) {
    int i = blockIdx.x*block_size + threadIdx.x;
    if (i >= nelem) return;
    dst[i] += src[i];
}

template <typename T, int block_size>
static __global__ void k_add_sym(int nelem, T * src, T * dst) {
    int i = blockIdx.x*block_size + threadIdx.x;
    if (i >= nelem) return;
    dst[i] += src[i];
    src[i] = dst[i];
}

struct copy_task {
    void * ptrs[GGML_CUDA_MAX_DEVICES];
    int nptr;
    int nelem;
};

template <typename T, int block_size>
static __global__ void k_reduce_add(copy_task task) {
    int i = blockIdx.x*block_size + threadIdx.x;
    if (i >= task.nelem) return;
    auto dst = (T *)task.ptrs[0];
    for (int j = 1; j < task.nptr; ++j) {
        auto src = (T *)task.ptrs[j];
        dst[i] += src[i];
    }
    for (int j = 1; j < task.nptr; ++j) {
        auto src = (T *)task.ptrs[j];
        src[i] = dst[i];
    }
}

template <typename T, int block_size, int nptr>
static __global__ void k_reduce_add_T(copy_task task) {
    int i = blockIdx.x*block_size + threadIdx.x;
    if (i >= task.nelem) return;
    auto dst = (T *)task.ptrs[0];
    #pragma unroll
    for (int j = 1; j < nptr; ++j) {
        auto src = (T *)task.ptrs[j];
        dst[i] += src[i];
    }
    #pragma unroll
    for (int j = 1; j < nptr; ++j) {
        auto src = (T *)task.ptrs[j];
        src[i] = dst[i];
    }
}

void ggml_cuda_op_reduce([[maybe_unused]] ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    auto op = (ggml_op)dst->op_params[0];
    GGML_ASSERT(op == GGML_OP_ADD);
    int nreduce = dst->op_params[1];
    int nhave   = dst->op_params[2];
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(nhave >=2 && nhave <= nreduce);
    if (dst->op_params[3] == 1) {
        // The dst tensor is just a container for the sources and the reduce op is turned off
        return;
    }

    auto & info = ggml_cuda_info();
#ifdef GGML_USE_NCCL
    // Somehow I'm not able to figure out how to use NCCL correctly.
    // It does not work at all if not all GPUs participate in the reduce op, and we
    // get suboptimal prompt processing performance when we have more than 2 GPUs.
    // Hence, if enabled, we use NCCL only for the cases where it works and performs well.
    if (info.have_nccl && nhave == nreduce && (nhave == 2 || dst->ne[1] < 32)) {
        GGML_ASSERT(info.have_nccl);
        GGML_ASSERT(info.device_count == nreduce);
        auto data_type = dst->type == GGML_TYPE_F32 ? ncclFloat : ncclHalf;
        ncclGroupStart();
        for (int i = 0; i < nreduce; ++i) {
            ggml_cuda_set_device(i);
            auto status = ncclAllReduce(dst->src[i] ? dst->src[i]->data : nullptr,
                    dst->src[i] ? dst->src[i]->data : nullptr,
                    ggml_nelements(dst), data_type, ncclSum, info.nccl_coms[i], info.all_ctx[i]->stream());
            if (status != ncclSuccess) {
                fprintf(stderr, "%s: ncclAllReduce failed with status %d\n", __func__, (int)status);
                GGML_ABORT("Fatal error");
            }
        }
        ncclGroupEnd();
        ggml_cuda_set_device(ctx.device);
        return;
    }
#endif
    GGML_ASSERT(dst->data == dst->src[ctx.device]->data);
    auto nbytes = ggml_nbytes(dst);
    int idx[GGML_CUDA_MAX_DEVICES];
    {
        int ii = 0;
        bool have_this_device = false;
        for (int i = 0; i < nreduce; ++i) {
            if (dst->src[i]) {
                idx[ii++] = i;
                if (i == ctx.device) have_this_device = true;
            }
        }
        GGML_ASSERT(ii == nhave);
        GGML_ASSERT(have_this_device);
    }
    //
    // For prompt processing) the objective is to minimize the amount of data being exchanged between
    // the GPUs, even if this means we need to launch a larger number of kernels (we are bandwidth
    // bound rather than latency bound).
    // The following implements a ring communication+reduction that achieves this goal.
    // I would have thought that this is automatically done by NCCL, but it doesn't look that
    // way (or I simply don't understand how to use NCCL) as the ring implementation bellow achieves quite a bit
    // better performance compared to what I get with NCCL.
    //
    // We do the data reduction in stages. Let's N be the number of GPUs.
    // In each stage, each GPU sends 1/N'th of the data to a peer GPU in a ring fashion
    // (i.e. 0->1, 1->2, 2->3, ..., N-1 ->0). Each GPU then performs the addition with the
    // portion just received. After N-1 stages, each GPU ends up having the full sum for 1/N'th
    // of the data. We then do a second round of N-1 stages where each GPU sends a fully reduced
    // portion to its peer. The following shows how all this works for 2, 3, and 4 GPUs:
    // Worth noting that because in each round each GPU sends and receives data, we use the
    // bidirectional p2p bandwidth, which tends to be 2X the unidirectional bandwidth.
    //
    // Examples
    //
    // ======================== 2 devices:
    // stage 0:
    //   i = 0, peer = 1, ichunk = 0 -> copy part 0 from device 1, add -> device 0 has part 0 complete
    //   i = 1, peer = 0, ichunk = 1 -> copy part 1 from device 0, add -> device 1 has part 1 complete
    // second loop
    // stage 0
    //   i = 0, peer = 1, ichunk = 1 -> copy part 1 from device 1 -> device 0 has parts 0, 1 complete
    //   i = 1, peer = 0, ichunk = 0 -> copy part 0 from device 0 -> device 1 has parts 0, 1 complete
    //
    // ======================== 3 devices
    // stage 0
    //   i = 0, peer = 1, ichunk = 0 -> copy part 0 from device 1, add -> part 0 = 0+1
    //   i = 1, peer = 2, ichunk = 1 -> copy part 1 from device 2, add -> part 1 = 1+2
    //   i = 2, peer = 0, ichunk = 2 -> copy part 2 from device 0, add -> part 2 = 0+2
    // stage 1
    //   i = 0, peer = 1, ichunk = 1 -> copy part 1 from device 1, add -> part 1 = 0+1+2
    //   i = 1, peer = 2, ichunk = 2 -> copy part 2 from device 2, add -> part 2 = 0+1+2
    //   i = 2, peer = 0, ichunk = 0 -> copy part 0 from device 0, add -> part 0 = 0+1+2
    // second loop
    // stage 0
    //   i = 0, peer = 1, ichunk = 2 -> copy part 2 from device 1, device 0 now has parts 1, 2 complete
    //   i = 1, peer = 2, ichunk = 0 -> copy part 0 from device 2, device 1 now has parts 0, 2 complete
    //   i = 2, peer = 0, ichunk = 1 -> copy part 1 from device 0, device 2 now has parts 0, 1 complete
    // stage 1
    //   i = 0, peer = 1, ichunk = 0 -> copy part 0 from device 1, device 0 now has parts 0, 1, 2, complete
    //   i = 1, peer = 2, ichunk = 1 -> copy part 1 from device 2, device 1 now has parts 0, 1, 2, complete
    //   i = 2, peer = 0, ichunk = 2 -> copy part 2 from device 0, device 2 now has parts 0, 1, 2, complete
    //
    // ======================== 4 devices
    // stage 0
    //   i = 0, peer = 1, ichunk = 0 -> copy part 0 from device 1, add -> part 0 = 0+1
    //   i = 1, peer = 2, ichunk = 1 -> copy part 1 from device 2, add -> part 1 = 1+2
    //   i = 2, peer = 3, ichunk = 2 -> copy part 2 from device 3, add -> part 2 = 2+3
    //   i = 3, peer = 0, ichunk = 3 -> copy part 3 from device 0, add -> part 3 = 0+3
    // stage 1
    //   i = 0, peer = 1, ichunk = 1 -> copy part 1 from device 1, add -> part 1 = 0+1+2
    //   i = 1, peer = 2, ichunk = 2 -> copy part 2 from device 2, add -> part 2 = 1+2+3
    //   i = 2, peer = 3, ichunk = 3 -> copy part 3 from device 3, add -> part 3 = 0+2+3
    //   i = 3, peer = 0, ichunk = 0 -> copy part 0 from device 0, add -> part 0 = 0+1+3
    // stage 2
    //   i = 0, peer = 1, ichunk = 2 -> copy part 2 from device 1, add -> part 2 = 0+1+2+3
    //   i = 1, peer = 2, ichunk = 3 -> copy part 3 from device 2, add -> part 3 = 0+1+2+3
    //   i = 2, peer = 3, ichunk = 0 -> copy part 0 from device 3, add -> part 0 = 0+1+2+3
    //   i = 3, peer = 0, ichunk = 1 -> copy part 1 from device 0, add -> part 1 = 0+1+2+3
    // second loop
    // stage 0
    //   i = 0, peer = 1, ichunk = 3 -> copy part 3 from device 1, device 0 now has parts 2, 3
    //   i = 1, peer = 2, ichunk = 0 -> copy part 0 from device 2, device 1 now has parts 3, 0
    //   i = 2, peer = 3, ichunk = 1 -> copy part 1 from device 3, device 2 now has parts 0, 1
    //   i = 3, peer = 0, ichunk = 2 -> copy part 2 from device 0, device 3 now has parts 1, 2
    // stage 1
    //   i = 0, peer = 1, ichunk = 0 -> copy part 0 from device 1, device 0 now has parts 0, 2, 3
    //   i = 1, peer = 2, ichunk = 1 -> copy part 1 from device 2, device 1 now has parts 3, 0, 1
    //   i = 2, peer = 3, ichunk = 2 -> copy part 2 from device 3, device 2 now has parts 0, 1, 2
    //   i = 3, peer = 0, ichunk = 3 -> copy part 3 from device 0, device 3 now has parts 1, 2, 3
    // stage 2
    //   i = 0, peer = 1, ichunk = 1 -> copy part 1 from device 1, device 0 now has parts 0, 1, 2, 3
    //   etc.
    //
    if (dst->ne[1] >= 32) {
        auto nelem = ggml_nelements(dst);
        auto elem_size = ggml_element_size(dst);
        auto nelem_per_device = (nelem + nhave - 1)/nhave;
        auto required_size = nelem_per_device*elem_size;
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            auto this_ctx = info.all_ctx[i];
            if (!this_ctx->copy_event) {
                ggml_cuda_set_device(this_ctx->device);
                CUDA_CHECK(cudaEventCreateWithFlags(&this_ctx->copy_event, cudaEventDisableTiming));
            }
            if (required_size > this_ctx->copy_size) {
                ggml_cuda_set_device(this_ctx->device);
                if (this_ctx->copy_buffer) {
                    CUDA_CHECK(cudaFree(this_ctx->copy_buffer));
                }
                CUDA_CHECK(ggml_cuda_device_malloc(&this_ctx->copy_buffer, required_size, this_ctx->device));
                this_ctx->copy_size = required_size;
            }
        }
        for (int stage = 0; stage < nhave-1; ++stage) {
            int ichunk = stage;
            for (int ii = 0; ii < nhave; ++ii) {
                int i = idx[ii];
                int peer = idx[(ii+1)%nhave];
                auto this_nelem = std::min(nelem_per_device, nelem - ichunk*nelem_per_device);
                ggml_cuda_set_device(info.all_ctx[peer]->device);
                CUDA_CHECK(cudaMemcpyPeerAsync(info.all_ctx[i]->copy_buffer, info.all_ctx[i]->device,
                            (const char *)dst->src[peer]->data + ichunk*nelem_per_device*elem_size, info.all_ctx[peer]->device,
                            this_nelem*elem_size, info.all_ctx[peer]->stream()));
                CUDA_CHECK(cudaEventRecord(info.all_ctx[peer]->copy_event, info.all_ctx[peer]->stream()));
                ggml_cuda_set_device(info.all_ctx[i]->device);
                CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[peer]->copy_event, 0));
                int num_blocks = (this_nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
                if (dst->type == GGML_TYPE_F16) {
                    k_add<half, CUDA_REDUCE_BLOCK_SIZE><<<num_blocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(this_nelem,
                            (const half *)info.all_ctx[i]->copy_buffer, (half *)dst->src[i]->data + ichunk*nelem_per_device);
                } else {
                    k_add<float, CUDA_REDUCE_BLOCK_SIZE><<<num_blocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(this_nelem,
                            (const float *)info.all_ctx[i]->copy_buffer, (float *)dst->src[i]->data + ichunk*nelem_per_device);
                }
                ichunk = (ichunk + 1)%nhave;
            }
        }
        for (int stage = 0; stage < nhave-1; ++stage) {
            int ichunk = (nhave - 1 + stage)%nhave;
            for (int ii = 0; ii < nhave; ++ii) {
                int i = idx[ii];
                int peer = idx[(ii+1)%nhave];
                auto this_nelem = std::min(nelem_per_device, nelem - ichunk*nelem_per_device);
                ggml_cuda_set_device(info.all_ctx[peer]->device);
                CUDA_CHECK(cudaMemcpyPeerAsync((char *)dst->src[i]->data + ichunk*nelem_per_device*elem_size, info.all_ctx[i]->device,
                            (const char *)dst->src[peer]->data + ichunk*nelem_per_device*elem_size, info.all_ctx[peer]->device,
                            this_nelem*elem_size, info.all_ctx[peer]->stream()));
                CUDA_CHECK(cudaEventRecord(info.all_ctx[peer]->copy_event, info.all_ctx[peer]->stream()));
                ggml_cuda_set_device(info.all_ctx[i]->device);
                CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[peer]->copy_event, 0));
                ichunk = (ichunk + 1)%nhave;
            }
        }
        ggml_cuda_set_device(ctx.device);
        return;
    }
    if (nhave == 4 && dst->ne[1] <= 8 && ctx.p2p_enabled) {
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            GGML_ASSERT(dst->src[i]->type == dst->type);
            GGML_ASSERT(ggml_are_same_shape(dst, dst->src[i]));
            ggml_cuda_set_device(i);
            if (!info.all_ctx[i]->copy_event) {
                CUDA_CHECK(cudaEventCreateWithFlags(&info.all_ctx[i]->copy_event, cudaEventDisableTiming));
            }
        }
        auto nelem = ggml_nelements(dst);
        for (int ii = 0; ii < nhave/2; ++ii) {
            int i = idx[2*ii+0];
            int nblocks = (nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
            copy_task task;
            task.nptr = nhave/2;
            task.nelem = nelem;
            task.ptrs[0] = (char *)dst->src[i]->data;
            int j = idx[2*ii+1];
            ggml_cuda_set_device(j);
            CUDA_CHECK(cudaEventRecord(info.all_ctx[j]->copy_event, info.all_ctx[j]->stream()));
            task.ptrs[1] = (char *)dst->src[j]->data;
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[j]->copy_event));
            if (dst->type == GGML_TYPE_F16) {
                k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
            } else {
                k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
            }
        }
        for (int ii = 0; ii < nhave/2; ++ii) {
            int i = idx[2*ii+0];
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->copy_event, info.all_ctx[i]->stream()));
        }
        for (int ii = 0; ii < nhave/2; ++ii) {
            int i = idx[2*ii+1];
            int nblocks = (nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
            copy_task task;
            task.nptr = nhave/2;
            task.nelem = nelem;
            task.ptrs[0] = (char *)dst->src[i]->data;
            int j = idx[(2*ii+2)%nhave];
            task.ptrs[1] = (char *)dst->src[j]->data;
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[j]->copy_event));
            if (dst->type == GGML_TYPE_F16) {
                k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
            } else {
                k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblocks, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
            }
        }
        for (int ii = 0; ii < nhave/2; ++ii) {
            int i = idx[2*ii+1];
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->copy_event, info.all_ctx[i]->stream()));
        }
        for (int ii = 0; ii < nhave/2; ++ii) {
            int i = idx[(2*ii+2)%nhave];
            ggml_cuda_set_device(i);
            int j = idx[2*ii+1];
            CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[j]->copy_event));
        }
        ggml_cuda_set_device(ctx.device);
        return;
    }
    if (dst->ne[1] <= 8 && ctx.p2p_enabled) {
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            GGML_ASSERT(dst->src[i]->type == dst->type);
            GGML_ASSERT(ggml_are_same_shape(dst, dst->src[i]));
            ggml_cuda_set_device(i);
            if (!info.all_ctx[i]->copy_event) {
                CUDA_CHECK(cudaEventCreateWithFlags(&info.all_ctx[i]->copy_event, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->copy_event, info.all_ctx[i]->stream()));
        }
        //printf("Recorded events\n");
        auto nelem = ggml_nelements(dst);
        auto nelem_per_device = (nelem + nhave - 1)/nhave;
        auto elem_size = ggml_element_size(dst);
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            ggml_cuda_set_device(i);
            int this_nelem = std::min(nelem_per_device, nelem - ii*nelem_per_device);
            copy_task task;
            task.nptr = nhave;
            task.nelem = this_nelem;
            task.ptrs[0] = (char *)dst->src[i]->data + ii*nelem_per_device*elem_size;
            int k = 1;
            for (int jj = 0; jj < nhave; ++jj) {
                if (jj == ii) continue;
                int j = idx[jj];
                CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[j]->copy_event));
                task.ptrs[k++] = (char *)dst->src[j]->data + ii*nelem_per_device*elem_size;
            }
            int nblock = (this_nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
            if (dst->type == GGML_TYPE_F16) {
                switch (nhave) {
                    case 2:
                        k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                        break;
                    case 3:
                        k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 3><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                        break;
                    case 4:
                        k_reduce_add_T<half, CUDA_REDUCE_BLOCK_SIZE, 4><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                        break;
                    default:
                        k_reduce_add<half, CUDA_REDUCE_BLOCK_SIZE><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                }
            } else {
                switch (nhave) {
                    case 2:
                        k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 2><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                        break;
                    case 3:
                        k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 3><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                        break;
                    case 4:
                        k_reduce_add_T<float, CUDA_REDUCE_BLOCK_SIZE, 4><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                        break;
                    default:
                        k_reduce_add<float, CUDA_REDUCE_BLOCK_SIZE><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[i]->stream()>>>(task);
                }
            }
        }
        //printf("Submitted kernels\n");
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->copy_event, info.all_ctx[i]->stream()));
        }
        //printf("Recorded events again\n");
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            ggml_cuda_set_device(i);
            for (int jj = 0; jj < nhave; ++jj) {
                if (jj == ii) continue;
                int j = idx[jj];
                CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[j]->copy_event));
            }
        }
        ggml_cuda_set_device(ctx.device);
        return;
    }
    auto required_size = nbytes*(nhave-1);
    if (required_size > ctx.copy_size) {
        if (ctx.copy_buffer) {
            CUDA_CHECK(cudaFree(ctx.copy_buffer));
        }
        CUDA_CHECK(ggml_cuda_device_malloc(&ctx.copy_buffer, required_size, ctx.device));
        ctx.copy_size = required_size;
    }
    auto ptr = (char *)ctx.copy_buffer;
    for (int ii = 0; ii < nhave; ++ii) {
        int i = idx[ii];
        GGML_ASSERT(dst->src[i]->type == dst->type);
        GGML_ASSERT(ggml_are_same_shape(dst, dst->src[i]));
        if (i == ctx.device) continue;
        ggml_cuda_set_device(i);
        CUDA_CHECK(cudaMemcpyPeerAsync(ptr, ctx.device, dst->src[i]->data, i, nbytes, info.all_ctx[i]->stream()));
        if (!info.all_ctx[i]->copy_event) {
            CUDA_CHECK(cudaEventCreateWithFlags(&info.all_ctx[i]->copy_event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->copy_event, info.all_ctx[i]->stream()));
        ptr += nbytes;
    }
    auto nelem = ggml_nelements(dst);
    int num_blocks = (nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
    ggml_cuda_set_device(ctx.device);
    ptr = (char *)ctx.copy_buffer;
    for (int ii = 0; ii < nhave; ++ii) {
        int i = idx[ii];
        if (i == ctx.device) continue;
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), info.all_ctx[i]->copy_event, 0));
        if (dst->type == GGML_TYPE_F16) {
            k_add<half, CUDA_REDUCE_BLOCK_SIZE><<<num_blocks, CUDA_REDUCE_BLOCK_SIZE, 0, ctx.stream()>>>(nelem, (const half *)ptr, (half *)dst->data);
        } else {
            k_add<float, CUDA_REDUCE_BLOCK_SIZE><<<num_blocks, CUDA_REDUCE_BLOCK_SIZE, 0, ctx.stream()>>>(nelem, (const float *)ptr, (float *)dst->data);
        }
        ptr += nbytes;
    }
    if (!ctx.copy_event) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.copy_event, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(ctx.copy_event, ctx.stream()));
    for (int ii = 0; ii < nhave; ++ii) {
        int i = idx[ii];
        if (i == ctx.device) continue;
        ggml_cuda_set_device(i);
        CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), ctx.copy_event, 0));
        CUDA_CHECK(cudaMemcpyPeerAsync(dst->src[i]->data, i, dst->data, ctx.device, nbytes, info.all_ctx[i]->stream()));
    }
    ggml_cuda_set_device(ctx.device);
}
