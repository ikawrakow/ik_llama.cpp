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
    if (info.have_nccl && nhave == nreduce) { // somehow I'm not able to figure out how to use NCCL when not all GPUs participate in the reduce op
        GGML_ASSERT(info.have_nccl);
        GGML_ASSERT(info.device_count == nreduce);
        auto type = dst->type;
        //int device = ctx.device;
        if (nreduce != info.device_count) {
            GGML_ABORT("Not implemented");
        }
        //auto tim1 = std::chrono::steady_clock::now();
        auto data_type = type == GGML_TYPE_F32 ? ncclFloat : ncclHalf;
        if (nreduce == 4 && dst->ne[1] > 32) {
            auto com = info.nccl_coms + info.device_count;
            static const int devs[8] = {0,1, 2,3, 0,2, 1,3};
            for (int ip = 0; ip < 4; ++ip) {
                ncclGroupStart();
                ggml_cuda_set_device(devs[2*ip+0]);
                auto status1 = ncclAllReduce(dst->src[devs[2*ip+0]]->data, dst->src[devs[2*ip+0]]->data,
                        ggml_nelements(dst), data_type, ncclSum, com[2*ip+0], info.all_ctx[devs[2*ip+0]]->stream());
                ggml_cuda_set_device(devs[2*ip+1]);
                auto status2 = ncclAllReduce(dst->src[devs[2*ip+1]]->data, dst->src[devs[2*ip+1]]->data,
                        ggml_nelements(dst), data_type, ncclSum, com[2*ip+1], info.all_ctx[devs[2*ip+1]]->stream());
                ncclGroupEnd();
                if (status1 != ncclSuccess || status2 != ncclSuccess) {
                    fprintf(stderr, "%s: ncclAllReduce failed with statuses %d, %d\n", __func__, (int)status1, (int)status2);
                    GGML_ABORT("Fatal error");
                }
            }
        }
        else if (nreduce == 3 && dst->ne[1] > 32) {
            auto com = info.nccl_coms + info.device_count;
            static const int devs[4] = {0,1, 0,2};
            for (int ip = 0; ip < 2; ++ip) {
                ncclGroupStart();
                ggml_cuda_set_device(devs[2*ip+0]);
                auto status1 = ncclAllReduce(dst->src[devs[2*ip+0]]->data, dst->src[devs[2*ip+0]]->data,
                        ggml_nelements(dst), data_type, ncclSum, com[2*ip+0], info.all_ctx[devs[2*ip+0]]->stream());
                ggml_cuda_set_device(devs[2*ip+1]);
                auto status2 = ncclAllReduce(dst->src[devs[2*ip+1]]->data, dst->src[devs[2*ip+1]]->data,
                        ggml_nelements(dst), data_type, ncclSum, com[2*ip+1], info.all_ctx[devs[2*ip+1]]->stream());
                ncclGroupEnd();
                if (status1 != ncclSuccess || status2 != ncclSuccess) {
                    fprintf(stderr, "%s: ncclAllReduce failed with statuses %d, %d\n", __func__, (int)status1, (int)status2);
                    GGML_ABORT("Fatal error");
                }
            }
            ncclGroupStart();
            ggml_cuda_set_device(0);
            auto status1 = ncclSend(dst->src[0]->data, ggml_nelements(dst), data_type, 1, com[0], info.all_ctx[0]->stream());
            ggml_cuda_set_device(1);
            auto status2 = ncclRecv(dst->src[1]->data, ggml_nelements(dst), data_type, 0, com[1], info.all_ctx[1]->stream());
            ncclGroupEnd();
            if (status1 != ncclSuccess || status2 != ncclSuccess) {
                fprintf(stderr, "%s: ncclSend/Recv failed with statuses %d, %d\n", __func__, (int)status1, (int)status2);
                GGML_ABORT("Fatal error");
            }
        }
        else {
            ncclGroupStart();
            for (int i = 0; i < nreduce; ++i) {
                ncclComm_t this_comm;
                if (nhave == nreduce) {
                    this_comm = info.nccl_coms[i];
                } else {
                    auto status = ncclCommSplit(info.nccl_coms[i], dst->src[i] ? 0 : NCCL_SPLIT_NOCOLOR, i, &this_comm, NULL);
                    GGML_ASSERT(status == ncclSuccess);
                }
                ggml_cuda_set_device(i);
                auto stream = info.all_ctx[i]->stream();
                GGML_ASSERT(stream);
                auto status = ncclAllReduce(dst->src[i] ? dst->src[i]->data : nullptr,
                        dst->src[i] ? dst->src[i]->data : nullptr,
                        ggml_nelements(dst), data_type, ncclSum, this_comm, stream);
                if (status != ncclSuccess) {
                    fprintf(stderr, "%s: ncclAllReduce failed with status %d\n", __func__, (int)status);
                    GGML_ABORT("Fatal error");
                }
            }
            ncclGroupEnd();
        }
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
            ggml_cuda_set_device(i);
            int nblocks = (nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
            copy_task task;
            task.nptr = nhave/2;
            task.nelem = nelem;
            task.ptrs[0] = (char *)dst->src[i]->data;
            int j = idx[2*ii+1];
            CUDA_CHECK(cudaEventRecord(info.all_ctx[j]->copy_event, info.all_ctx[j]->stream()));
            task.ptrs[1] = (char *)dst->src[j]->data;
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
            ggml_cuda_set_device(i);
            int nblocks = (nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
            copy_task task;
            task.nptr = nhave/2;
            task.nelem = nelem;
            task.ptrs[0] = (char *)dst->src[i]->data;
            int j = idx[(2*ii+2)%nhave];
            task.ptrs[1] = (char *)dst->src[j]->data;
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
            CUDA_CHECK(cudaEventRecord(info.all_ctx[i]->copy_event, info.all_ctx[i]->stream()));
        }
        //printf("Recorded events again\n");
        for (int ii = 0; ii < nhave; ++ii) {
            int i = idx[ii];
            for (int jj = 0; jj < nhave; ++jj) {
                if (jj == ii) continue;
                int j = idx[jj];
                CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[i]->stream(), info.all_ctx[j]->copy_event));
            }
        }
        //printf("All good so far\n");
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
