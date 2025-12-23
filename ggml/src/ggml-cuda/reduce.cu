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

void ggml_cuda_op_reduce([[maybe_unused]] ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    auto op = (ggml_op)dst->op_params[0];
    GGML_ASSERT(op == GGML_OP_ADD);
    int nreduce = dst->op_params[1];
    int nhave   = dst->op_params[2];
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(nhave >=2 && nhave <= nreduce);

    //printf("============================== %s on device %d with %d sources\n", __func__, ctx.device, nreduce);

    auto & info = ggml_cuda_info();
#ifdef GGML_USE_NCCL
    if (info.have_nccl) {
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
                auto status = ncclCommSplit(info.nccl_coms[i], dst->src[i] ? 1 : 0, i, &this_comm, NULL);
                GGML_ASSERT(status == ncclSuccess);
            }
            ggml_cuda_set_device(i);
            auto stream = info.all_ctx[i]->stream();
            GGML_ASSERT(stream);
            ncclResult_t status;
            if (type == GGML_TYPE_F32) {
                status = ncclAllReduce(dst->src[i] ? dst->src[i]->data : nullptr,
                        dst->src[i] ? dst->src[i]->data : nullptr,
                        ggml_nelements(dst),
                        ncclFloat, ncclSum, this_comm, stream);
            } else {
                status = ncclAllReduce(dst->src[i] ? dst->src[i]->data : nullptr,
                        dst->src[i] ? dst->src[i]->data : nullptr,
                        ggml_nelements(dst),
                        ncclHalf, ncclSum, this_comm, stream);
            }
            if (status != ncclSuccess) {
                fprintf(stderr, "%s: ncclAllReduce failed with status %d\n", __func__, (int)status);
                GGML_ABORT("Fatal error");
            }
        }
        ncclGroupEnd();
    }
    ggml_cuda_set_device(ctx.device);
    //auto tim2 = std::chrono::steady_clock::now();
    //printf("%s: launched in %g us\n", __func__, 1e-3*std::chrono::duration_cast<std::chrono::nanoseconds>(tim2-tim1).count());
    return;
    }
#endif
    //auto tim1 = std::chrono::steady_clock::now();
    //GGML_ASSERT(nhave == nreduce);
    GGML_ASSERT(dst->data == dst->src[ctx.device]->data);
    auto nbytes = ggml_nbytes(dst);
    if (nhave == 2 && (nhave == nreduce || dst->ne[1] <= 8)) {
        int idx[2];
        int ii = 0;
        for (int i = 0; i < nreduce; ++i) {
            if (dst->src[i]) {
                idx[ii++] = i;
            }
        }
        // With P2P access enabled, we can access peer memory so as if it was local.
        // Hence, we can launch two reduce kernels, one on each device, each kernel
        // processing half of the data. This very simply approach almost matches NCCL
        // performance (I see ~1% lower PP and TG performance on my 2x3090 system).
        for (int i = 0; i < nhave; ++i) {
            GGML_ASSERT(dst->src[idx[i]]->type == dst->type);
            GGML_ASSERT(ggml_are_same_shape(dst, dst->src[idx[i]]));
            ggml_cuda_set_device(idx[i]);
            if (!info.all_ctx[idx[i]]->copy_event) {
                CUDA_CHECK(cudaEventCreateWithFlags(&info.all_ctx[idx[i]]->copy_event, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventRecord(info.all_ctx[idx[i]]->copy_event, info.all_ctx[idx[i]]->stream()));
        }
        auto nelem = ggml_nelements(dst);
        auto nelem_half = (nelem + 1)/2;
        for (int i = 0; i < nhave; ++i) {
            ggml_cuda_set_device(idx[i]);
            CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[idx[i]]->stream(), info.all_ctx[idx[(i+1)%2]]->copy_event, 0));
            auto this_nelem = std::min(nelem_half, nelem - nelem_half);
            int nblock = (this_nelem + CUDA_REDUCE_BLOCK_SIZE - 1)/CUDA_REDUCE_BLOCK_SIZE;
            if (dst->type == GGML_TYPE_F16) {
                auto src_ptr = (half *)dst->src[idx[i]]->data + i*nelem_half;
                auto dst_ptr = (half *)dst->src[idx[(i+1)%2]]->data + i*nelem_half;
                k_add_sym<half, CUDA_REDUCE_BLOCK_SIZE><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[idx[i]]->stream()>>>(this_nelem, src_ptr, dst_ptr);
            } else {
                auto src_ptr = (float *)dst->src[idx[i]]->data + i*nelem_half;
                auto dst_ptr = (float *)dst->src[idx[(i+1)%2]]->data + i*nelem_half;
                k_add_sym<float, CUDA_REDUCE_BLOCK_SIZE><<<nblock, CUDA_REDUCE_BLOCK_SIZE, 0, info.all_ctx[idx[i]]->stream()>>>(this_nelem, src_ptr, dst_ptr);
            }
        }
        for (int i = 0; i < nhave; ++i) {
            ggml_cuda_set_device(idx[i]);
            CUDA_CHECK(cudaEventRecord(info.all_ctx[idx[i]]->copy_event, info.all_ctx[idx[i]]->stream()));
            ggml_cuda_set_device(idx[(i+1)%2]);
            CUDA_CHECK(cudaStreamWaitEvent(info.all_ctx[idx[(i+1)%2]]->stream(), info.all_ctx[idx[i]]->copy_event));
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
    //auto tim2 = std::chrono::steady_clock::now();
    //printf("%s: launched in %g us\n", __func__, 1e-3*std::chrono::duration_cast<std::chrono::nanoseconds>(tim2-tim1).count());
    //fprintf(stderr, "%s: not implemented without NCCL\n", __func__);
    //GGML_ABORT("Fatal error");
}
