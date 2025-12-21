//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "reduce.cuh"
#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif

void ggml_cuda_op_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    int idx = dst->op_params[0];
    int nreduce = dst->op_params[1];
    auto op = (ggml_op)dst->op_params[2];
    GGML_ASSERT(op == GGML_OP_ADD);
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(idx >= 0 && nreduce > 1 && idx < nreduce);

    //printf("============================== %s on device %d\n", __func__, ctx.device);

#ifdef GGML_USE_NCCL
    auto & info = ggml_cuda_info();
    GGML_ASSERT(info.have_nccl);
    GGML_ASSERT(info.device_count >= nreduce);
    auto type = dst->type;
    int device = ctx.device;
    ncclComm_t this_comm;
    if (nreduce == info.device_count) {
        this_comm = info.nccl_coms[device];
    } else {
        GGML_ABORT("Not implemented");
        //int color = extra->splits[device] ? 1 : 0;
        //auto status = ncclCommSplit(info.nccl_coms[device], color, ctx.device, &this_comm, nullptr);
        //GGML_ASSERT(status == ncclSuccess);
    }
    GGML_ASSERT(this_comm);
    if (idx == 0) {
        ncclGroupStart();
    }
    ncclResult_t status;
    if (type == GGML_TYPE_F32) {
        status = ncclAllReduce(dst->src[0]->data,
                               dst->data,
                               ggml_nelements(dst),
                               ncclFloat, ncclSum, this_comm, ctx.stream());
    } else {
        status = ncclAllReduce(dst->src[0]->data,
                               dst->data,
                               ggml_nelements(dst),
                               ncclHalf, ncclSum, this_comm, ctx.stream());
    }
    if (status != ncclSuccess) {
        fprintf(stderr, "%s: ncclAllReduce failed with status %d\n", __func__, (int)status);
        GGML_ABORT("Fatal error");
    }
    if (idx == nreduce-1) {
        ncclGroupEnd();
    }
    return;
#endif
    fprintf(stderr, "%s: not implemented without NCCL\n", __func__);
    GGML_ABORT("Fatal error");
}
