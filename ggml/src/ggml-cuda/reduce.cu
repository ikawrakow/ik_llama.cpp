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

    auto op = (ggml_op)dst->op_params[0];
    GGML_ASSERT(op == GGML_OP_ADD);
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    auto extra = (ggml_split_tensor_t *)dst->extra;
    GGML_ASSERT(extra && extra->n_device > 1);

    printf("============================== %s on device %d\n", __func__, ctx.device);

#ifdef GGML_USE_NCCL
    auto & info = ggml_cuda_info();
    GGML_ASSERT(info.have_nccl);
    GGML_ASSERT(info.device_count >= extra->n_device);
    int nhave = 0;
    auto type = dst->type;
    for (int j = 0; j < extra->n_device; ++j) {
        if (extra->splits[j]) {
            GGML_ASSERT(extra->splits[j]->type == type);
            GGML_ASSERT(ggml_are_same_shape(dst, extra->splits[j]));
            ++nhave;
        }
    }
    int device = ctx.device;
    ncclComm_t this_comm;
    if (nhave == info.device_count) {
        this_comm = info.nccl_coms[device];
    } else {
        int color = extra->splits[device] ? 1 : 0;
        auto status = ncclCommSplit(info.nccl_coms[0], color, ctx.device, &this_comm, nullptr);
        GGML_ASSERT(status == ncclSuccess);
    }
    GGML_ASSERT(this_comm);
    ncclResult_t status;
    if (type == GGML_TYPE_F32) {
        status = ncclAllReduce(extra->splits[device]->data,
                               extra->splits[device]->data,
                               ggml_nelements(extra->splits[device]),
                               ncclFloat, ncclSum, this_comm, ctx.stream());
    } else {
        status = ncclAllReduce(extra->splits[device]->data,
                               extra->splits[device]->data,
                               ggml_nelements(extra->splits[device]),
                               ncclHalf, ncclSum, this_comm, ctx.stream());
    }
    if (status != ncclSuccess) {
        fprintf(stderr, "%s: ncclAllReduce failed with status %d\n", __func__, (int)status);
        GGML_ABORT("Fatal error");
    }
    return;
#endif
    fprintf(stderr, "%s: not implemented without NCCL\n", __func__);
    GGML_ABORT("Fatal error");
}
