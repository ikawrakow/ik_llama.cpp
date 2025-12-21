//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "reduce.cuh"

void ggml_cuda_op_reduce([[maybe_unused]] ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    auto op = (ggml_op)dst->op_params[0];
    GGML_ASSERT(op == GGML_OP_ADD);
    int nreduce = dst->op_params[1];
    int nhave   = dst->op_params[2];
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(nhave >=2 && nhave <= nreduce);

    //printf("============================== %s on device %d\n", __func__, ctx.device);

#ifdef GGML_USE_NCCL
    auto & info = ggml_cuda_info();
    GGML_ASSERT(info.have_nccl);
    GGML_ASSERT(info.device_count == nreduce);
    auto type = dst->type;
    //int device = ctx.device;
    if (nreduce != info.device_count) {
        GGML_ABORT("Not implemented");
    }
    ncclGroupStart();
    for (int i = 0; i < nreduce; ++i) {
        ncclComm_t this_comm;
        if (nhave == nreduce) {
            this_comm = info.nccl_coms[i];
        } else {
            auto status = ncclCommSplit(info.nccl_coms[i], dst->src[i] ? 1 : 0, i, &this_comm, NULL);
            GGML_ASSERT(status == ncclSuccess);
        }
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
    return;
#endif
    fprintf(stderr, "%s: not implemented without NCCL\n", __func__);
    GGML_ABORT("Fatal error");
}
