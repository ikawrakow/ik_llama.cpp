#include <algorithm>
#include <cstdint>

#include "exp-cache-classify.cuh"
#include "common.cuh"

// Phase 4 expert cache (M3e): device-side routing classify.
//
// Maps the routed top-k ids through the layer's device-resident remap table and
// emits one of the three mask components. The integer logic MUST stay
// bit-identical to the host classify (llama_expert_cache_classify_host in
// llama.cpp) and its scalar CPU reference (ggml.c) — the mask feeds the MoE
// numerics and the slot assignment feeds the mmq_id shared-memory sizing rule
// (misses get distinct free slots per token over the <= 64-slot domain, pending
// slots excluded).
//
// remap/pending mutate only at TG step boundaries and are re-uploaded there,
// so within a step these tables are read-only.
//
// One thread per token column; the scan is serial per column (k <= 8 at TG
// shape, ntok <= n_ubatch).

// staging layout: [c*EXP_CACHE_STAGE_KMAX + j]; TG shapes only (k,ntok <= 8).
// Must match ggml_compute_forward_exp_cache_classify (ggml.c) and the step
// boundary readback in llama.cpp.
static constexpr int EXP_CACHE_STAGE_KMAX = 8;

static __global__ void k_exp_cache_classify(
        const int32_t * __restrict__ ids, const int64_t ids_nb1,
        const int32_t * __restrict__ remap, const int32_t * __restrict__ pending,
        int32_t * __restrict__ staging,
        const int32_t k, const int32_t ntok, const int32_t n_expert, const int32_t trash_slot,
        const int32_t component,
        void * __restrict__ dst, const int64_t dst_nb1) {
    const int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= ntok) {
        return;
    }
    const int32_t * ids_c = (const int32_t *) ((const char *) ids + (size_t) c*ids_nb1);
    if (staging && component == 2) {
        // stage the raw routed ids for the step boundary's sim/admission
        // readback (TG shapes only; the launcher passes staging == nullptr
        // otherwise). Persistent buffer — an arena tensor could be reused
        // before the boundary runs.
        for (int32_t j = 0; j < k; j++) {
            staging[c*EXP_CACHE_STAGE_KMAX + j] = ids_c[j];
        }
    }
    const uint64_t pend = (uint64_t) (uint32_t) pending[0] | ((uint64_t) (uint32_t) pending[1] << 32);

    uint64_t used = 0; // slot bitmask (H+1 <= 64 slots; pending's uint64 domain)
    for (int32_t j = 0; j < k; j++) {
        const int32_t id   = ids_c[j];
        const int32_t slot = (id >= 0 && id < n_expert) ? remap[id] : -1;
        if (slot >= 0 && slot < 64) {
            used |= (1ull << slot);
        }
    }
    for (int32_t j = 0; j < k; j++) {
        const int32_t id   = ids_c[j];
        const int32_t slot = (id >= 0 && id < n_expert) ? remap[id] : -1;
        const bool    hit  = slot >= 0;
        int32_t       h    = hit ? slot : -1;
        if (!hit) {
            int32_t free_slot = trash_slot;
            for (int32_t s = 0; s <= trash_slot && s < 64; s++) {
                if (!(used & (1ull << s)) && !(pend & (1ull << s))) { free_slot = s; break; }
            }
            if (free_slot < 64) {
                used |= (1ull << free_slot);
            }
            h = free_slot;
        }
        char * dst_e = (char *) dst + (size_t) c*dst_nb1 + (size_t) j*4;
        if (component == 1) {
            *(float *)   dst_e = hit ? 1.0f : 0.0f;
        } else if (component == 0) {
            *(int32_t *) dst_e = h;
        } else {
            *(int32_t *) dst_e = hit ? -1 : id;
        }
    }
}

void ggml_cuda_exp_cache_classify(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids     = dst->src[0];
    const ggml_tensor * remap   = dst->src[1];
    const ggml_tensor * pending = dst->src[2];
    const ggml_tensor * staging = dst->src[3];

    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == (int64_t) sizeof(int32_t));
    GGML_ASSERT(remap->type == GGML_TYPE_I32 && ggml_is_contiguous(remap));
    GGML_ASSERT(pending->type == GGML_TYPE_I32 && pending->ne[0] == 2 && ggml_is_contiguous(pending));
    GGML_ASSERT(!staging || (staging->type == GGML_TYPE_I32 && staging->ne[0] >= 64 && ggml_is_contiguous(staging)));
    GGML_ASSERT(dst->nb[0] == (int64_t) sizeof(int32_t)); // I32 and F32 outputs alike

    const int32_t component  = dst->op_params[0];
    const int32_t trash_slot = dst->op_params[1];

    const int32_t k        = (int32_t) ids->ne[0];
    const int32_t ntok     = (int32_t) ids->ne[1];
    const int32_t n_expert = (int32_t) remap->ne[0];

    const int32_t nthreads = std::min(ntok, 256);
    const int32_t nblocks  = (ntok + nthreads - 1)/nthreads;

    k_exp_cache_classify<<<nblocks, nthreads, 0, ctx.stream()>>>(
            (const int32_t *) ids->data, ids->nb[1],
            (const int32_t *) remap->data, (const int32_t *) pending->data,
            component == 2 && staging && k <= EXP_CACHE_STAGE_KMAX && ntok <= EXP_CACHE_STAGE_KMAX ? (int32_t *) staging->data : nullptr,
            k, ntok, n_expert, trash_slot, component,
            dst->data, dst->nb[1]);
    CUDA_CHECK(cudaGetLastError());
}
