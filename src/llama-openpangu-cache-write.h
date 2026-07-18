#pragma once

#include "ggml.h"
#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <vector>

enum class llama_openpangu_cache_write_kind {
    none,
    legacy_cpy,
    pack_cache_rows,
};

struct llama_openpangu_cache_write {
    ggml_tensor * node = nullptr;
    ggml_tensor * root = nullptr;
    size_t        step = 0;
    llama_openpangu_cache_write_kind kind = llama_openpangu_cache_write_kind::none;
};

struct llama_openpangu_cache_write_plan {
    uint32_t n_layer = 0;
    uint32_t first_mtp_layer = 0;
    uint32_t n_mtp = 0;
    uint32_t step_idx = 0;
    llama_mtp_op_type op = MTP_OP_NONE;
};

bool llama_openpangu_pack_cache_rows_enabled();

llama_openpangu_cache_write_kind llama_openpangu_select_cache_write(
        bool is_openpangu,
        bool pack_enabled,
        bool backend_available,
        bool backend_supports_pack);

bool llama_openpangu_pack_candidate_matches(
        const ggml_tensor * candidate,
        const ggml_tensor * a,
        const ggml_tensor * b,
        const ggml_tensor * root,
        size_t              dst_offset);

bool llama_openpangu_cache_write_plan_init(
        llama_openpangu_cache_write_plan * plan,
        llama_mtp_op_type                  op,
        uint32_t                           n_layer,
        uint32_t                           n_next,
        int32_t                            mtp_n_heads,
        int32_t                            mtp_step_idx);

bool llama_openpangu_cache_write_expected(
        const llama_openpangu_cache_write_plan & plan,
        uint32_t                                  il);

bool llama_openpangu_cache_writes_validate_and_retarget(
        const std::vector<llama_openpangu_cache_write> & writes,
        const std::vector<ggml_tensor *> &               roots,
        llama_mtp_op_type                                op,
        uint32_t                                         n_layer,
        uint32_t                                         n_next,
        int32_t                                          mtp_n_heads,
        int32_t                                          mtp_step_idx,
        uint32_t                                         cache_head,
        bool                                             retarget);
