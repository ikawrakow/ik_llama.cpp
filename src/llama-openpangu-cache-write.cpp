#include "llama-openpangu-cache-write.h"

#include "ggml-pack-cache-rows.h"
#include "graphs/openpangu-op-policy.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>

bool llama_openpangu_pack_cache_rows_enabled() {
    return !openpangu_legacy_ops_forced();
}

llama_openpangu_cache_write_kind llama_openpangu_select_cache_write(
        bool is_openpangu,
        bool pack_enabled,
        bool backend_available,
        bool backend_supports_pack) {
    if (!is_openpangu) {
        return llama_openpangu_cache_write_kind::none;
    }
    return pack_enabled && backend_available && backend_supports_pack ?
        llama_openpangu_cache_write_kind::pack_cache_rows :
        llama_openpangu_cache_write_kind::legacy_cpy;
}

bool llama_openpangu_pack_candidate_matches(
        const ggml_tensor * candidate,
        const ggml_tensor * a,
        const ggml_tensor * b,
        const ggml_tensor * root,
        size_t              dst_offset) {
    if (candidate == nullptr || a == nullptr || b == nullptr || root == nullptr ||
        candidate->op != GGML_OP_PACK_CACHE_ROWS ||
        candidate->src[0] != a || candidate->src[1] != b || candidate->src[2] != root ||
        candidate->view_src != root || candidate->view_offs != dst_offset ||
        candidate->type != root->type || candidate->ne[0] != root->ne[0] ||
        candidate->ne[1] != a->ne[1] || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
        candidate->nb[0] != root->nb[0] || candidate->nb[1] != root->nb[1] ||
        !ggml_pack_cache_rows_layout_is_valid(a, b, root, dst_offset)) {
        return false;
    }

    uint64_t encoded_offset = 0;
    std::memcpy(&encoded_offset, candidate->op_params, sizeof(encoded_offset));
    const void * expected_data = root->data == nullptr ? nullptr :
        (const char *) root->data + dst_offset;
    return encoded_offset == dst_offset && candidate->data == expected_data;
}

bool llama_openpangu_cache_write_plan_init(
        llama_openpangu_cache_write_plan * plan,
        llama_mtp_op_type                  op,
        uint32_t                           n_layer,
        uint32_t                           n_next,
        int32_t                            mtp_n_heads,
        int32_t                            mtp_step_idx) {
    if (plan == nullptr || n_next > n_layer) {
        return false;
    }
    if (op != MTP_OP_NONE && op != MTP_OP_DRAFT_GEN &&
        op != MTP_OP_WARMUP && op != MTP_OP_UPDATE_ACCEPTED) {
        return false;
    }

    *plan = {};
    plan->n_layer = n_layer;
    plan->first_mtp_layer = n_layer - n_next;
    plan->op = op;
    if (op == MTP_OP_NONE) {
        return plan->first_mtp_layer > 0;
    }
    if (n_next == 0 || n_next > (uint32_t) INT32_MAX) {
        return false;
    }

    plan->n_mtp = mtp_n_heads > 0 ?
        (uint32_t) std::max(1, std::min(mtp_n_heads, (int32_t) n_next)) : n_next;
    plan->step_idx = (uint32_t) std::min<int32_t>(
            std::max<int32_t>(0, mtp_step_idx), (int32_t) plan->n_mtp - 1);
    return true;
}

bool llama_openpangu_cache_write_expected(
        const llama_openpangu_cache_write_plan & plan,
        uint32_t                                  il) {
    if (il >= plan.n_layer) {
        return false;
    }
    switch (plan.op) {
        case MTP_OP_NONE:
            return il < plan.first_mtp_layer;
        case MTP_OP_DRAFT_GEN:
            return il == plan.first_mtp_layer + plan.step_idx ||
                (plan.step_idx == 1 && plan.n_mtp > 2 && il == plan.first_mtp_layer + 2);
        case MTP_OP_WARMUP:
        case MTP_OP_UPDATE_ACCEPTED:
            return il >= plan.first_mtp_layer &&
                il < plan.first_mtp_layer + plan.n_mtp;
    }
    return false;
}

static bool llama_openpangu_legacy_cpy_matches(
        const llama_openpangu_cache_write & write,
        size_t                               dst_offset) {
    const ggml_tensor * node = write.node;
    const ggml_tensor * root = write.root;
    const ggml_tensor * src = node->src[0];
    const ggml_tensor * dst = node->src[1];
    if (node->op != GGML_OP_CPY || node->view_src != root ||
        src == nullptr || dst == nullptr || dst->view_src != root ||
        node->type != root->type || dst->type != root->type ||
        node->ne[0] != root->ne[0] || node->ne[0] != src->ne[0] || node->ne[0] != dst->ne[0] ||
        node->ne[1] <= 0 || node->ne[1] != src->ne[1] || node->ne[1] != dst->ne[1] ||
        node->ne[2] != 1 || node->ne[3] != 1 || src->ne[2] != 1 || src->ne[3] != 1 ||
        dst->ne[2] != 1 || dst->ne[3] != 1 ||
        node->nb[0] != root->nb[0] || node->nb[1] != root->nb[1] ||
        dst->nb[0] != root->nb[0] || dst->nb[1] != root->nb[1] ||
        root->ne[1] <= 0 || root->nb[1] == 0 ||
        node->view_offs % root->nb[1] != 0 || dst_offset % root->nb[1] != 0) {
        return false;
    }
    const size_t current_row = node->view_offs/root->nb[1];
    const size_t dst_row = dst_offset/root->nb[1];
    const void * current_data = root->data == nullptr ? nullptr :
        (const char *) root->data + node->view_offs;
    return current_row < (size_t) root->ne[1] &&
        (size_t) node->ne[1] <= (size_t) root->ne[1] - current_row &&
        node->data == current_data && dst->data == current_data &&
        dst_row < (size_t) root->ne[1] &&
        (size_t) node->ne[1] <= (size_t) root->ne[1] - dst_row;
}

bool llama_openpangu_cache_writes_validate_and_retarget(
        const std::vector<llama_openpangu_cache_write> & writes,
        const std::vector<ggml_tensor *> &               roots,
        llama_mtp_op_type                                op,
        uint32_t                                         n_layer,
        uint32_t                                         n_next,
        int32_t                                          mtp_n_heads,
        int32_t                                          mtp_step_idx,
        uint32_t                                         cache_head,
        bool                                             retarget) {
    llama_openpangu_cache_write_plan plan;
    if (!llama_openpangu_cache_write_plan_init(
            &plan, op, n_layer, n_next, mtp_n_heads, mtp_step_idx) ||
        writes.size() != n_layer || roots.size() != n_layer) {
        return false;
    }

    size_t expected_count = 0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        const auto & write = writes[il];
        const bool expected = llama_openpangu_cache_write_expected(plan, il);
        const bool populated = write.node != nullptr || write.root != nullptr || write.step != 0 ||
            write.kind != llama_openpangu_cache_write_kind::none;
        if (expected != populated) {
            return false;
        }
        if (!expected) {
            continue;
        }
        ++expected_count;

        ggml_tensor * root = roots[il];
        if (write.node == nullptr || root == nullptr || write.root != root ||
            write.step == 0 || write.step != root->nb[1] ||
            cache_head > SIZE_MAX/write.step) {
            return false;
        }
        const size_t dst_offset = (size_t) cache_head*write.step;
        switch (write.kind) {
            case llama_openpangu_cache_write_kind::pack_cache_rows:
                if (!llama_openpangu_pack_candidate_matches(
                        write.node, write.node->src[0], write.node->src[1], root,
                        write.node->view_offs) ||
                    !ggml_pack_cache_rows_layout_is_valid(
                        write.node->src[0], write.node->src[1], root, dst_offset)) {
                    return false;
                }
                break;
            case llama_openpangu_cache_write_kind::legacy_cpy:
                if (!llama_openpangu_legacy_cpy_matches(write, dst_offset)) {
                    return false;
                }
                break;
            case llama_openpangu_cache_write_kind::none:
                return false;
        }
    }
    if (expected_count == 0) {
        return false;
    }

    if (retarget) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (!llama_openpangu_cache_write_expected(plan, il)) {
                continue;
            }
            const auto & write = writes[il];
            const size_t dst_offset = (size_t) cache_head*write.step;
            if (write.kind == llama_openpangu_cache_write_kind::pack_cache_rows) {
                ggml_pack_cache_rows_set_dst_offset(write.node, dst_offset);
            } else {
                write.node->view_offs = dst_offset;
                write.node->data = write.root->data == nullptr ? nullptr :
                    (char *) write.root->data + dst_offset;
                write.node->src[1]->data = write.node->data;
            }
        }
    }
    return true;
}
