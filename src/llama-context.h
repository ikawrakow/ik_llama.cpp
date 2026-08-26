#pragma once

#include "llama-impl.h"
#include "llama-cparams.h"
#include "llama-sampling.h"

#include "llama-spec-features.h"

struct llama_model;

#include <vector>
#include <map>
#include <set>
#include <memory>

struct llama_swa_window_view {
    int64_t w_view  = 0;
    int64_t win_off = 0;
    bool engaged    = false;
};

static inline llama_swa_window_view llama_swa_calc_window_view(
        int64_t n_kv, int64_t n_tokens, int64_t window, int64_t pad) {
    llama_swa_window_view result;
    if (window <= 0 || n_kv <= 0) {
        result.w_view = n_kv;
        return result;
    }

    const int64_t unpadded = window + pad + n_tokens;
    const int64_t overcovered = pad > 1 ? ((unpadded + pad - 1) / pad) * pad : unpadded;
    result.w_view  = overcovered < n_kv ? overcovered : n_kv;
    result.win_off = n_kv - result.w_view;
    result.engaged = result.w_view < n_kv;
    return result;
}

static inline llama_swa_window_view llama_swa_calc_window_view_compact(
        int64_t live, int64_t sink_rows, int64_t n_tokens, int64_t window, int64_t pad) {
    llama_swa_window_view result;
    const int64_t live_padded = pad > 1 ? ((live + pad - 1) / pad) * pad : live;

    const int64_t unpadded = window + pad + n_tokens;
    const int64_t overcovered = pad > 1 ? ((unpadded + pad - 1) / pad) * pad : unpadded;
    result.w_view  = overcovered < live_padded ? overcovered : live_padded;
    result.win_off = sink_rows + live_padded - result.w_view;
    result.engaged = true;
    return result;
}

struct llama_kv_cell {
    llama_pos pos   = -1;
    llama_pos delta = 0;
    int32_t   src   = 0; // used by recurrent state models to copy states

    std::set<llama_seq_id> seq_id;

    bool has_seq_id(const llama_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }

    bool is_empty() const {
        return seq_id.empty();
    }

    bool is_same_seq(const llama_kv_cell & other) const {
        return seq_id == other.seq_id;
    }
};

// ring-buffer of cached KV data
struct llama_kv_cache {
    // the FA kernels require padding to avoid extra runtime boundary checks
    static uint32_t get_padding(bool flash_attn) { return flash_attn ? 256u : 32u; }

    bool has_shift = false;
    bool do_defrag = false;
    bool do_copy   = false;
    bool recurrent = false; // with recurrent state models, a cell can hold the state for more than one past token
    bool hybrid    = false;
    bool v_trans   = true;  // the value tensor is transposed

    // openPangu s_l holds position-strict MoME conv state, not per-sequence recurrent slots: qnext
    // seq ops and generic serialization skip it, the openPangu state layouts carry it instead.
    bool s_l_position_strict = false;

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_internal also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    std::vector<uint32_t> row_count;

    bool any_compacted() const { return !row_count.empty(); }

    uint32_t rows(int il) const {
        return row_count.empty() ? size : row_count[il];
    }

    bool is_compacted(int il) const {
        return !row_count.empty() && row_count[il] < size;
    }

    // rows [sinks|window]; row(pos) = sink_rows + pos - pos_base_swa
    // sink_rows == hparams.param_sink_number, which only the openPangu loader sets (0 elsewhere)
    uint32_t  size_swa     = 0;
    uint32_t  sink_rows    = 0;
    uint32_t  window_swa   = 0;
    uint32_t  head_swa     = 0;
    llama_pos pos_base_swa = 0;

    uint32_t live_swa() const { return head_swa - sink_rows; }

    // computed before each graph build
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<llama_kv_cell> cells;

    std::vector<struct ggml_tensor *> k_l; // per layer
    std::vector<struct ggml_tensor *> v_l;
    std::vector<struct ggml_tensor *> s_l; // per layer recurrent state storage (Qwen3Next)

    // Persistent DSA indexer-key cache. One per indexer layer, MQA single head:
    // [indexer_head_size, kv_size]. Stores architecture-specific indexer keys in their
    // scoring representation so a decoded token scores against all past indexer keys.
    // Empty unless the model has the DSA indexer.
    std::vector<struct ggml_tensor *> kr_l;

    // When true, the delta_net graph builder will enable per-step SSM state saves
    bool save_per_step_ssm = false;

    std::vector<llama_split_tensor> split_k_l;
    std::vector<llama_split_tensor> split_v_l;
    std::vector<llama_split_tensor> split_s_l;

    // Per-device replicas of the MLA compressed-latent KV cache (-sm graph for DEEPSEEK2/GLM_DSA/MISTRAL4).
    std::vector<llama_split_tensor> replicated_k_l;

    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    size_t total_size() const {
        size_t size = 0;
        for (ggml_backend_buffer_t buf : bufs) {
            size += ggml_backend_buffer_get_size(buf);
        }
        return size;
    }

    // GPU-resident checkpoint for recurrent/hybrid speculative decoding
    struct gpu_checkpoint {
        std::vector<llama_kv_cell> cells_snapshot;
        uint32_t head_snapshot = 0;
        uint32_t used_snapshot = 0;

        std::vector<ggml_tensor *> s_l_shadow;

        std::vector<std::vector<ggml_tensor *>> split_s_l_shadow;

        // Per-step SSM state checkpoints for speculative decoding.
        std::vector<std::vector<ggml_tensor *>> per_step_ssm;

        // Per-step conv feature buffer: stores qkv_mixed features from the
        // verification forward pass so conv state can be reconstructed at any step.
        // One tensor per recurrent layer, each sized [conv_dim * max_tokens].
        //std::vector<std::vector<ggml_tensor *>> per_step_qkv;
        std::vector<std::vector<ggml_tensor *>> per_step_conv;

        int32_t per_step_n_tokens = 0;
        int32_t per_step_max_allocated = 0;
        int64_t per_step_ssm_state_size = 0;
        int64_t per_step_conv_state_dim = 0;
        int64_t per_step_conv_dim = 0;
        int32_t per_step_d_conv = 0;

        // DSV4 per-step compressor-state base and per-row deltas.
        std::vector<ggml_tensor *> dsv4_per_step_state;
        std::vector<ggml_tensor *> dsv4_per_step_state_shadow;
        std::vector<ggml_tensor *> dsv4_per_step_delta;
        std::vector<ggml_context *> dsv4_per_step_shadow_ctxs;
        std::vector<ggml_backend_buffer_t> dsv4_per_step_shadow_bufs;
        std::vector<int32_t> dsv4_per_step_csa_dst;
        std::vector<int32_t> dsv4_per_step_hca_dst;
        std::vector<int32_t> dsv4_per_step_lid_dst;
        std::vector<int32_t> dsv4_per_step_csa_src;
        std::vector<int32_t> dsv4_per_step_hca_src;
        std::vector<int32_t> dsv4_per_step_lid_src;
        bool dsv4_per_step_allocated = false;
        bool dsv4_per_step_saved = false;
        int32_t dsv4_per_step_max_tokens = 0;
        size_t dsv4_per_step_base_bytes = 0;
        size_t dsv4_per_step_delta_bytes = 0;

        int selected_spec_mode = -1;
        int fixed_spec_mode = LLAMA_SPEC_CKPT_NONE;
        int32_t fixed_max_tokens = 0;

        // Serialised sequence state for CPU mode
        std::vector<uint8_t> cpu_state_data;

        // Private DSV4 state snapshot used by GPU/CPU fallback modes.
        std::vector<std::vector<uint8_t>> dsv4_state_data;
        std::vector<ggml_tensor *> dsv4_state_shadow;
        std::vector<struct ggml_context *> dsv4_shadow_ctxs;
        std::vector<ggml_backend_buffer_t> dsv4_shadow_bufs;
        bool dsv4_shadow_allocated = false;
        bool dsv4_shadow_saved = false;

        // Separate storage for per-step allocations
        std::vector<struct ggml_context *>   per_step_ctxs;
        std::vector<ggml_backend_buffer_t>   per_step_bufs;

        std::vector<struct ggml_context *>   shadow_ctxs;
        std::vector<ggml_backend_buffer_t>   shadow_bufs;

        bool allocated = false;
        bool shadow_conv_only = false;
        bool saved     = false;

        void release_dsv4_per_step();
        void release_dsv4_snapshot();

        void release() {
            release_dsv4_per_step();
            release_dsv4_snapshot();

            for (struct ggml_context * ctx : shadow_ctxs) {
                ggml_free(ctx);
            }
            shadow_ctxs.clear();
            for (ggml_backend_buffer_t buf : shadow_bufs) {
                ggml_backend_buffer_free(buf);
            }
            shadow_bufs.clear();
            s_l_shadow.clear();
            split_s_l_shadow.clear();
            allocated = false;
            saved = false;

            for (struct ggml_context * ctx : per_step_ctxs) {
                ggml_free(ctx);
            }
            per_step_ctxs.clear();
            for (ggml_backend_buffer_t buf : per_step_bufs) {
                ggml_backend_buffer_free(buf);
            }
            per_step_bufs.clear();
            per_step_ssm.clear();
            per_step_conv.clear();
            per_step_max_allocated = 0;
        }

        ~gpu_checkpoint() {
            release();
        }
    };

    gpu_checkpoint ckpt;

    bool checkpoint_alloc_shadows(bool conv_only_shadow = false);
    bool checkpoint_supported() const;
    bool checkpoint_save(ggml_backend_sched_t sched);
    bool checkpoint_restore(ggml_backend_sched_t sched);
    void checkpoint_delete();

    // Per-step checkpoint: allocate, restore step k's full state (SSM + conv) to cache
    bool per_step_alloc(const llama_model & model, int max_tokens);
    bool per_step_restore(const llama_model & model, ggml_backend_sched_t sched, int step);

    ~llama_kv_cache() {
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
    }
};

struct llama_control_vector {
    std::vector<struct ggml_tensor *> tensors; // per layer
    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    struct ggml_tensor * tensor_for(int il) const {
        if (il < 0 || il < layer_start || il > layer_end || (size_t) il >= tensors.size()) {
            return nullptr;
        }
        return tensors[il];
    }

    struct ggml_tensor * apply_to(struct ggml_context * ctx, struct ggml_tensor * cur, int  il) const {
        ggml_tensor * layer_dir = tensor_for(il);
        if (layer_dir != nullptr) {
            cur = ggml_add(ctx, cur, layer_dir);
        }
        return cur;
    }

    ~llama_control_vector() {
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
    }
};

struct llama_context {

    llama_context(const llama_model & model);

    ~llama_context();

    const struct llama_model & model;

    struct llama_cparams        cparams;
    struct llama_sampling       sampling;
    struct llama_kv_cache       kv_self;
    struct llama_context      * mtp_target_ctx   = nullptr;
    struct llama_control_vector cvec;

    std::vector<float> scale_data;

    std::unordered_map<struct llama_lora_adapter *, float> lora_adapters;

    std::vector<ggml_backend_t> backends;
#ifdef GGML_USE_METAL
    ggml_backend_t backend_metal = nullptr;
#endif
    ggml_backend_t backend_cpu = nullptr;

    bool has_evaluated_once = false;

    int64_t t_start_us;
    int64_t t_load_us;
    int64_t t_p_eval_us = 0;
    int64_t t_eval_us   = 0;

    int64_t t_compute_start_us = 0;
    int64_t n_queued_tokens = 0;

    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    int32_t n_eval   = 0; // number of eval calls

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_t buf_output = nullptr;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers
    size_t  output_size = 0; // capacity (of tokens positions) for the output buffers
    int32_t n_outputs   = 0; // number of actually-used outputs in the current ubatch or last logical batch
    int32_t n_outputs_embd = 0; // number of embedding rows produced for the current logical batch

    bool logits_all = false;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // whether we are computing encoder output or decoder output
    bool is_encoding = false;

    // output of the encoder part of the encoder-decoder models
    std::vector<float> embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_t sched = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    const float * draft_input_hidden_state = nullptr;
    size_t draft_input_hidden_state_n_floats = 0;
    std::vector<float> draft_input_hidden_state_owned;

    struct dflash_runtime {
        struct target_window_state {
            const float * features = nullptr;
            size_t features_n_floats = 0;
            int32_t features_n_rows = 0;
            const float * append_features = nullptr;
            size_t append_features_n_floats = 0;
            int32_t append_features_n_rows = 0;
            const llama_pos * positions = nullptr;
            size_t positions_n = 0;
            uint64_t version = 0;
            int32_t keep_rows = 0;
            int32_t append_rows = 0;
            bool replace = false;
            std::vector<float> features_owned;
            std::vector<float> append_features_owned;
            std::vector<llama_pos> positions_owned;
            std::vector<float> features_padded;
            std::vector<llama_pos> pos_ctx_data;
            std::vector<float> kq_mask_data;
            std::vector<float> kq_mask_swa_data;
        };

        struct kv_runtime_state {
            std::vector<struct ggml_tensor *> k_ctx_cache;
            std::vector<struct ggml_tensor *> v_ctx_cache;
            struct ggml_context * cache_ctx = nullptr;
            std::vector<ggml_backend_buffer_t> cache_bufs;
            std::vector<llama_pos> cache_pos;
            std::vector<uint8_t> cache_slot_valid;
            int32_t cache_write_pos = 0;
            int32_t cache_n_filled = 0;
            int32_t cache_update_rows = 0;
            int32_t cache_reserved_rows = 0;
            int32_t cache_view_write_pos = 0;
            int32_t cache_view_n_filled = 0;
            uint64_t cache_applied_window_version = 0;
            bool cache_valid = false;
            bool cache_view_valid = false;
            std::vector<uint8_t> cache_compute_meta;
            ggml_backend_sched_t cache_sched = nullptr;
            ggml_cgraph * cache_graph = nullptr;
            int32_t cache_graph_rows = 0;
            int32_t cache_graph_write_pos = 0;
            struct ggml_tensor * cache_input_target_features = nullptr;
            struct ggml_tensor * cache_input_pos_ctx = nullptr;
            struct ggml_tensor * cache_input_rows = nullptr;
            struct ggml_tensor * kq_mask_tensor = nullptr;
            struct ggml_tensor * kq_mask_swa_tensor = nullptr;
            struct ggml_tensor * draft_tail_rows_tensor = nullptr;
        };

        struct capture_state {
            std::vector<int32_t> layer_ids;
            std::vector<std::vector<float>> layer_rows;
            struct capture_chunk {
                int32_t row_offset = 0;
                int32_t row_count = 0;
                size_t byte_offset = 0;
                ggml_type type = GGML_TYPE_COUNT;
            };
            std::vector<std::vector<capture_chunk>> layer_chunks;
            std::vector<int32_t> layer_rows_written;
            int32_t row_count = 0;
            int32_t row_width = 0;
            int32_t expected_rows = 0;
            uint64_t capture_batch_id = 0;
            std::vector<uint64_t> layer_seen_batch_id;
            bool readback_pending = false;
            bool invalid = false;
            ggml_backend_sched_eval_callback prev_cb_eval = nullptr;
            void * prev_cb_eval_user_data = nullptr;
        };

        struct input_state {
            struct ggml_tensor * target_features = nullptr; // F32 [n_target_features, cross_ctx]
            struct ggml_tensor * pos_ctx = nullptr;         // I32 [cross_ctx]
            struct ggml_tensor * kq_mask = nullptr;         // F32 [cross_ctx + n_batch, GGML_PAD(n_batch)]
            struct ggml_tensor * kq_mask_swa = nullptr;     // F32 [cross_ctx + n_batch, GGML_PAD(n_batch)]
        };

        target_window_state target;
        kv_runtime_state kv;
        std::unique_ptr<capture_state> capture;
        std::vector<float> feature_view_buffer;
        input_state inputs;
        int32_t visible_cross_ctx = 0;
        bool dspark = false;

        std::vector<llama_token> draft_tokens;
        struct ggml_tensor * draft_tokens_tensor = nullptr;
        struct ggml_tensor * draft_lattice_tensor = nullptr;
        struct ggml_tensor * draft_lattice_ids_tensor = nullptr;
        std::vector<float> draft_lattice;
        std::vector<int32_t> draft_lattice_ids;
        int32_t draft_lattice_top_k = 0;
    };
    dflash_runtime dflash;
    using dflash_capture_state = dflash_runtime::capture_state;

    struct dsv4_runtime {
        static constexpr uint32_t CSA_RATIO = 4;
        static constexpr uint32_t HCA_RATIO = 128;

        struct slot_info {
            int32_t s0 = 0;
            int32_t s1 = 0;
            std::vector<llama_seq_id> strm;
            std::vector<std::vector<uint32_t>> idxs;

            void resize(size_t n) {
                strm.resize(n);
                idxs.resize(n);
            }

            size_t size() const {
                GGML_ASSERT(strm.size() == idxs.size());
                if (idxs.empty()) {
                    return 0;
                }

                return idxs[0].size();
            }

            size_t n_stream() const {
                GGML_ASSERT(strm.size() == idxs.size());
                return strm.size();
            }

            bool empty() const {
                return idxs.empty();
            }
        };

        struct raw_context {
            std::vector<int32_t> write_src_idxs;
            std::vector<int32_t> write_dst_idxs;
            std::vector<int32_t> read_dst_idxs;
            std::vector<int32_t> write_counts;
            std::vector<int32_t> read_counts;
            slot_info sinfo_write;
            slot_info sinfo_read;
            int64_t graph_n_stream = 1;
            int64_t n_kv = 0;
        };

        struct comp_context {
            slot_info sinfo;
            int64_t graph_n_stream = 1;
            int64_t n_kv = 0;
        };

        struct comp_plan {
            std::vector<int32_t> state_pos;
            std::vector<int32_t> state_delta_src_idxs;
            std::vector<int32_t> state_delta_dst_idxs;
            std::vector<int32_t> state_persist_src_idxs;
            std::vector<int32_t> state_persist_dst_idxs;
            std::vector<int32_t> state_read_idxs;
            std::vector<int64_t> state_write_idxs;
            std::vector<int32_t> state_write_pos;
            std::vector<int32_t> n_visible;
            int64_t n_stream = 1;
            int64_t n_kv = 0;
        };

        struct comp_inputs {
            struct ggml_tensor * state_pos = nullptr;
            struct ggml_tensor * state_persist_src_idxs = nullptr;
            struct ggml_tensor * state_persist_dst_idxs = nullptr;
            struct ggml_tensor * state_read_idxs = nullptr;
            struct ggml_tensor * state_write_idxs = nullptr;
            struct ggml_tensor * state_write_pos = nullptr;
            struct ggml_tensor * kq_mask = nullptr;
        };

        struct storage {
            std::vector<struct ggml_tensor *> csa_k;
            std::vector<struct ggml_tensor *> hca_k;
            std::vector<struct ggml_tensor *> lid_k;

            std::vector<struct ggml_tensor *> csa_state_kv;
            std::vector<struct ggml_tensor *> csa_state_score;
            std::vector<struct ggml_tensor *> hca_state_kv;
            std::vector<struct ggml_tensor *> hca_state_score;
            std::vector<struct ggml_tensor *> lid_state_kv;
            std::vector<struct ggml_tensor *> lid_state_score;

            struct ggml_context * cache_ctx = nullptr;
            std::vector<ggml_backend_buffer_t> cache_bufs;
            uint32_t n_stream = 1;
        };

        struct input_state {
            struct ggml_tensor * raw_k_write_src_idxs = nullptr;
            struct ggml_tensor * raw_k_write_idxs = nullptr;
            struct ggml_tensor * raw_k_read_idxs = nullptr;
            comp_inputs csa;
            comp_inputs hca;
            comp_inputs lid;
        };

        storage cache;
        input_state inputs;
        raw_context raw;
        comp_context csa_ctx;
        comp_context hca_ctx;
        comp_context lid_ctx;
        comp_plan csa_plan;
        comp_plan hca_plan;
        comp_plan lid_plan;

        std::vector<float> csa_mask_data;
        std::vector<float> hca_mask_data;
    };
    dsv4_runtime dsv4;

    // input tensors
    struct ggml_tensor * inp_tokens;      // I32 [n_batch]
    struct ggml_tensor * inp_embd;        // F32 [n_embd, n_batch]
    struct ggml_tensor * inp_pos;         // I32 [n_batch]
    struct ggml_tensor * inp_out_ids;     // I32 [n_outputs]
    struct ggml_tensor * inp_KQ_mask;     // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_swa; // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_swa_win = nullptr; // F32 [SWA W_view, n_batch]
    struct ggml_tensor * inp_K_shift;     // I32 [kv_size]
    struct ggml_tensor * inp_mean;        // F32 [n_batch, n_batch]
    struct ggml_tensor * inp_cls;         // I32 [n_batch]
    struct ggml_tensor * inp_s_copy;      // I32 [kv_size]
    struct ggml_tensor * inp_s_mask;      // F32 [1, n_kv]
    struct ggml_tensor * inp_s_seq;       // I32 [n_kv, n_batch]
    struct ggml_tensor * inp_s_seq_qnext; // I32 [1, n_batch]
    struct ggml_tensor * inp_pos_bucket;    // I32 [n_batch|n_kv, n_batch]
    struct ggml_tensor * inp_embd_enc;      // F32 [n_embd, n_outputs_enc]
    struct ggml_tensor * inp_KQ_mask_cross; // F32 [n_outputs_enc, n_batch]
    struct ggml_tensor * inp_scale = nullptr; // F32 [n_tokens]
    struct ggml_tensor * inp_mtp_states = nullptr;
    struct ggml_tensor * inp_mtp_carry = nullptr; // F32 [n_embd, nextn-1] per-head hidden at the last committed position
    struct ggml_tensor * inp_dsa_sink = nullptr; // F32 [n_kv, n_tokens] per-sequence attention-sink boost for DSA indexer top-k

    struct swa_window_view_state {
        bool active       = false;
        bool compacted    = false;
        int32_t n_kv      = 0;
        int32_t n_tokens  = 0;
        uint32_t window   = 0;
        uint32_t pad      = 0;
        int64_t w_view    = 0;
        int64_t win_off   = 0;
    } swa_window_view;

    // multi-head MTP chaining state: head k's output row at the last committed position,
    // written back after each warmup/update decode and fed into the next MTP graph through
    // inp_mtp_carry (zeroed when a prompt warmup restarts from position 0). The readback is
    // issued async after compute; mtp_carry_pending marks a copy that must be synchronized
    // before the host buffer is read or resized.
    std::vector<float> mtp_carry;
    bool mtp_carry_pending = false;

    std::vector<uint8_t> swa_compact_buf;

    ggml_backend_t ggml_backend_by_name(const char * name);

    struct Prev;
    std::unique_ptr<Prev> prev;
    std::unique_ptr<Prev> prev_mtp;
    int32_t mtp_step_idx = 0;
    int32_t mtp_n_heads = 0;

    void reset_scheduler();
    bool can_reuse_graph(const llama_batch & u_batch, uint64_t seq_fingerprint, uint64_t model_state_hash);

    struct CacheCopy {
        ggml_tensor * cpy = nullptr;
        size_t        step = 0;
    };
    std::vector<CacheCopy> cache_copies;
    // GLM-DSA lightning indexer: the indexer-key cache (kr_l) write is a separate ggml_cpy that
    // the K/V cache_copies fixup does NOT cover. Under graph reuse (FA pads KV to 256, so n_kv
    // stays constant across consecutive decode ubatches and the graph IS reused) its view_offs
    // would stay baked at the first ubatch's kv_head, scattering this ubatch's indexer keys to a
    // stale slot. Later ubatches never populate their own recent index-key cells (those cells read
    // uninitialized -> wrong block-max-pool/top-k -> degraded/NaN sparse-FA decode). Register the
    // kr_l cpy per layer here and patch its offset in update_cache_copies(), exactly like K/V.
    std::vector<CacheCopy> dsa_cache_copies;
    std::vector<CacheCopy> openpangu_cache_copies;
    std::vector<CacheCopy> openpangu_cache_copies_mtp;

    bool update_cache_copies();

    bool ensure_dflash_kv_cache_tensors(int32_t cross_ctx);
    void free_dflash_kv_cache_tensors();
    bool ensure_dsv4_cache_tensors();
    void free_dsv4_cache_tensors();

    bool prepare_mtp_graph_inputs(
        struct llama_context & lctx);
    void set_mtp_op_type(llama_mtp_op_type value);
    void set_mtp_step_idx(int32_t value);
    void set_mtp_n_heads(int32_t value);

    int max_nodes(int n_tokens, int n_kv) const;
};
