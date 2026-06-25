I have all the code. Let me verify `is_power_of_2` exists and grab the `ggml_set_op_params_f32`/op_params usage isn't needed further. I now have complete coverage. Assembling the final deliverable.

---

# DeepSeek-V3.2 / GLM-DSA "Lightning Indexer" Sparse Attention — Verbatim Source from `ggml-org/llama.cpp` master

All code below was fetched from `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/<path>` on 2026-06-24. Web content treated as untrusted: I extracted only code, cite the paths, and ignored any embedded directives.

## ⚠️ Key structural findings (read before porting)

1. **`deepseek32.cpp` exists** at `src/models/deepseek32.cpp` (499 lines) and contains the full DSA lightning-indexer graph. The arch enum is `LLM_ARCH_DEEPSEEK32` / arch name `"deepseek32"` — this is DeepSeek-V3.2.

2. **`glm-dsa.cpp` exists** (`src/models/glm-dsa.cpp`, arch `LLM_ARCH_GLM_DSA`, name `"glm-dsa"`) BUT **does NOT use the indexer graph**. In `src/models/models.h:1104`:
   ```cpp
   struct llama_model_glm_dsa : public llama_model_base {
       ...
       using graph = llama_model_deepseek2::graph;   // <-- plain DeepSeek-V2 MLA graph, NO indexer
   ```
   glm-dsa **loads** the indexer tensors (all marked `TENSOR_NOT_REQUIRED`) but builds the regular deepseek2 MLA graph and **never references them**. Corroborating evidence:
   - In `create_memory` (`llama-model.cpp:2026`) only `LLM_ARCH_DEEPSEEK32` constructs `llama_kv_cache_dsa`; `GLM_DSA` falls through to the standard cache.
   - The Hadamard/DSA gate in `llama-kv-cache.cpp:339` is `if (model.arch == LLM_ARCH_DEEPSEEK32 && ...)` — `GLM_DSA` is **excluded**.
   - `GLM_DSA` returns `LLAMA_ROPE_TYPE_NORM` (`llama-model.cpp:2426`), whereas `DEEPSEEK32` is in the default-fallthrough rope group.
   
   **So at master HEAD, glm-dsa is a stub: the DSA pathway is fully implemented only for deepseek32.** For your GLM-5.2 port, use the `deepseek32` graph as the reference and wire glm-dsa to it (mirroring what deepseek32 does), since the GLM-DSA scaffolding/tensor names already exist.

3. The per-arch tensor *names* are global (one `LLM_TENSOR_NAMES` map in `llama-arch.cpp:359`); arch differentiation happens entirely in each model's `load_arch_tensors` / graph constructor, not via per-arch name tables.

---

## 1. `src/models/deepseek32.cpp` — full graph constructor (the DSA lightning-indexer block)

This is the file that actually contains the indexer. Quoted in full.

`src/models/deepseek32.cpp` (lines 1–499):

```cpp
#include "models.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-dsa.h"

void llama_model_deepseek32::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,     hparams.n_ff_exp);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
    hparams.f_norm_eps = 1e-6;  // eps for layer norm
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    // MoE parameters
    ml.get_key(LLM_KV_EXPERT_COUNT,                hparams.n_expert);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT,           hparams.n_expert_used);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);

    // deepseek MLA parameters
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,      hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,     hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   hparams.n_embd_head_k_mla_impl, false);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, hparams.n_embd_head_v_mla_impl, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,        hparams.n_expert_shared);

    // DSA parameters
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    // Expert gating function
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);

    if (ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul, 0.0f)) {
        // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
        // cancel the factor from the convert script
        hparams.rope_yarn_log_mul /= 0.1f;
    }

    // NextN/MTP parameters
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer");

    switch (hparams.n_layer()) {
        case 62: type = LLM_TYPE_685B_A37B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_deepseek32::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const bool is_mla = hparams.is_mla();
    if (!is_mla) {
        throw std::runtime_error("DEEPSEEK32 architecture requires MLA");
    }

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // try to load output.weight, if not found, use token_embd (tied embeddings)
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer_all; ++i) {
        int flags = 0;
        if (i >= n_layer) {
            // skip all tensors in the NextN layers
            // TODO @ngxson : TENSOR_NOT_REQUIRED was a hack, need to remove it later
            flags |= TENSOR_SKIP | TENSOR_NOT_REQUIRED;
        }

        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);

        layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, flags);
        layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, flags);

        layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + n_embd_head_qk_rope}, flags);

        // note: only old legacy GGUF files will have the unsplit wkv_b tensor in
        layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_qk_nope, kv_lora_rank, n_head}, flags);
        layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, flags);

        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, flags);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        // DSA indexer
        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {hparams.indexer_head_size}, flags);
        layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   i), {hparams.indexer_head_size}, flags);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd, hparams.indexer_head_size}, flags);
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * hparams.indexer_head_size}, flags);
        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, flags);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);

            if (n_expert == 0) {
                throw std::runtime_error("n_expert must be > 0");
            }
            if (n_expert_used == 0) {
                throw std::runtime_error("n_expert_used must be > 0");
            }

            // MoE branch
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, flags);

            // Shared expert branch
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, flags);
        }

        // NextN/MTP tensors (preserved but unused) - conditionally load for last nextn_predict_layers
        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", i), { 2 * n_embd, n_embd }, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM, "weight", i), { n_embd }, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM, "weight", i), { n_embd }, flags);

            // Optional tensors
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS, "weight", i), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), { n_embd }, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek32::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_deepseek32::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const bool is_mla = hparams.is_mla();
    GGML_ASSERT(is_mla);

    // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
    const int64_t n_embd_head_k = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v = hparams.n_embd_head_v_mla();
    GGML_UNUSED(n_embd_head_v);

    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k - n_embd_head_qk_rope;

    const int64_t n_indexer_head = hparams.indexer_n_head;
    const int64_t n_embd_indexer_head = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const uint32_t n_indexer_top_k = hparams.indexer_top_k;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
    // See https://github.com/ggml-org/llama.cpp/discussions/7416 for detailed explanation.
    // And also: https://github.com/ggml-org/llama.cpp/pull/17945 [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]

    // first cancel the adjustment from llama_hparams::yarn_attn_factor_adjust to get the original attn_factor
    GGML_ASSERT(ext_factor >= 0.0f);
    const float attn_factor_org = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));

    // use the original attn_factor to pre-scale the kq_scale
    const float mscale   = attn_factor_org * (1.0f + 0.1f * hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_dsa * inp_attn_dsa = build_attn_inp_k_dsa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            ggml_tensor * qr = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
            cb(qr, "qr", il);

            qr = build_norm(qr, model.layers[il].attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(qr, "qr", il);

            ggml_tensor * top_k = nullptr;

            // lightning indexer
            {
                ggml_tensor * indexer_q = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_q_b, qr);
                cb(indexer_q, "indexer_q", il);

                // split into {n_embd_indexer_head_rope, n_indexer_head, n_tokens}
                ggml_tensor * indexer_q_pe =
                    ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_rope, n_indexer_head, n_tokens,
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head, 0);
                cb(indexer_q_pe, "indexer_q_pe", il);

                // and {n_embd_indexer_head_nope, n_indexer_head, n_tokens}
                ggml_tensor * indexer_q_nope =
                    ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_nope, n_indexer_head, n_tokens,
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head) * n_indexer_head,
                                 ggml_row_size(indexer_q->type, n_embd_indexer_head_nope));
                cb(indexer_q_nope, "indexer_q_nope", il);

                indexer_q_pe = ggml_rope_ext(ctx0, indexer_q_pe, inp_pos, nullptr, n_rot,
                                     LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
                cb(indexer_q_pe, "indexer_q_pe", il);

                // {n_embd_indexer_head_rope + n_embd_indexer_head_nope, n_head, n_tokens}
                indexer_q = ggml_concat(ctx0, indexer_q_pe, indexer_q_nope, 0);
                cb(indexer_q, "indexer_q", il);

                ggml_tensor * indexer_k = ggml_mul_mat(ctx0, model.layers[il].indexer_attn_k, cur);
                cb(indexer_k, "indexer_k", il);

                indexer_k = build_norm(indexer_k, model.layers[il].indexer_k_norm, model.layers[il].indexer_k_norm_b, LLM_NORM, il);
                cb(indexer_k, "indexer_k", il);

                // split into {n_embd_indexer_head_rope, 1, n_tokens}
                ggml_tensor * indexer_k_pe =
                    ggml_view_3d(ctx0, indexer_k, n_embd_indexer_head_rope, 1, n_tokens,
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head) * 1, 0);
                cb(indexer_k_pe, "indexer_k_pe", il);

                // and {n_embd_indexer_head_nope, 1, n_tokens}
                ggml_tensor * indexer_k_nope =
                    ggml_view_3d(ctx0, indexer_k, n_embd_indexer_head_nope, 1, n_tokens,
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head),
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head) * 1,
                                 ggml_row_size(indexer_k->type, n_embd_indexer_head_nope));
                cb(indexer_k_nope, "indexer_k_nope", il);

                indexer_k_pe = ggml_rope_ext(ctx0, indexer_k_pe, inp_pos, nullptr, n_rot,
                                     LLAMA_ROPE_TYPE_NEOX, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
                cb(indexer_k_pe, "indexer_k_pe", il);

                // {n_embd_indexer_head_rope + n_embd_indexer_head_nope, 1, n_tokens}
                indexer_k = ggml_concat(ctx0, indexer_k_pe, indexer_k_nope, 0);
                cb(indexer_k, "indexer_k", il);

                // perform Hadamard transform on indexer q and k
                indexer_q = ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_q);
                cb(indexer_q, "indexer_q", il);
                indexer_k = ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_k);
                cb(indexer_k, "indexer_k", il);

                // store indexer keys to KV cache
                const auto * mctx_lid = inp_attn_dsa->mctx->get_lid();
                const auto & k_idxs_lid = inp_attn_dsa->get_k_idxs_lid();
                ggml_build_forward_expand(gf, mctx_lid->cpy_k(ctx0, indexer_k, k_idxs_lid, il));

                // prepare indexer weights
                ggml_tensor * indexer_weights = ggml_mul_mat(ctx0, model.layers[il].indexer_proj, cur);
                cb(indexer_weights, "indexer_weights", il);

                // get cached indexer keys
                indexer_k = mctx_lid->get_k(ctx0, il);

                // split the batch into streams if needed
                const auto n_stream = indexer_k->ne[3];
                indexer_q = ggml_view_4d(ctx0, indexer_q, indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream, indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
                indexer_weights = ggml_view_4d(ctx0, indexer_weights, indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream, indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

                // calculate indexer kq
                indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
                cb(indexer_q, "indexer_q", il);
                indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
                cb(indexer_k, "indexer_k", il);

                ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
                cb(indexer_kq, "indexer_kq", il);

                // ReLU requires contiguous tensors
                indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
                cb(indexer_kq, "indexer_kq", il);

                // apply ReLU
                ggml_tensor * indexer_score = ggml_relu(ctx0, indexer_kq);
                cb(indexer_score, "indexer_score", il);

                // pre-scale weights to avoid scaling operations on huge indexer_score tensor
                indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f / sqrtf(float(n_embd_indexer_head * n_indexer_head)));
                cb(indexer_weights, "indexer_weights", il);

                // multiply scores by indexer weights
                indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
                cb(indexer_score, "indexer_score", il);

                // sum by q n_indexer_head dimension
                indexer_score = ggml_sum_rows(ctx0, indexer_score);
                cb(indexer_score, "indexer_score", il);

                // permute result to match KQ mask
                indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
                cb(indexer_score, "indexer_score", il);

                // mask indexer scores
                ggml_tensor * indexer_kq_mask = inp_attn_dsa->get_kq_mask_lid();
                indexer_score = ggml_add(ctx0, indexer_score, indexer_kq_mask);
                cb(indexer_score, "indexer_score", il);

                // get indices of top k indexer scores
                uint32_t n_top_k = indexer_score->ne[0] < n_indexer_top_k ? indexer_score->ne[0] : n_indexer_top_k;
                top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
                cb(top_k, "top_k", il);
            }

            ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq_b, qr);
            cb(q, "q", il);

            // split into {n_embd_head_qk_nope, n_head, n_tokens}
            ggml_tensor * q_nope =
                ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                             ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
            cb(q_nope, "q_nope", il);

            // and {n_embd_head_qk_rope, n_head, n_tokens}
            ggml_tensor * q_pe = ggml_view_3d(
                ctx0, q, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(q->type, n_embd_head_k),
                ggml_row_size(q->type, n_embd_head_k) * n_head, ggml_row_size(q->type, n_embd_head_qk_nope));
            cb(q_pe, "q_pe", il);

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
            cb(kv_cmpr_pe, "kv_cmpr_pe", il);

            // split into {kv_lora_rank, n_tokens}
            ggml_tensor * kv_cmpr =
                ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                             ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            cb(kv_cmpr, "kv_cmpr", il);

            // and {n_embd_head_qk_rope, 1, n_tokens}
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                              ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            cb(k_pe, "k_pe", il);

            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            // MLA attention
            {
                // {n_embd_head_qk_nope, n_tokens, n_head}
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                cb(q_nope, "q_nope_perm", il);

                // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, model.layers[il].wk_b, q_nope);
                cb(q_nope_absorbed, "q_nope_absorbed", il);

                // {kv_lora_rank, n_head, n_tokens}
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);
                cb(q_nope_absorbed, "q_nope_absorbed_perm", il);

                // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
                // note: rope must go first for in-place context shifting in build_rope_shift()
                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
                cb(Qcur, "Qcur", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                cb(kv_cmpr, "kv_cmpr_reshape", il);

                // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                cb(Kcur, "Kcur", il);

                // {kv_lora_rank, 1, n_tokens}
                ggml_tensor * Vcur = kv_cmpr;
                cb(Vcur, "Vcur", il);

                // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
                cur = build_attn(inp_attn_dsa,
                        model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, model.layers[il].wv_b, top_k, kq_scale, il);
            }
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // MoE branch
            ggml_tensor * moe_out = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                model.layers[il].ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                model.layers[il].ffn_gate_up_exps,
                model.layers[il].ffn_up_exps_s,
                model.layers[il].ffn_gate_exps_s,
                model.layers[il].ffn_down_exps_s);
            cb(moe_out, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor * ffn_shexp =
                    build_ffn(cur,
                        model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                        model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                        model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);

                cur = ggml_add(ctx0, moe_out, ffn_shexp);
                cb(cur, "ffn_out", il);
            }
        }
        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
```

**Note on Hadamard usage in the graph:** Inside the indexer block, the Hadamard rotation is applied directly via `ggml_mul_mat(ctx0, inp_attn_dsa->self_k_rot_lid, indexer_q/_k)` — i.e. the lid cache's `build_input_k_rot` tensor is multiplied into both indexer q and k *before* caching/scoring. This is distinct from the `build_attn(...)` path's `ggml_mul_mat_aux` rotation used for the main quantized KV cache.

---

## 2. `src/llama-kv-cache-dsa.{h,cpp}` — the dual KV-cache (kv_mla + kv_lid)

### `src/llama-kv-cache-dsa.h` (full, 138 lines)

```cpp
#pragma once

#include "llama-kv-cache.h"

#include <vector>

//
// llama_kv_cache_dsa
//

// utilizes two instances of llama_kv_cache:
// - the first instance is for caching key tensors of the model,
// - the second instance is for caching lightning indexer key tensors

class llama_kv_cache_dsa : public llama_memory_i {
public:
    llama_kv_cache_dsa(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache_dsa() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache_dsa specific API
    //

    llama_kv_cache * get_mla() const;
    llama_kv_cache * get_lid() const;

private:
    // we keep indexer KV cache hparams instance here as llama_kv_cache stores only reference to it
    llama_hparams hparams_lid;
    const uint32_t n_stream  = 1;

    std::unique_ptr<llama_kv_cache> kv_mla;
    std::unique_ptr<llama_kv_cache> kv_lid;
};

class llama_kv_cache_dsa_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    llama_kv_cache_dsa_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_dsa_context(
            llama_kv_cache_dsa * kv);

    // used to create an update context
    llama_kv_cache_dsa_context(
            llama_kv_cache_dsa * kv,
            llama_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llama_kv_cache_dsa_context(
            llama_kv_cache_dsa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_ik,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_dsa_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_dsa_context specific API
    //

    const llama_kv_cache_context * get_mla() const;
    const llama_kv_cache_context * get_lid()  const;

private:
    //llama_kv_cache_dsa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_mla;
    const llama_memory_context_ptr ctx_lid;

    const llama_memory_status status;
};
```

### `src/llama-kv-cache-dsa.cpp:llama_kv_cache_dsa::llama_kv_cache_dsa` — the two sub-caches + indexer hparam overrides

This is the critical constructor: it builds `kv_mla` from the model's real hparams, then **hand-tweaks a copied `hparams_lid`** (`n_head_kv = 1`, `n_embd_head_k_full = indexer_head_size`, `rope_type = NEOX`) and builds `kv_lid` from it.

```cpp
llama_kv_cache_dsa::llama_kv_cache_dsa(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse) :
    hparams_lid(model.hparams), n_stream(unified ? 1 : n_seq_max) {

    LLAMA_LOG_INFO("%s: creating main KV cache, size = %u cells\n", __func__, kv_size);

    kv_mla = std::make_unique<llama_kv_cache>(
            model, model.hparams, type_k, type_v,
            v_trans, offload, unified, kv_size, n_seq_max, n_pad,
            n_swa, swa_type, nullptr, filter, reuse, nullptr);

    // we use llama_kv_cache for caching indexer keys
    // by hand-tweaking some hparams we fool it to create
    // indexer key cache tensors with correct dimensions
    // https://github.com/ggml-org/llama.cpp/pull/21149#discussion_r3015940823

    // DSA lightning indexer uses MQA with single key head
    std::fill(hparams_lid.n_head_kv_arr.begin(), hparams_lid.n_head_kv_arr.end(), 1);
    hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;
    hparams_lid.rope_type          = LLAMA_ROPE_TYPE_NEOX;

    LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

    kv_lid = std::make_unique<llama_kv_cache>(
            model, hparams_lid, type_k, type_v,
            v_trans, offload, unified, kv_size, n_seq_max, n_pad,
            n_swa, swa_type, nullptr, filter, reuse, nullptr);
}
```

The rest of `llama-kv-cache-dsa.cpp` simply fans every `llama_memory_i` method out to both sub-caches and combines statuses. `init_batch` prepares both caches and asserts `sinfos_mla.size() == sinfos_lid.size()`:

```cpp
llama_memory_context_ptr llama_kv_cache_dsa::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_mla = kv_mla->prepare(ubatches);
        if (sinfos_mla.empty()) {
            break;
        }

        auto sinfos_lid = kv_lid->prepare(ubatches);
        if (sinfos_lid.empty()) {
            break;
        }

        assert(sinfos_mla.size() == sinfos_lid.size());

        return std::make_unique<llama_kv_cache_dsa_context>(
                this, std::move(sinfos_mla), std::move(sinfos_lid), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_dsa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_kv_cache * llama_kv_cache_dsa::get_mla() const { return kv_mla.get(); }
llama_kv_cache * llama_kv_cache_dsa::get_lid() const { return kv_lid.get(); }
```

And the context accessors:

```cpp
const llama_kv_cache_context * llama_kv_cache_dsa_context::get_mla() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return static_cast<const llama_kv_cache_context *>(ctx_mla.get());
}

const llama_kv_cache_context * llama_kv_cache_dsa_context::get_lid()  const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return static_cast<const llama_kv_cache_context *>(ctx_lid.get());
}
```

**`cpy_k` / `get_k` for the lid cache** are NOT special — the lid cache is a plain `llama_kv_cache`. In `deepseek32.cpp` the lid cache is driven through the standard context methods (`mctx_lid->cpy_k(...)`, `mctx_lid->get_k(...)`):

`src/llama-kv-cache.cpp:llama_kv_cache_context::get_k / cpy_k`:
```cpp
ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}
ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}
```

`src/llama-kv-cache.cpp:llama_kv_cache::get_k` (the actual view):
```cpp
ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}
```

`src/llama-kv-cache.cpp:llama_kv_cache::cpy_k`:
```cpp
ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}
```

---

## 3. `src/llama-graph.{h,cpp}` — the `top_k` build_attn overload, `build_attn_inp_k_dsa`, and `llm_graph_input_attn_k_dsa`

### `src/llama-graph.h:llm_graph_input_attn_k_dsa` (the input struct, lines 378–417)

```cpp
class llm_graph_input_attn_k_dsa : public llm_graph_input_i {
public:
    llm_graph_input_attn_k_dsa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_dsa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k_dsa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs_mla() const { return self_k_idxs_mla; }
    ggml_tensor * get_k_idxs_lid() const { return self_k_idxs_lid; }

    ggml_tensor * get_kq_mask_mla() const { return self_kq_mask_mla_cnv; }
    ggml_tensor * get_kq_mask_lid() const { return self_kq_mask_lid; }

    ggml_tensor * self_k_idxs_mla = nullptr; // I64 [n_batch]
    ggml_tensor * self_k_idxs_lid = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask_mla     = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_mla_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_lid     = nullptr; // F32     [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_lid_cnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

    ggml_tensor * self_k_rot_lid = nullptr;

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_dsa_context * mctx;
};
```

### `src/llama-graph.h` — the two `build_attn` decls (k_dsa input + the `top_k` overload, lines 1029–1044)

```cpp
    llm_graph_input_attn_k_dsa * build_attn_inp_k_dsa() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_k_dsa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
            ggml_tensor * top_k, // [n_indexer_top_k, n_tokens]
                  float   kq_scale,
                    int   il) const;
```

### `src/llama-graph.cpp:llm_graph_context::build_attn` (the `top_k` sparse-mask overload)

This is the function that builds the sparse mask via `ggml_fill(-INFINITY)` + `ggml_set_rows`.

```cpp
ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask_mla();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}
```

### `src/llama-graph.cpp:llm_graph_context::build_attn_inp_k_dsa`

Note: the lid mask is forced F32 by setting a copied `cparams.flash_attn = false`.

```cpp
llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_mla = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
        inp->self_kq_mask_mla_cnv = inp->self_kq_mask_mla;
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure F32 mask
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = false;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
    }

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}
```

### `src/llama-graph.cpp:llm_graph_input_attn_k_dsa::set_input` / `can_reuse`

```cpp
void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}
```

---

## 4. `src/llama-kv-cache.{h,cpp}` — Walsh-Hadamard generation + k_rot gating

### `src/llama-kv-cache.cpp:ggml_gen_hadamard` (the orthonormal rotation matrix generator)

```cpp
// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}
```

### `src/llama-kv-cache.cpp:ggml_mul_mat_aux` (the helper that applies the rotation with the Hadamard hint)

```cpp
static ggml_tensor * ggml_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;

    res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat   (ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}
```

### `src/llama-kv-cache.cpp` — where `attn_rot_k` is gated by arch + Hadamard precompute (constructor body)

This is the **only arch gate** and it is `LLM_ARCH_DEEPSEEK32`-specific (glm-dsa NOT included):

```cpp
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek V3.2 DSA lightning indexer
        if (model.arch == LLM_ARCH_DEEPSEEK32 && hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }
```

> ⚠️ For the lid (indexer) cache: the override `hparams_lid.n_embd_head_k_full = indexer_head_size` (from §2) is what makes `hparams.n_embd_head_k_full == hparams.indexer_head_size` true, so `attn_rot_k` is forced on for the lid cache. The condition uses `model.arch` (DEEPSEEK32) which is shared by both sub-caches since both are built from the same `model`.

### `src/llama-kv-cache.cpp:llama_kv_cache::build_input_k_rot` / `build_input_v_rot`

```cpp
ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}
```

### `src/llama-kv-cache.cpp:llama_kv_cache::set_input_k_rot` / `set_input_v_rot`

```cpp
void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}
```

### `src/llama-kv-cache.h` — member + accessor declarations

```cpp
    // (member section)
    bool attn_rot_k = false;
    bool attn_rot_v = false;
    ...
    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;
```
```cpp
    // llama_kv_cache method decls
    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;   // line 203
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;   // line 204
    void set_input_k_rot(ggml_tensor * dst) const;               // line 214
    void set_input_v_rot(ggml_tensor * dst) const;               // line 215
```
```cpp
    // llama_kv_cache_context method decls (single-arg, line 386/387/396/397)
    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;
    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;
```

The context wrappers (`src/llama-kv-cache.cpp:2598-2631`) just forward to `kv->...`:
```cpp
ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const { return kv->build_input_k_rot(ctx); }
ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const { return kv->build_input_v_rot(ctx); }
void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const { kv->set_input_k_rot(dst); }
void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const { kv->set_input_v_rot(dst); }
```

---

## 5. `ggml_fill` F16 support (PR #23346)

### `ggml/include/ggml.h:ggml_fill` declaration (lines 2349–2357)

```cpp
    // Fill tensor a with constant c
    GGML_API struct ggml_tensor * ggml_fill(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 c);

    GGML_API struct ggml_tensor * ggml_fill_inplace(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            float                 c);
```
Op enum: `GGML_OP_FILL` (`ggml/include/ggml.h:556`). Related: `ggml_top_k` (line 2387), `ggml_argsort_top_k` (line 2380), `ggml_set_rows` (line 1683), and the hint enum `GGML_HINT_SRC0_IS_HADAMARD = 1` (line 444) used by `ggml_mul_mat_set_hint` (line 1430).

### `ggml/src/ggml-cuda/fill.cu` (full — the F16 handling PR #23346 added)

```cpp
#include "fill.cuh"
#include "convert.cuh"

#define CUDA_FILL_BLOCK_SIZE 256

template <typename T>
static __global__ void fill_kernel(T * dst, const int64_t k, const T value) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    dst[i] = value;
}

void ggml_cuda_op_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(dst));

    float value;
    memcpy(&value, dst->op_params, sizeof(float));

    const int64_t k = ggml_nelements(dst);
    const int64_t num_blocks = (k + CUDA_FILL_BLOCK_SIZE - 1) / CUDA_FILL_BLOCK_SIZE;

    switch (dst->type) {
        case GGML_TYPE_F32:
            fill_kernel<<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream>>>((float *)dst_d, k, value);
            break;
        case GGML_TYPE_F16:
            fill_kernel<<<num_blocks, CUDA_FILL_BLOCK_SIZE, 0, stream>>>((half *)dst_d, k, ggml_cuda_cast<half>(value));
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
```
`ggml/src/ggml-cuda/fill.cuh`:
```cpp
#include "common.cuh"

void ggml_cuda_op_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
```

### `ggml/src/ggml-cpu/ops.cpp` — CPU F32 + F16 fill (lines 2217–2275)

```cpp
// ggml_compute_fill

static void ggml_compute_forward_fill_f32(const ggml_compute_params * params, ggml_tensor * dst) {
    const float c = ggml_get_op_params_f32(dst, 0);

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    const auto [ir0, ir1] = get_thread_range(params, dst);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne2*ne1);
        const int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        const int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        float * dst_ptr  = (float *) ((char *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f32(ne0, dst_ptr, c);
    }
}

static void ggml_compute_forward_fill_f16(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_fp16_t c = GGML_CPU_FP32_TO_FP16(ggml_get_op_params_f32(dst, 0));

    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne);
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb);

    const auto [ir0, ir1] = get_thread_range(params, dst);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne2*ne1);
        const int64_t i02 = (ir - i03*ne2*ne1)/ne1;
        const int64_t i01 = (ir - i03*ne2*ne1 - i02*ne1);

        ggml_fp16_t * dst_ptr  = (ggml_fp16_t *) ((char *) dst->data + i03*nb3 + i02*nb2 + i01*nb1);

        ggml_vec_set_f16(ne0, dst_ptr, c);
    }
}

void ggml_compute_forward_fill(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            {
                ggml_compute_forward_fill_f32(params, dst);
            } break;
        case GGML_TYPE_F16:
            {
                ggml_compute_forward_fill_f16(params, dst);
            } break;
        default:
            {
                GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(src0->type));
            }
    }
}
```

---

## 6. Arch / hparams wiring for `glm-dsa` vs `deepseek32`

### `src/llama-arch.h` — enum entries

```cpp
    LLM_ARCH_DEEPSEEK32,            // line 84
    ...
    LLM_ARCH_GLM_DSA,              // line 88
    ...
    // KV keys (lines 254-256)
    LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,
    LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
    LLM_KV_ATTENTION_INDEXER_TOP_K,
    ...
    // tensor enum (lines 564-567)
    LLM_TENSOR_INDEXER_K_NORM,
    LLM_TENSOR_INDEXER_PROJ,
    LLM_TENSOR_INDEXER_ATTN_K,
    LLM_TENSOR_INDEXER_ATTN_Q_B,
```

### `src/llama-arch.cpp` — names, KV-key strings, tensor names, tensor-info ops

```cpp
// LLM_ARCH_NAMES (lines 79, 83)
    { LLM_ARCH_DEEPSEEK32,       "deepseek32"       },
    { LLM_ARCH_GLM_DSA,          "glm-dsa"          },

// LLM_KV_NAMES (lines 249-251) — GGUF KV keys (printf'd with arch name)
    { LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,           "%s.attention.indexer.head_count"           },
    { LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,           "%s.attention.indexer.key_length"           },
    { LLM_KV_ATTENTION_INDEXER_TOP_K,                "%s.attention.indexer.top_k"                },

// LLM_TENSOR_NAMES (lines 564-567) — GGUF tensor name templates
    { LLM_TENSOR_INDEXER_K_NORM,                         "blk.%d.indexer.k_norm" },
    { LLM_TENSOR_INDEXER_PROJ,                           "blk.%d.indexer.proj" },
    { LLM_TENSOR_INDEXER_ATTN_K,                         "blk.%d.indexer.attn_k" },
    { LLM_TENSOR_INDEXER_ATTN_Q_B,                       "blk.%d.indexer.attn_q_b" },

// LLM_TENSOR_INFOS (lines 777-780) — op type per tensor (used by quant/offload logic)
    {LLM_TENSOR_INDEXER_K_NORM,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_INDEXER_PROJ,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_INDEXER_ATTN_K,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_INDEXER_ATTN_Q_B,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
```
Note: `indexer.k_norm` carries a `"bias"` variant too (loaded via `tn(LLM_TENSOR_INDEXER_K_NORM, "bias", i)`), reusing the same `LLM_TENSOR_INDEXER_K_NORM` name with the `bias` suffix.

Both archs also appear together in `llm_arch_supports_sm_tensor` (lines 934-935):
```cpp
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_GLM_DSA:
```

### `src/llama-hparams.h` — the indexer members (lines 224-227) + key length accessors

```cpp
    // DSA (deepseek sparse attention)
    uint32_t indexer_n_head    = 0;
    uint32_t indexer_head_size = 0;
    uint32_t indexer_top_k     = 0;
```
Plus the MLA infrastructure the indexer relies on:
```cpp
    uint32_t n_embd_head_k_full;        // line 62 (overridden to indexer_head_size for the lid cache)
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;   // line 82 (set to 1 for lid cache)
    uint32_t n_embd_head_k_mla_impl = 0;  // line 72
    uint32_t n_embd_head_v_mla_impl = 0;  // line 73
    uint32_t n_lora_q  = 0;               // line 86
    uint32_t n_lora_kv = 0;               // line 87
    enum llama_rope_type rope_type = LLAMA_ROPE_TYPE_NONE;  // line 252 (set to NEOX for lid cache)
    uint32_t n_embd_head_k_mla() const;   // line 348
    uint32_t n_embd_head_v_mla() const;   // line 349
    bool     is_mla()           const;    // line 346
```

### `src/llama-hparams.cpp` — `is_mla`, MLA head accessors, swa/full key length

```cpp
bool llama_hparams::is_mla() const {
    assert((n_embd_head_k_mla_impl == 0 && n_embd_head_v_mla_impl == 0) ||
           (n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0));

    return n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0;
}
// (line 252) n_embd_head_k() uses n_embd_head_k_full:
//     return is_swa(il) ? n_embd_head_k_swa : n_embd_head_k_full;
// n_embd_head_k_mla()  -> is_mla() ? n_embd_head_k_mla_impl : n_embd_head_k();
// n_embd_head_v_mla()  -> is_mla() ? n_embd_head_v_mla_impl : n_embd_head_v();
```

### `src/llama-model.cpp` — instantiation, LLM_TYPE, memory, rope (the divergence points)

```cpp
// create_model dispatch (lines 182-185)
        case LLM_ARCH_DEEPSEEK32:
            return new llama_model_deepseek32(params);
        case LLM_ARCH_GLM_DSA:
            return new llama_model_glm_dsa(params);

// LLM_TYPE names (lines 805-806)
        case LLM_TYPE_685B_A37B:     return "685B.A37B";   // deepseek32
        case LLM_TYPE_744B_A40B:     return "744B.A40B";   // glm-dsa

// create_memory (lines 2026-2042): ONLY deepseek32 builds the dual cache
        case LLM_ARCH_DEEPSEEK32:
            {
                res = new llama_kv_cache_dsa(
                        *this,
                        params.type_k,
                        params.type_v,
                        !cparams.flash_attn,
                        cparams.offload_kqv,
                        cparams.kv_unified,
                        cparams.n_ctx_seq,
                        cparams.n_seq_max,
                        1,
                        hparams.n_swa,
                        hparams.swa_type,
                        nullptr,
                        nullptr);
            } break;
        // GLM_DSA is NOT here -> falls through to the standard kv_cache default branch

// llama_model_rope_type (lines 2408 + 2426)
        case LLM_ARCH_DEEPSEEK32:   // ... falls into LLAMA_ROPE_TYPE_NORM group with the deepseek family
        ...
        case LLM_ARCH_GLM_DSA:
            return LLAMA_ROPE_TYPE_NORM;
```

### `src/models/models.h` — model struct declarations (the graph divergence)

```cpp
struct llama_model_deepseek32 : public llama_model_base {            // line 1075
    llama_model_deepseek32(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {           // <-- OWN DSA indexer graph
        graph(const llama_model & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_glm_dsa : public llama_model_base {               // line 1099
    llama_model_glm_dsa(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    using graph = llama_model_deepseek2::graph;          // <-- ALIAS to plain DeepSeek-V2 MLA graph (NO indexer)

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
```

---

## Divergence summary: `deepseek32` vs `glm-dsa`

| Aspect | `deepseek32` (DSA active) | `glm-dsa` (stub at HEAD) |
|---|---|---|
| Graph | Own `graph` ctor with full lightning-indexer block | `using graph = llama_model_deepseek2::graph` (plain MLA, no indexer) |
| KV cache | `llama_kv_cache_dsa` (kv_mla + kv_lid) via `create_memory` | Standard `llama_kv_cache` (default fallthrough) |
| Indexer tensors | Loaded **required** (`flags`) | Loaded `TENSOR_NOT_REQUIRED` but **unused** |
| Hadamard gate | `model.arch == LLM_ARCH_DEEPSEEK32` forces `attn_rot_k = true` | Not gated → no Hadamard rotation |
| rope_type | deepseek family default group | `LLAMA_ROPE_TYPE_NORM` |
| LLM_TYPE | `LLM_TYPE_685B_A37B` (62 layers) | `LLM_TYPE_744B_A40B` (79 layers) |
| Expert gating | `EXPERT_GATING_FUNC` required | defaults to SIGMOID if absent (GLM-4.5 style) |

For your GLM-5.2 / DeepSeek-V3.2 DSA port into ik (`build_deepseek2.cpp` in `src/graphs/`), **`deepseek32` is the single source of truth.** To get DSA on GLM, you would replicate what deepseek32 does (own graph + `llama_kv_cache_dsa` + arch gate including GLM_DSA), since mainline's glm-dsa does not yet exercise the indexer.

---

## Flags / things to note
- **Nothing failed to fetch.** Every requested file/function was located and quoted verbatim from master.
- The path you guessed (`src/llama-kv-cache-dsa.{h,cpp}`) is correct; `src/models/deepseek32.cpp` exists (not merged into deepseek2.cpp).
- The indexer **lives only in `src/models/deepseek32.cpp`** — searches for `indexer`/`lightning`/`top_k` in `src/models/` returned matches there and (as tensor loads only) in `glm-dsa.cpp`.
- `ggml_gen_hadamard` / `ggml_mul_mat_aux` are `static` (file-local) in `llama-kv-cache.cpp` — you will need to copy them into ik's kv-cache TU.
- The Hadamard hint constant `GGML_HINT_SRC0_IS_HADAMARD` and `ggml_mul_mat_set_hint` must exist in ik's ggml; if not, that's an additional ggml-side port (the hint lets the mul_mat kernel know src0 is a dense ±1/√n matrix). The indexer graph (§1) applies the rotation with a bare `ggml_mul_mat` (no hint), so the hint is only strictly needed for the quantized-KV `ggml_mul_mat_aux` path.
- Local copies of all fetched files are in the scratchpad dir if you want to diff them later.