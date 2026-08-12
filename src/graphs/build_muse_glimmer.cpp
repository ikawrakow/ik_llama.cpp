#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_muse_glimmer() {
    ggml_cgraph * gf = new_graph_custom();

    // TODO: propagate this for the post norm ops
    const float post_norm_eps = 1e-8f;

    // mutable variable, needed during the last layer of the computation to skip unused tokens
    int32_t n_tokens = this->n_tokens;

    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    inpL = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
    cb(inpL, "inpL_normed", -1);

    struct ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * KQ_mask_swa = build_inp_KQ_mask_swa();

    auto inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * ffn_inp = nullptr;

    post_norm_data pnd;
    pnd.f_rms_eps = post_norm_eps;
    post_norm_data * pnd_ptr = nullptr;

    bool add_input = model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? false : true;

    int n_active_layer = hparams.n_layer - hparams.nextn_predict_layers;

    for (int il = 0; il < n_active_layer; ++il) {

        bool use_rope = hparams.swa_layers[il];
        auto this_KQ_mask = use_rope ? KQ_mask_swa : KQ_mask;
        int this_n_swa = use_rope ? hparams.n_swa : 0;

        if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH && il > 0) {
            GGML_ASSERT(pnd.next_input.size() == model.devices.size());
            pnd.norm = model.layers[il-1].ffn_post_norm;
            pnd_ptr = &pnd;
        }

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_active_layer - 1 ? inp_out_ids : nullptr, nullptr,
                this_KQ_mask, nullptr, nullptr, kq_scale, 0.0f, this_n_swa, il, use_rope, false, add_input, false, false,
                model.layers[il].attn_post_norm, -1, post_norm_eps, pnd_ptr);

        if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH) {
            pnd_ptr = &pnd;
            if (il == 0) {
                pnd.next_input.resize(model.devices.size(), inpL);
            } else {
                GGML_ASSERT(pnd.next_input.size() == model.devices.size());
            }
            pnd.norm = model.layers[il].attn_post_norm;
        }

        ffn_inp = cur;

        cur = llm_build_ffn(ctx0, lctx, model.layers[il].ffn_norm, ffn_inp,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf, add_input, false, nullptr,
                    model.layers[il].ffn_post_norm, post_norm_eps, pnd_ptr);
        cb(cur, "ffn_out", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH) {
        GGML_ASSERT(inpL->op == GGML_OP_REDUCE);
        int idx = model.default_layer_device[n_active_layer];
        cur = inpL->src[idx];
        if (!cur) {
            for (idx = 0; idx < int(model.devices.size()); ++idx) {
                if (inpL->src[idx]) {
                    cur = inpL->src[idx]; break;
                }
            }
            GGML_ASSERT(cur);
        }
        auto pn_extra = (ggml_split_tensor_t *)model.layers[n_active_layer-1].ffn_post_norm->extra;
        GGML_ASSERT(pn_extra && pn_extra->splits[idx]);
        cur = ggml_fused_rms_norm(ctx0, cur, pn_extra->splits[idx], pnd.f_rms_eps);
        cb(cur, "ffn_post_norm", n_active_layer-1);
        GGML_ASSERT(idx < (int)pnd.next_input.size());
        auto add = pnd.next_input[idx];
        if (!add) {
            for (int j = 0; j < int(pnd.next_input.size()); ++j) {
                if (pnd.next_input[j]) {
                    add = pnd.next_input[j]; break;
                }
            }
            GGML_ASSERT(add);
        }
        cur = ggml_add(ctx0, cur, add);
        cb(cur, "ffn_final", -1);
    }

    // lm_head
    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cur = ggml_scale(ctx0, cur, hparams.f_logit_scale);
    cb(cur, "output_scaled", -1);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_softcap(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

