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

    post_norm_data * pnd_ptr = nullptr;

    bool add_input = model.split_mode == LLAMA_SPLIT_MODE_GRAPH ? false : true;

    int n_active_layer = hparams.n_layer - hparams.nextn_predict_layers;

    std::vector<ggml_tensor *> pn_tensors;

    auto do_post_norm = [&] (ggml_tensor * cur, ggml_tensor * post_norm, ggml_tensor * inp, const std::string & tag, int il, bool get_rows) {
        GGML_ASSERT(cur->op == GGML_OP_REDUCE);
        int n = cur->op_params[1];
        if ((int)pn_tensors.size() != n) pn_tensors.resize(n);
        for (int id = 0; id < n; ++id) {
            if (!cur->src[id]) {
                pn_tensors[id] = nullptr;
                continue;
            }
            auto pn_extra = (ggml_split_tensor_t *)post_norm->extra;
            GGML_ASSERT(pn_extra && pn_extra->splits[id]);
            auto normed = ggml_fused_rms_norm(ctx0, cur->src[id], pn_extra->splits[id], post_norm_eps);
            cb(normed, (tag + "_pn").c_str(), 1000*(il+1) + id);
            auto add = get_input_tensor_sm_graph(ctx0, inp, id);
            if (get_rows && il == n_active_layer - 1 && inp_out_ids) {
                add = ggml_get_rows(ctx0, add, inp_out_ids);
            }
            auto added = ggml_add(ctx0, normed, add);
            cb(added, (tag + "_pn_add").c_str(), 1000*(il+1) + id);
            pn_tensors[id] = added;
        }
        cur = ggml_reduce(ctx0, pn_tensors.data(), n, GGML_OP_ADD);
        cb(cur, (tag + "_final").c_str(), il);
        cur->op_params[3] = 1;
        ggml_build_forward_expand(gf, cur);
        return cur;
    };

    for (int il = 0; il < n_active_layer; ++il) {

        bool use_rope = hparams.swa_layers[il];
        auto this_KQ_mask = use_rope ? KQ_mask_swa : KQ_mask;
        int this_n_swa = use_rope ? hparams.n_swa : 0;

        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_active_layer - 1 ? inp_out_ids : nullptr, nullptr,
                this_KQ_mask, nullptr, nullptr, kq_scale, 0.0f, this_n_swa, il, use_rope, false, add_input, false, false,
                model.layers[il].attn_post_norm, -1, post_norm_eps, pnd_ptr);

        if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH) {
            cur = do_post_norm(cur, model.layers[il].attn_post_norm, inpL, "attn", il, true);
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

        if (model.split_mode == LLAMA_SPLIT_MODE_GRAPH) {
            cur = do_post_norm(cur, model.layers[il].ffn_post_norm, ffn_inp, "ffn", il, false);
        }

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

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

