#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <cmath>

// clef: Qwen3.5 backbone (see build_qwen35()) with a joint decision head on top of its hidden states
// ref: JointSchemaHead in joint_schema_model.py of the model repo

// the spans read by the head, [start, end) in ubatch token indices, see llama_set_decision_order()
// if the ubatch has no usable order, there is one empty question with one empty option
void llm_build_context::llama_decision_get_spans(const llama_context & lctx, const llama_batch & batch, llama_decision_spans & spans) {
    spans.q_type.clear();
    spans.q_start.clear();
    spans.q_end.clear();
    spans.o_question.clear();
    spans.o_start.clear();
    spans.o_end.clear();
    spans.valid = false;

    const int32_t * order = lctx.decision_order_cur;
    const int32_t n_tokens = batch.n_tokens;

    // the head needs the whole prompt of one sequence, the tokens of an image share a position
    bool ok = order != nullptr && batch.token != nullptr && batch.pos != nullptr && batch.n_seq_id != nullptr && batch.seq_id != nullptr;
    ok = ok && batch.pos[0] == 0;
    for (int32_t i = 0; ok && i < n_tokens; ++i) {
        ok = (i == 0 || batch.pos[i] >= batch.pos[i - 1]) && batch.n_seq_id[i] == 1 && batch.seq_id[i][0] == batch.seq_id[0][0];
    }

    std::vector<bool> has_option;
    for (int32_t i = 0; ok && i < n_tokens;) {
        const int32_t cur = order[i];
        int32_t end = i + 1;
        while (end < n_tokens && order[end] == cur) {
            end++;
        }
        switch (cur) {
            case LLAMA_DECISION_ORDER_NONE:
                break;
            case LLAMA_DECISION_ORDER_QUESTION_NOUL:
            case LLAMA_DECISION_ORDER_QUESTION_CHOICE:
            case LLAMA_DECISION_ORDER_QUESTION_SCORE:
                spans.q_type.push_back(cur - LLAMA_DECISION_ORDER_QUESTION_NOUL);
                spans.q_start.push_back(i);
                spans.q_end.push_back(end);
                has_option.push_back(false);
                break;
            case LLAMA_DECISION_ORDER_OPTION:
                ok = !spans.q_type.empty();
                if (ok) {
                    spans.o_question.push_back((int32_t) spans.q_type.size() - 1);
                    spans.o_start.push_back(i);
                    spans.o_end.push_back(end);
                    has_option.back() = true;
                }
                break;
            default:
                ok = false;
        }
        i = end;
    }

    // the head reads the token ids of the spans, they cannot be embeddings
    for (int32_t i = 0; ok && batch.embd && i < n_tokens; ++i) {
        ok = batch.token[i] >= 0 || order[i] == LLAMA_DECISION_ORDER_NONE;
    }

    // each question needs an option, the scores are written to a row of logits
    ok = ok && !spans.q_type.empty() && (int64_t) spans.o_question.size() <= (int64_t) lctx.model.hparams.n_vocab;
    for (size_t i = 0; ok && i < has_option.size(); ++i) {
        ok = has_option[i];
    }

    spans.valid = ok;
    if (!ok) {
        spans.q_type.assign(1, 0);
        spans.q_start.assign(1, 0);
        spans.q_end.assign(1, 0);
        spans.o_question.assign(1, 0);
        spans.o_start.assign(1, 0);
        spans.o_end.assign(1, 0);
    }
}

void llm_build_context::llama_decision_set_inputs(const llama_context & lctx, const llama_batch & batch) {
    llama_decision_spans spans;
    llama_decision_get_spans(lctx, batch, spans);

    const int64_t n_tokens    = batch.n_tokens;
    const int64_t n_questions = spans.q_type.size();
    const int64_t n_options   = spans.o_question.size();

    GGML_ASSERT(lctx.inp_decision_pool_q->ne[1] == n_questions && lctx.inp_decision_pool_o->ne[1] == n_options);

    // the scores are NaN if an order was given that cannot be used
    const bool has_order = lctx.decision_order_cur != nullptr || lctx.decision_order_bad;
    const float status = spans.valid || !has_order ? 0.0f : NAN;
    ggml_backend_tensor_set(lctx.inp_decision_status, &status, 0, sizeof(float));

    std::vector<float> pool_q(n_tokens * n_questions, 0.0f);
    for (int64_t i = 0; i < n_questions; ++i) {
        for (int32_t j = spans.q_start[i]; j < spans.q_end[i]; ++j) {
            pool_q[i * n_tokens + j] = 1.0f / (spans.q_end[i] - spans.q_start[i]);
        }
    }

    std::vector<float> pool_o(n_tokens * n_options, 0.0f);
    std::vector<float> mask(n_options * n_questions, -INFINITY);
    for (int64_t i = 0; i < n_options; ++i) {
        for (int32_t j = spans.o_start[i]; j < spans.o_end[i]; ++j) {
            pool_o[i * n_tokens + j] = 1.0f / (spans.o_end[i] - spans.o_start[i]);
        }
        mask[spans.o_question[i] * n_options + i] = 0.0f;
    }

    ggml_backend_tensor_set(lctx.inp_decision_pool_q, pool_q.data(),           0, ggml_nbytes(lctx.inp_decision_pool_q));
    ggml_backend_tensor_set(lctx.inp_decision_pool_o, pool_o.data(),           0, ggml_nbytes(lctx.inp_decision_pool_o));
    ggml_backend_tensor_set(lctx.inp_decision_qtype,  spans.q_type.data(),     0, ggml_nbytes(lctx.inp_decision_qtype));
    ggml_backend_tensor_set(lctx.inp_decision_owner,  spans.o_question.data(), 0, ggml_nbytes(lctx.inp_decision_owner));
    ggml_backend_tensor_set(lctx.inp_decision_mask,   mask.data(),             0, ggml_nbytes(lctx.inp_decision_mask));
}

ggml_tensor * llm_build_context::build_clef_head_norm(ggml_tensor * cur, ggml_tensor * w, ggml_tensor * b) {
    return llm_build_norm(ctx0, cur, hparams, w, b, LLM_NORM, cb, -1);
}

// torch.nn.MultiheadAttention without mask, q: [n_embd_h, n_q], kv: [n_embd_h, n_kv]
ggml_tensor * llm_build_context::build_clef_head_attn(ggml_tensor * q, ggml_tensor * kv, const llama_decision_attn & a) {
    const int64_t n_embd_h = q->ne[0];
    const int64_t n_head_h = hparams.n_head_decision;
    const int64_t d_head   = n_embd_h / n_head_h;
    const int64_t n_q      = q->ne[1];
    const int64_t n_kv     = kv->ne[1];

    ggml_tensor * Qcur = ggml_add(ctx0, ggml_mul_mat(ctx0, a.wq, q),  a.bq);
    ggml_tensor * Kcur = ggml_add(ctx0, ggml_mul_mat(ctx0, a.wk, kv), a.bk);
    ggml_tensor * Vcur = ggml_add(ctx0, ggml_mul_mat(ctx0, a.wv, kv), a.bv);

    Qcur = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Qcur, d_head, n_head_h, n_q),  0, 2, 1, 3)); // [d_head, n_q,  n_head]
    Kcur = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Kcur, d_head, n_head_h, n_kv), 0, 2, 1, 3)); // [d_head, n_kv, n_head]
    Vcur = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, Vcur, d_head, n_head_h, n_kv), 1, 2, 0, 3)); // [n_kv, d_head, n_head]

    ggml_tensor * kq = ggml_mul_mat(ctx0, Kcur, Qcur); // [n_kv, n_q, n_head]
    kq = ggml_soft_max_ext(ctx0, kq, nullptr, 1.0f / sqrtf(float(d_head)), 0.0f);

    ggml_tensor * cur = ggml_mul_mat(ctx0, Vcur, kq); // [d_head, n_q, n_head]
    cur = ggml_cont_2d(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3), n_embd_h, n_q);

    return ggml_add(ctx0, ggml_mul_mat(ctx0, a.wo, cur), a.bo);
}

ggml_tensor * llm_build_context::build_clef_head_ffn(ggml_tensor * cur, const llama_decision_layer & layer) {
    cur = build_clef_head_norm(cur, layer.ffn_norm, layer.ffn_norm_b);
    cur = ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_up, cur), layer.ffn_up_b);
    cur = ggml_gelu_erf(ctx0, cur);
    return ggml_add(ctx0, ggml_mul_mat(ctx0, layer.ffn_down, cur), layer.ffn_down_b);
}

// hidden: [n_embd, n_tokens], the normed hidden states of all the tokens of the prompt
// returns the logits [n_vocab, n_outputs], the first n_options values of each row are the scores of the options
ggml_tensor * llm_build_context::build_clef_head(ggml_tensor * hidden) {
    const auto & head = model.decision;

    llama_decision_spans spans;
    llama_decision_get_spans(lctx, batch, spans);

    const int64_t n_questions = spans.q_type.size();
    const int64_t n_options   = spans.o_question.size();

    lctx.inp_decision_pool_q = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_questions);
    lctx.inp_decision_pool_o = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_options);
    lctx.inp_decision_qtype  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_questions);
    lctx.inp_decision_owner  = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_options);
    lctx.inp_decision_mask   = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_options, n_questions);
    lctx.inp_decision_status = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_input(lctx.inp_decision_pool_q);
    ggml_set_input(lctx.inp_decision_pool_o);
    ggml_set_input(lctx.inp_decision_qtype);
    ggml_set_input(lctx.inp_decision_owner);
    ggml_set_input(lctx.inp_decision_mask);
    ggml_set_input(lctx.inp_decision_status);

    hidden = build_clef_head_norm(hidden, head.hidden_norm, head.hidden_norm_b);

    ggml_tensor * memory = ggml_mul_mat(ctx0, head.proj_memory, hidden); // [n_embd_h, n_tokens]
    const int64_t n_embd_h = memory->ne[0];

    // hidden state of the last token
    ggml_tensor * global = ggml_view_2d(ctx0, hidden, n_embd, 1, hidden->nb[1], (n_tokens - 1) * hidden->nb[1]);

    // mean of the hidden states over each span
    ggml_tensor * hidden_t = ggml_cont(ctx0, ggml_transpose(ctx0, hidden));
    ggml_tensor * question = ggml_mul_mat(ctx0, hidden_t, lctx.inp_decision_pool_q); // [n_embd, n_questions]
    ggml_tensor * opt_ctx  = ggml_mul_mat(ctx0, hidden_t, lctx.inp_decision_pool_o); // [n_embd, n_options]

    // mean of the output embeddings of the tokens of each option
    ggml_tensor * lexical;
    if (lctx.inp_tokens) {
        lexical = ggml_get_rows(ctx0, model.output, lctx.inp_tokens);
        lexical = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, lexical)), lctx.inp_decision_pool_o); // [n_embd, n_options]
    } else {
        // embeddings input: there are no tokens, the order is not usable anyway
        lexical = ggml_scale(ctx0, opt_ctx, 0.0f);
    }

    // one query per option
    ggml_tensor * options = ggml_mul_mat(ctx0, head.proj_option_context, opt_ctx);
    options = ggml_add(ctx0, options, ggml_mul_mat(ctx0, head.proj_option_lexical, lexical));
    options = ggml_add(ctx0, options, ggml_get_rows(ctx0, ggml_mul_mat(ctx0, head.proj_option_question, question), lctx.inp_decision_owner));

    // the options read the prompt
    for (uint32_t il = 0; il < hparams.n_layer_decision_routing; ++il) {
        const auto & layer = head.layers[il];

        ggml_tensor * cur = build_clef_head_attn(
                build_clef_head_norm(options, layer.cross_norm,    layer.cross_norm_b),
                build_clef_head_norm(memory,  layer.cross_norm_kv, layer.cross_norm_kv_b),
                layer.cross_attn);
        options = ggml_add(ctx0, options, cur);
        options = ggml_add(ctx0, options, build_clef_head_ffn(options, layer));
    }
    cb(options, "decision_options", -1);

    // one vector per question: its text, a summary of its options, the end of the prompt and its type
    ggml_tensor * fields = ggml_mul_mat(ctx0, head.proj_question, question); // [n_embd_h, n_questions]
    {
        // each question weights its own options
        ggml_tensor * weights = ggml_mul_mat(ctx0, options, fields); // [n_options, n_questions]
        weights = ggml_scale(ctx0, weights, 1.0f / sqrtf(float(n_embd_h)));
        weights = ggml_soft_max(ctx0, ggml_add(ctx0, weights, lctx.inp_decision_mask));

        ggml_tensor * summary = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, options)), weights); // [n_embd_h, n_questions]

        fields = ggml_add(ctx0, fields, build_clef_head_norm(summary, head.option_summary_norm, head.option_summary_norm_b));
        fields = ggml_add(ctx0, fields, ggml_mul_mat(ctx0, head.proj_global, global));
        fields = ggml_add(ctx0, fields, ggml_get_rows(ctx0, head.type_embd, lctx.inp_decision_qtype));
    }

    // the questions read each other and the prompt
    for (uint32_t il = hparams.n_layer_decision_routing; il < head.layers.size(); ++il) {
        const auto & layer = head.layers[il];

        ggml_tensor * cur = build_clef_head_norm(fields, layer.self_norm, layer.self_norm_b);
        fields = ggml_add(ctx0, fields, build_clef_head_attn(cur, cur, layer.self_attn));

        cur = build_clef_head_norm(fields, layer.cross_norm, layer.cross_norm_b);
        fields = ggml_add(ctx0, fields, build_clef_head_attn(cur, memory, layer.cross_attn));

        fields = ggml_add(ctx0, fields, build_clef_head_ffn(fields, layer));
    }
    fields = build_clef_head_norm(fields, head.field_norm, head.field_norm_b);
    cb(fields, "decision_fields", -1);

    const float eps = 1e-12f;

    // prior: the output embeddings of the option against the question and the end of the prompt
    ggml_tensor * anchor = ggml_l2_norm(ctx0, ggml_add(ctx0, question, global), eps);
    anchor = ggml_get_rows(ctx0, anchor, lctx.inp_decision_owner);
    ggml_tensor * prior = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_l2_norm(ctx0, lexical, eps), anchor)); // [1, n_options]

    // joint: each option against the vector of its question
    options = build_clef_head_norm(options, head.option_norm, head.option_norm_b);
    ggml_tensor * field = ggml_get_rows(ctx0, fields, lctx.inp_decision_owner); // [n_embd_h, n_options]

    ggml_tensor * cosine = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_l2_norm(ctx0, field, eps), ggml_l2_norm(ctx0, options, eps)));

    ggml_tensor * features = ggml_concat(ctx0, field, options, 0);
    features = ggml_concat(ctx0, features, ggml_mul(ctx0, field, options), 0);
    features = ggml_concat(ctx0, features, ggml_abs(ctx0, ggml_sub(ctx0, field, options)), 0);

    ggml_tensor * residual = ggml_add(ctx0, ggml_mul_mat(ctx0, head.scorer, features), head.scorer_b);
    residual = ggml_gelu_erf(ctx0, residual);
    residual = ggml_add(ctx0, ggml_mul_mat(ctx0, head.scorer_out, residual), head.scorer_out_b); // [1, n_options]

    ggml_tensor * scale_prior    = ggml_view_1d(ctx0, head.scales, 1, 0);
    ggml_tensor * scale_joint    = ggml_view_1d(ctx0, head.scales, 1, 1 * ggml_element_size(head.scales));
    ggml_tensor * scale_residual = ggml_view_1d(ctx0, head.scales, 1, 2 * ggml_element_size(head.scales));

    ggml_tensor * joint = ggml_add(ctx0, ggml_mul(ctx0, cosine, scale_joint), residual);
    ggml_tensor * cur   = ggml_add(ctx0, ggml_mul(ctx0, prior, scale_prior), ggml_mul(ctx0, joint, scale_residual));

    // the scores are written at the start of each row of logits
    cur = ggml_reshape_2d(ctx0, cur, n_options, 1);
    cur = ggml_add(ctx0, cur, lctx.inp_decision_status);
    cur = ggml_pad(ctx0, cur, hparams.n_vocab - n_options, 0, 0, 0);
    if (n_outputs > 1) {
        cur = ggml_repeat_4d(ctx0, cur, hparams.n_vocab, n_outputs, 1, 1);
    }
    return cur;
}
