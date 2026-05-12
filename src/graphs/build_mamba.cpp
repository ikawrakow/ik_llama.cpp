#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

ggml_cgraph * llm_build_context::build_mamba() {
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, model.max_nodes(n_tokens), false);

    const int64_t d_model = n_embd;
    const int64_t d_conv  = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    GGML_ASSERT(2 * d_model == d_inner);
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t dt_rank = hparams.ssm_dt_rank;

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    struct ggml_tensor * state_mask = build_inp_s_mask();
    struct ggml_tensor * state_seq  = build_inp_s_seq();

    for (int il = 0; il < n_layer; ++il) {
        // (ab)using the KV cache to store the states
        struct ggml_tensor * conv_states = ggml_reshape_2d(ctx0, kv_self.k_l[il], hparams.n_embd_k_s(), kv_self.size);
        struct ggml_tensor * ssm_states  = ggml_reshape_2d(ctx0, kv_self.v_l[il], hparams.n_embd_v_s(), kv_self.size);

        // clear states of sequences which are starting at the beginning of this batch
        {
            conv_states = ggml_mul(ctx0,
                    ggml_view_2d(ctx0, conv_states, conv_states->ne[0], n_kv, conv_states->nb[1], kv_head*conv_states->nb[1]),
                    state_mask);
            ssm_states  = ggml_mul(ctx0,
                    ggml_view_2d(ctx0, ssm_states, ssm_states->ne[0], n_kv, ssm_states->nb[1], kv_head*ssm_states->nb[1]),
                    state_mask);
        }

        conv_states = ggml_reshape_3d(ctx0, conv_states, d_conv - 1, d_inner, n_kv);
        ssm_states  = ggml_reshape_3d(ctx0,  ssm_states,    d_state, d_inner, n_kv);

        // norm
        cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        // {n_embd, 2*d_inner} * {n_embd, n_tokens} => {2*d_inner, n_tokens}
        struct ggml_tensor * xz = llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_in, cur);
        // split the above in two
        // => {d_inner, n_tokens}
        struct ggml_tensor * x = ggml_view_2d(ctx0, xz, d_inner, xz->ne[1], xz->nb[1], 0);
        struct ggml_tensor * z = ggml_view_2d(ctx0, xz, d_inner, xz->ne[1], xz->nb[1], ggml_element_size(xz)*d_inner);

        // conv
        {
            // Custom operator which is needed only to ease simultaneous sequence processing.
            // For a single sequence, the equivalent is to concatenate the columns of conv_states and x,
            // then make a self-overlapping view of that over d_conv columns at each stride in the 3rd dimension,
            // then element-wise multiply that with the conv1d weigth,
            // then sum the elements of each row,
            // (the last two steps are a dot product over rows (also doable with mul_mat))
            // then permute away the ne[0] dimension,
            // and then you're left with the resulting x tensor.
            // The new conv_states is the last (d_conv - 1) columns
            // of the last 3rd dimensional "layer" of the self-overlapping view.
            // For simultaneous sequences, it's more complicated.
            struct ggml_tensor * x_conv = ggml_ssm_conv(ctx0, conv_states, x, model.layers[il].ssm_conv1d, state_seq, nullptr);

            // store last (d_conv - 1) columns of the conv_state part of x_conv back into the KV cache
            ggml_build_forward_expand(gf,
                    ggml_cpy(ctx0,
                        ggml_view_2d(ctx0, x_conv, d_conv - 1, d_inner*n_kv, d_conv*ggml_element_size(x_conv), (1+d_inner*n_tokens)*ggml_element_size(x_conv)),
                        ggml_view_1d(ctx0, kv_self.k_l[il], (d_conv - 1)*(d_inner)*(n_kv), kv_head*(d_conv - 1)*(d_inner)*ggml_element_size(x_conv))));

            // extract x from x_conv
            x = ggml_view_2d(ctx0, x_conv, d_inner, n_tokens, d_inner*ggml_element_size(x_conv), 0);

            // bias
            x = ggml_add(ctx0, x, model.layers[il].ssm_conv1d_b);

            x = ggml_silu(ctx0, x);
        }

        // ssm
        {
            // {d_inner, dt_rank + 2*d_state} * {d_inner, n_tokens} => {dt_rank + 2*d_state, n_tokens}
            struct ggml_tensor * x_db = llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_x, x);
            // split
            struct ggml_tensor * dt = ggml_view_2d(ctx0, x_db, dt_rank, n_tokens, x_db->nb[1], 0);
            struct ggml_tensor * B  = ggml_view_2d(ctx0, x_db, d_state, n_tokens, x_db->nb[1], ggml_element_size(x_db)*dt_rank);
            struct ggml_tensor * C  = ggml_view_2d(ctx0, x_db, d_state, n_tokens, x_db->nb[1], ggml_element_size(x_db)*(dt_rank+d_state));

            // {dt_rank, d_inner} * {dt_rank, n_tokens} => {d_inner, n_tokens}
            dt = llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_dt, dt);
            dt = ggml_add(ctx0, dt, model.layers[il].ssm_dt_b);

            // Custom operator to optimize the parallel associative scan
            // as described in the Annex D of the Mamba paper.
            // => {d_inner, n_tokens} and {d_state, d_inner, n_kv} combined,
            // because only a single tensor can be returned.
            struct ggml_tensor * y_ssm_states = ggml_ssm_scan(ctx0, ssm_states, x, dt, model.layers[il].ssm_a, B, C, state_seq);

            // store last states (the second part of y_ssm_states)
            ggml_build_forward_expand(gf,
                    ggml_cpy(ctx0,
                        ggml_view_1d(ctx0, y_ssm_states, d_state*d_inner*n_kv, d_inner*n_tokens*ggml_element_size(y_ssm_states)),
                        ggml_view_1d(ctx0, kv_self.v_l[il], d_state*d_inner*n_kv, kv_head*d_state*d_inner*ggml_element_size(ssm_states))));

            struct ggml_tensor * y = ggml_view_2d(ctx0, y_ssm_states, d_inner, n_tokens, d_inner*ggml_element_size(y_ssm_states), 0);

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor * inp_out_ids = build_inp_out_ids();
                x    = ggml_get_rows(ctx0,    x, inp_out_ids);
                y    = ggml_get_rows(ctx0,    y, inp_out_ids);
                z    = ggml_get_rows(ctx0,    z, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // {d_inner, n_tokens} * {d_inner} => {d_inner, n_tokens}
            y = ggml_add(ctx0, y, ggml_mul(ctx0, x, model.layers[il].ssm_d));
            y = ggml_mul(ctx0, y, ggml_silu(ctx0, z));
            cb(y, "y", il);

            // {d_inner, n_embd} * {d_inner, n_tokens} => {n_embd, n_tokens}
            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].ssm_out, y);
        }

        // residual
        cur = ggml_add(ctx0, cur, inpL);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    // final rmsnorm
    cur = llm_build_norm(ctx0, inpL, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
    cb(cur, "result_norm", -1);

    // lm_head
    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
