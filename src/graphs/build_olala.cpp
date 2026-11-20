#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <vector>

// Olala 7A1B: Mamba3-MIMO and Differential-TPA blocks, geodesic residual, latent MoE.

namespace {

// Mamba3-MIMO prefill chunk size (64/R); 0 = single chunk.
static const int64_t olala_chunk_size = []() -> int64_t {
    const char * e = std::getenv("IK_OLALA_CHUNK_SIZE");
    return e && e[0] ? (int64_t) std::strtoll(e, nullptr, 10) : (int64_t) 16;
}();

// cos/sin through timestep embedding (CPU and CUDA, no trig op).
static void cos_sin(ggml_context * ctx, ggml_tensor * x, ggml_tensor ** c, ggml_tensor ** s) {
    const int64_t n = ggml_nelements(x);
    ggml_tensor * xc   = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
    ggml_tensor * flat = ggml_reshape_1d(ctx, xc, n);
    ggml_tensor * emb  = ggml_timestep_embedding(ctx, flat, 2, 10000);
    const size_t es = ggml_element_size(emb);
    ggml_tensor * vc = ggml_view_2d(ctx, emb, 1, n, emb->nb[1], 0);
    ggml_tensor * vs = ggml_view_2d(ctx, emb, 1, n, emb->nb[1], es);
    if (!ggml_is_contiguous(vc)) vc = ggml_cont(ctx, vc);
    if (!ggml_is_contiguous(vs)) vs = ggml_cont(ctx, vs);
    *c = ggml_reshape_4d(ctx, vc, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    *s = ggml_reshape_4d(ctx, vs, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

ggml_tensor * geodesic(ggml_context * ctx, ggml_tensor * x, ggml_tensor * g,
        ggml_tensor * scale, ggml_tensor * bias, int il) {
    constexpr float k_pi4 = 0.785398163f;

    ggml_tensor * x_norm_sq = ggml_clamp(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, x)), 1e-12f, FLT_MAX);
    ggml_tensor * proj      = ggml_div(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, x, g)), x_norm_sq);
    ggml_tensor * grad      = ggml_fma(ctx, g, x, ggml_scale(ctx, proj, -1.0f));
    ggml_tensor * tan_norm  = ggml_clamp(ctx, ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, grad))), 1e-8f, FLT_MAX);
    ggml_tensor * radius    = ggml_sqrt(ctx, x_norm_sq);

    ggml_tensor * theta = ggml_clamp(ctx, ggml_div(ctx, tan_norm, radius), -FLT_MAX, k_pi4);
    theta = ggml_mul(ctx, theta, scale);
    theta = ggml_add(ctx, theta, bias);
    theta = ggml_scale(ctx, theta, 1.0f/(float) (il + 1));
    theta = ggml_clamp(ctx, theta, -FLT_MAX, k_pi4);

    ggml_tensor * c, * s;
    cos_sin(ctx, theta, &c, &s);
    // x·cos + g_orth·(R·sin/||g_orth||); g_orth vanishes when x and g align.
    ggml_tensor * coeff = ggml_div(ctx, ggml_mul(ctx, radius, s), tan_norm);
    return ggml_fma(ctx, ggml_mul(ctx, x, c), grad, coeff);
}

// Chunked Mamba3-MIMO recurrence.
static ggml_tensor * mamba3_mimo_chunked(
        ggml_context * ctx,
        ggml_tensor * q_biased,    // (D_qk, R, H, L)
        ggml_tensor * k_biased,    // (D_qk, R, H, L)
        ggml_tensor * v_proj,      // (D_v,  R, H, L)
        ggml_tensor * z_proj,      // (D_v,  R, H, L)
        ggml_tensor * log_alpha,   // (H, L) ADT = log(alpha)
        ggml_tensor * dt,          // (H, L)
        ggml_tensor * trap_post,   // (H, L)
        ggml_tensor * ang_raw,     // (num_angles, L)
        ggml_tensor * D_skip,      // (H,)
        ggml_tensor * mimo_o,      // (D_v, R, H)
        int64_t D_qk, int64_t D_v, int64_t R, int64_t H, int64_t L,
        int64_t num_angles,
        int64_t CS,
        int il,
        ggml_tensor * state_in_packed = nullptr,
        ggml_tensor ** state_out_packed = nullptr) {
    (void) il;
    const int64_t quarter = D_qk / 4;
    GGML_ASSERT(num_angles == quarter && "halved rotary expects num_angles = D_qk/4");
    GGML_ASSERT(CS > 0 && CS <= L);
    constexpr float k_pi = 3.14159265358979323846f;

    // Must match llama_hparams::n_embd_s().
    const int64_t off_K_glob   = H * D_v * D_qk;
    const int64_t off_V_glob   = off_K_glob + H * R * D_qk;
    const int64_t off_ang_glob = off_V_glob + H * R * D_v;

    ggml_tensor * state_in_4d  = nullptr;
    ggml_tensor * K_in_4d      = nullptr;
    ggml_tensor * V_in_4d      = nullptr;
    ggml_tensor * angle_in_2d  = nullptr;
    if (state_in_packed != nullptr) {
        const size_t fs = ggml_element_size(state_in_packed);
        ggml_tensor * sp = ggml_reshape_1d(ctx, state_in_packed,
                                            off_ang_glob + H * num_angles);
        auto slice = [&](int64_t off, int64_t n) {
            return ggml_view_1d(ctx, sp, n, off * fs);
        };
        state_in_4d = ggml_reshape_4d(ctx, slice(0,                H*D_v*D_qk),    D_qk, D_v, H, 1);
        K_in_4d     = ggml_reshape_4d(ctx, slice(off_K_glob,       H*R  *D_qk),    D_qk, R,   H, 1);
        V_in_4d     = ggml_reshape_4d(ctx, slice(off_V_glob,       H*R  *D_v ),    D_v,  R,   H, 1);
        angle_in_2d = ggml_reshape_2d(ctx, slice(off_ang_glob,     H*num_angles),  num_angles, H);
    }

    ggml_tensor * ang_tanh = ggml_tanh(ctx, ang_raw);
    ggml_tensor * ang_pi   = ggml_scale(ctx, ang_tanh, k_pi);
    ggml_tensor * ang_3d   = ggml_reshape_3d(ctx, ang_pi, num_angles, 1, L);
    ggml_tensor * ang_b    = ggml_repeat_4d(ctx, ang_3d, num_angles, H, L, 1);
    ggml_tensor * dt_3d    = ggml_reshape_3d(ctx, dt, 1, H, L);
    ggml_tensor * contrib  = ggml_mul(ctx, ang_b, dt_3d);
    ggml_tensor * contrib_lt = ggml_cont(ctx, ggml_permute(ctx, contrib, 1, 2, 0, 3));
    ggml_tensor * cum_lt   = ggml_cumsum(ctx, contrib_lt);
    if (angle_in_2d != nullptr) {
        ggml_tensor * angle_in_1lh = ggml_reshape_3d(ctx, angle_in_2d, 1, num_angles, H);
        ggml_tensor * angle_in_full = ggml_repeat_4d(ctx, angle_in_1lh, L, num_angles, H, 1);
        cum_lt = ggml_add(ctx, cum_lt, angle_in_full);
    }
    ggml_tensor * cos_lt, * sin_lt;
    cos_sin(ctx, cum_lt, &cos_lt, &sin_lt);
    ggml_tensor * cos_a    = ggml_cont(ctx, ggml_permute(ctx, cos_lt, 2, 0, 1, 3));
    ggml_tensor * sin_a    = ggml_cont(ctx, ggml_permute(ctx, sin_lt, 2, 0, 1, 3));

    ggml_tensor * c4  = ggml_reshape_4d(ctx, cos_a, quarter, 1, H, L);
    ggml_tensor * s4  = ggml_reshape_4d(ctx, sin_a, quarter, 1, H, L);
    ggml_tensor * ns4 = ggml_scale(ctx, s4, -1.0f);
    auto apply_rotary = [&](ggml_tensor * x) {
        const size_t es = ggml_element_size(x);
        auto group = [&](int gi) {
            return ggml_view_4d(ctx, x, quarter, R, H, L,
                                x->nb[1], x->nb[2], x->nb[3],
                                gi * quarter * es);
        };
        ggml_tensor * g0 = group(0);
        ggml_tensor * g1 = group(1);
        ggml_tensor * g2 = group(2);
        ggml_tensor * g3 = group(3);
        ggml_tensor * g0_new = ggml_fma(ctx, ggml_mul(ctx, g0, c4), g2, ns4);
        ggml_tensor * g2_new = ggml_fma(ctx, ggml_mul(ctx, g0, s4), g2, c4);
        ggml_tensor * lo = ggml_concat(ctx, g0_new, g1, /*dim=*/ 0);
        ggml_tensor * hi = ggml_concat(ctx, g2_new, g3, /*dim=*/ 0);
        return ggml_concat(ctx, lo, hi, /*dim=*/ 0);
    };
    ggml_tensor * q_rot = apply_rotary(q_biased);  // (D_qk, R, H, L)
    ggml_tensor * k_rot = apply_rotary(k_biased);

    ggml_tensor * log_a    = log_alpha;
    ggml_tensor * log_a_LH = ggml_cont(ctx, ggml_transpose(ctx, log_a));  // (L, H)

    ggml_tensor * gamma          = ggml_mul(ctx, trap_post, dt);
    ggml_tensor * one_minus_trap = ggml_scale_bias(ctx, trap_post, -1.0f, 1.0f);
    ggml_tensor * coeff_prev     = ggml_mul(ctx, one_minus_trap, dt);
    ggml_tensor * gamma_shifted;
    if (L > 1) {
        ggml_tensor * tail = ggml_view_2d(ctx, coeff_prev, H, L - 1,
                                          coeff_prev->nb[1], coeff_prev->nb[1]);
        gamma_shifted = ggml_pad(ctx, ggml_cont(ctx, tail), 0, 1, 0, 0);
    } else {
        gamma_shifted = ggml_scale(ctx, coeff_prev, 0.0f);
    }
    ggml_tensor * factor = ggml_add(ctx, gamma, gamma_shifted);                                   // (H, L)
    ggml_tensor * factor_LH = ggml_cont(ctx, ggml_transpose(ctx, factor));                        // (L, H)

    ggml_tensor * q_LH = ggml_cont(ctx, ggml_permute(ctx, q_rot,  0, 1, 3, 2));   // (D_qk, R, L, H)
    ggml_tensor * v_LH = ggml_cont(ctx, ggml_permute(ctx, v_proj, 0, 1, 3, 2));   // (D_v,  R, L, H)
    ggml_tensor * z_LH = ggml_cont(ctx, ggml_permute(ctx, z_proj, 0, 1, 3, 2));   // (D_v,  R, L, H)
    ggml_tensor * k_R0 = ggml_cont(ctx, ggml_permute(ctx, k_rot,  1, 0, 2, 3));   // (R, D_qk, H, L)
    ggml_tensor * v_R0 = ggml_cont(ctx, ggml_permute(ctx, v_proj, 1, 0, 2, 3));   // (R, D_v,  H, L)

    const int64_t n_chunks = (L + CS - 1) / CS;
    ggml_tensor * state_carry = nullptr;
    std::vector<ggml_tensor *> y_chunks;
    y_chunks.reserve((size_t) n_chunks);

    ggml_tensor * mimo_4d = ggml_reshape_4d(ctx, mimo_o, D_v, R, 1, H);
    ggml_tensor * D4      = ggml_reshape_4d(ctx, D_skip, 1, 1, 1, H);

    // (K V^T) Q = V (K^T Q): one pass over L instead of D_qk*D_v*R per token.
    ggml_tensor * kq_diag  = ggml_mul_mat(ctx, k_rot, q_rot);      // (R, R, H, L)
    ggml_tensor * qk_diag  = ggml_mul_mat(ctx, v_R0, kq_diag);     // (D_v, R, H, L)
    ggml_tensor * over_all = ggml_mul(ctx, qk_diag,
            ggml_reshape_4d(ctx, gamma_shifted, 1, 1, H, L));
    ggml_tensor * over_LH  = ggml_cont(ctx, ggml_permute(ctx, over_all, 0, 1, 3, 2)); // (D_v,R,L,H)

    // Seed with state_in + (1-trap[0])·dt[0]·(K_in @ V_in).
    if (state_in_4d != nullptr) {
        ggml_tensor * K_R0_in = ggml_cont(ctx, ggml_permute(ctx, K_in_4d, 1, 0, 2, 3));
        ggml_tensor * V_R0_in = ggml_cont(ctx, ggml_permute(ctx, V_in_4d, 1, 0, 2, 3));
        ggml_tensor * prev_kv_in = ggml_mul_mat(ctx, K_R0_in, V_R0_in);                       // (D_qk, D_v, H, 1)
        const size_t fs = ggml_element_size(coeff_prev);
        ggml_tensor * gshift_neg1 = ggml_view_1d(ctx, coeff_prev, H, 0 * fs);                 // (H,)
        ggml_tensor * gshift_neg1_4d = ggml_reshape_4d(ctx,
            ggml_cont(ctx, gshift_neg1), 1, 1, H, 1);
        ggml_tensor * pk_scaled = ggml_mul(ctx, prev_kv_in, gshift_neg1_4d);                  // (D_qk, D_v, H, 1)
        ggml_tensor * state_in_perm = ggml_reshape_4d(ctx, state_in_4d, D_qk, D_v, 1, H);
        ggml_tensor * pk_scaled_p   = ggml_reshape_4d(ctx, pk_scaled,   D_qk, D_v, 1, H);
        state_carry = ggml_add(ctx, state_in_perm, pk_scaled_p);                              // (D_qk, D_v, 1, H)
    }

    for (int64_t c = 0; c < n_chunks; ++c) {
        const int64_t t_start = c * CS;
        const int64_t cs_act  = std::min(CS, L - t_start);

        ggml_tensor * k_R0_c  = ggml_view_4d(ctx, k_R0, R, D_qk, H, cs_act,
                                              k_R0->nb[1], k_R0->nb[2], k_R0->nb[3],
                                              t_start * k_R0->nb[3]);
        ggml_tensor * v_R0_c  = ggml_view_4d(ctx, v_R0, R, D_v, H, cs_act,
                                              v_R0->nb[1], v_R0->nb[2], v_R0->nb[3],
                                              t_start * v_R0->nb[3]);
        ggml_tensor * q_LH_c  = ggml_view_4d(ctx, q_LH, D_qk, R, cs_act, H,
                                              q_LH->nb[1], q_LH->nb[2], q_LH->nb[3],
                                              t_start * q_LH->nb[2]);
        ggml_tensor * v_LH_c  = ggml_view_4d(ctx, v_LH, D_v, R, cs_act, H,
                                              v_LH->nb[1], v_LH->nb[2], v_LH->nb[3],
                                              t_start * v_LH->nb[2]);
        ggml_tensor * z_LH_c  = ggml_view_4d(ctx, z_LH, D_v, R, cs_act, H,
                                              z_LH->nb[1], z_LH->nb[2], z_LH->nb[3],
                                              t_start * z_LH->nb[2]);
        ggml_tensor * log_a_c  = ggml_view_2d(ctx, log_a_LH, cs_act, H,
                                               log_a_LH->nb[1], t_start * log_a_LH->nb[0]);
        ggml_tensor * factor_c = ggml_view_2d(ctx, factor_LH, cs_act, H,
                                               factor_LH->nb[1], t_start * factor_LH->nb[0]);

        ggml_tensor * cum_c = ggml_cumsum(ctx, ggml_cont(ctx, log_a_c));               // (cs, H)
        ggml_tensor * cum_c_t = ggml_reshape_3d(ctx, cum_c, 1, cs_act, H);
        ggml_tensor * cum_c_u = ggml_reshape_3d(ctx, cum_c, cs_act, 1, H);
        ggml_tensor * cum_c_full = ggml_repeat_4d(ctx, cum_c_t, cs_act, cs_act, H, 1);
        ggml_tensor * diff_c = ggml_sub(ctx, cum_c_full, cum_c_u);                     // (cs, cs, H)
        // clamp so exp() cannot overflow on the upper triangle.
        diff_c = ggml_clamp(ctx, diff_c, -FLT_MAX, 0.0f);
        ggml_tensor * W_c    = ggml_tri(ctx, ggml_exp(ctx, diff_c), GGML_TRI_TYPE_LOWER_DIAG);

        ggml_tensor * factor_c_cont = ggml_cont(ctx, factor_c);
        ggml_tensor * factor_c_bc   = ggml_reshape_3d(ctx, factor_c_cont, cs_act, 1, H);
        ggml_tensor * Wf_c          = ggml_mul(ctx, W_c, factor_c_bc);                 // (cs, cs, H)

        ggml_tensor * kv_c = ggml_mul_mat(ctx, k_R0_c, v_R0_c);

        ggml_tensor * kv_perm_c     = ggml_cont(ctx, ggml_permute(ctx, kv_c, 1, 2, 3, 0));
        ggml_tensor * kv_flat_c     = ggml_reshape_3d(ctx, kv_perm_c, cs_act, D_qk * D_v, H);
        ggml_tensor * state_flat_c  = ggml_mul_mat(ctx, kv_flat_c, Wf_c);
        ggml_tensor * state_within_c = ggml_reshape_4d(ctx, state_flat_c, D_qk, D_v, cs_act, H);

        // (S_within + a*S_carry) @ Q = S_within @ Q + a*(S_carry @ Q)
        ggml_tensor * q_LH_cc = ggml_cont(ctx, q_LH_c);
        ggml_tensor * qstate_c = ggml_mul_mat(ctx, state_within_c, q_LH_cc);
        ggml_tensor * alpha_decay = nullptr;
        if (state_carry != nullptr) {
            alpha_decay = ggml_exp(ctx, cum_c);
            ggml_tensor * ad_4d  = ggml_reshape_4d(ctx, alpha_decay, 1, 1, cs_act, H);
            ggml_tensor * q_flat = ggml_reshape_4d(ctx, q_LH_cc, D_qk, R*cs_act, 1, H);
            ggml_tensor * q_carry = ggml_reshape_4d(ctx,
                    ggml_mul_mat(ctx, state_carry, q_flat), D_v, R, cs_act, H);
            qstate_c = ggml_fma(ctx, qstate_c, q_carry, ad_4d);
        }

        ggml_tensor * over_c = ggml_view_4d(ctx, over_LH, D_v, R, cs_act, H,
                over_LH->nb[1], over_LH->nb[2], over_LH->nb[3], t_start*over_LH->nb[2]);
        qstate_c = ggml_sub(ctx, qstate_c, over_c);

        ggml_tensor * o_c        = ggml_fma(ctx, qstate_c, v_LH_c, D4);
        o_c                      = ggml_fused_mul_unary(ctx, ggml_cont(ctx, z_LH_c), o_c, GGML_UNARY_OP_SILU);
        ggml_tensor * o_w_c      = ggml_mul(ctx, o_c, mimo_4d);
        ggml_tensor * o_rsum_in  = ggml_cont(ctx, ggml_permute(ctx, o_w_c, 1, 0, 2, 3));
        ggml_tensor * y_rsum     = ggml_sum_rows(ctx, o_rsum_in);                     // (1, D_v, cs, H)
        ggml_tensor * y_Dv_cs_H  = ggml_reshape_3d(ctx, y_rsum, D_v, cs_act, H);
        ggml_tensor * y_c_DvHcs  = ggml_cont(ctx, ggml_permute(ctx, y_Dv_cs_H, 0, 2, 1, 3));  // (D_v, H, cs)
        y_chunks.push_back(y_c_DvHcs);

        // state at the chunk end, needed by the next chunk and by state_out.
        {
            ggml_tensor * sw_last = ggml_view_4d(ctx, state_within_c, D_qk, D_v, 1, H,
                                                  state_within_c->nb[1],
                                                  state_within_c->nb[2],
                                                  state_within_c->nb[3],
                                                  (cs_act - 1) * state_within_c->nb[2]);
            if (state_carry != nullptr) {
                ggml_tensor * decay_4d = ggml_view_4d(ctx, alpha_decay, 1, 1, 1, H,
                        4, 4, (size_t) cs_act*4, (size_t) (cs_act - 1)*4);
                state_carry = ggml_fma(ctx, sw_last, state_carry, decay_4d);
            } else {
                state_carry = ggml_cont(ctx, sw_last);
            }
        }
    }

    // Tree concat (linear chaining recopies all previous chunks).
    while (y_chunks.size() > 1) {
        std::vector<ggml_tensor *> next;
        next.reserve((y_chunks.size() + 1) / 2);
        size_t i = 0;
        for (; i + 1 < y_chunks.size(); i += 2) {
            next.push_back(ggml_concat(ctx, y_chunks[i], y_chunks[i + 1], /*dim=*/ 2));
        }
        if (i < y_chunks.size()) {
            next.push_back(y_chunks[i]);
        }
        y_chunks = std::move(next);
    }
    ggml_tensor * y = y_chunks[0];

    // Pack state_out_packed if requested. state_carry is state at L-1.
    if (state_out_packed != nullptr) {
        ggml_tensor * k_last = ggml_view_4d(ctx, k_rot, D_qk, R, H, 1,
                                             k_rot->nb[1], k_rot->nb[2], k_rot->nb[3],
                                             (L - 1) * k_rot->nb[3]);
        ggml_tensor * v_last = ggml_view_4d(ctx, v_proj, D_v, R, H, 1,
                                             v_proj->nb[1], v_proj->nb[2], v_proj->nb[3],
                                             (L - 1) * v_proj->nb[3]);
        ggml_tensor * ang_last = ggml_view_3d(ctx, cum_lt, 1, num_angles, H,
                                               cum_lt->nb[1], cum_lt->nb[2],
                                               (L - 1) * cum_lt->nb[0]);
        auto flat = [&](ggml_tensor * t, int64_t n) {
            return ggml_reshape_2d(ctx, ggml_cont(ctx, t), n, 1);
        };
        *state_out_packed = ggml_concat(ctx,
            ggml_concat(ctx,
                ggml_concat(ctx,
                    flat(state_carry, H * D_v * D_qk),
                    flat(k_last,      H * R   * D_qk), /*dim=*/ 0),
                flat(v_last,          H * R   * D_v ), /*dim=*/ 0),
            flat(ang_last,            H * num_angles), /*dim=*/ 0);
    }

    return y;
}

// Mamba3-MIMO decode step (L == 1); longer batches use the chunked path.
ggml_tensor * mamba3_mimo(
        ggml_context * ctx,
        ggml_tensor * q_biased,    // (D_qk, R, H, 1)
        ggml_tensor * k_biased,    // (D_qk, R, H, 1)
        ggml_tensor * v_proj,      // (D_v,  R, H, 1)
        ggml_tensor * z_proj,      // (D_v,  R, H, 1)
        ggml_tensor * log_alpha,   // (H, 1) ADT = log(alpha)
        ggml_tensor * dt,          // (H, 1) post-softplus
        ggml_tensor * trap_post,   // (H, 1) post-sigmoid
        ggml_tensor * ang_raw,     // (num_angles, 1)
        ggml_tensor * D_skip,      // (H,)
        ggml_tensor * mimo_o,      // (D_v, R, H)
        int64_t D_qk, int64_t D_v, int64_t R, int64_t H, int64_t L, int64_t num_angles,
        ggml_tensor * state_in_packed,  // (n_embd_s, 1) or nullptr
        ggml_tensor ** state_out_packed, int il) {
    if (L > 1) {
        const int64_t cs = olala_chunk_size > 0 ? std::min<int64_t>(olala_chunk_size, L) : L;
        return mamba3_mimo_chunked(ctx, q_biased, k_biased, v_proj, z_proj,
                log_alpha, dt, trap_post, ang_raw, D_skip, mimo_o,
                D_qk, D_v, R, H, L, num_angles, cs, il,
                state_in_packed, state_out_packed);
    }
    const int64_t quarter = D_qk/4;
    GGML_ASSERT(num_angles == quarter);

    const int64_t off_K   = H * D_v * D_qk;
    const int64_t off_V   = off_K + H * R * D_qk;
    const int64_t off_ang = off_V + H * R * D_v;

    ggml_tensor * state_in_4d = nullptr;
    ggml_tensor * K_in_4d     = nullptr;
    ggml_tensor * V_in_4d     = nullptr;
    ggml_tensor * angle_in_3d = nullptr;
    if (state_in_packed != nullptr) {
        const size_t fs = ggml_element_size(state_in_packed);
        const int64_t n_embd_s = off_ang + H * num_angles;
        GGML_ASSERT(ggml_is_contiguous(state_in_packed));
        GGML_ASSERT(ggml_nelements(state_in_packed) == n_embd_s);

        auto slice4 = [&](int64_t off, int64_t ne0, int64_t ne1, int64_t ne2) {
            return ggml_view_4d(ctx, state_in_packed, ne0, ne1, ne2, 1,
                                ne0*fs, ne0*ne1*fs, (size_t) n_embd_s*fs, off*fs);
        };
        state_in_4d = slice4(0,       D_qk, D_v, H);
        K_in_4d     = slice4(off_K,   D_qk, R,   H);
        V_in_4d     = slice4(off_V,   D_v,  R,   H);
        angle_in_3d = ggml_view_3d(ctx, state_in_packed, num_angles, H, 1,
                                   num_angles*fs, (size_t) n_embd_s*fs, off_ang*fs);
    }

    constexpr float k_pi = 3.14159265358979323846f;
    ggml_tensor * ang_3d = ggml_reshape_3d(ctx, ggml_scale(ctx, ggml_tanh(ctx, ang_raw), k_pi), num_angles, 1, 1);
    ggml_tensor * contrib = ggml_mul(ctx, ggml_repeat_4d(ctx, ang_3d, num_angles, H, 1, 1),
                                     ggml_reshape_3d(ctx, dt, 1, H, 1));
    ggml_tensor * cum1 = contrib;
    if (angle_in_3d != nullptr) {
        cum1 = ggml_add(ctx, cum1, angle_in_3d);
    }
    ggml_tensor * cos_a = nullptr;
    ggml_tensor * sin_a = nullptr;
    cos_sin(ctx, cum1, &cos_a, &sin_a);
    ggml_tensor * cum_lt = ggml_reshape_3d(ctx, cum1, 1, num_angles, H);

    ggml_tensor * c4  = ggml_reshape_4d(ctx, cos_a, quarter, 1, H, 1);
    ggml_tensor * s4  = ggml_reshape_4d(ctx, sin_a, quarter, 1, H, 1);
    ggml_tensor * ns4 = ggml_scale(ctx, s4, -1.0f);
    auto apply_rotary = [&](ggml_tensor * x) {
        const size_t es = ggml_element_size(x);
        auto group = [&](int gi) {
            return ggml_view_4d(ctx, x, quarter, R, H, 1,
                                x->nb[1], x->nb[2], x->nb[3], gi*quarter*es);
        };
        ggml_tensor * g0 = group(0);
        ggml_tensor * g1 = group(1);
        ggml_tensor * g2 = group(2);
        ggml_tensor * g3 = group(3);
        ggml_tensor * g0n = ggml_fma(ctx, ggml_mul(ctx, g0, c4), g2, ns4);
        ggml_tensor * g2n = ggml_fma(ctx, ggml_mul(ctx, g0, s4), g2, c4);
        return ggml_concat(ctx, ggml_concat(ctx, g0n, g1, 0), ggml_concat(ctx, g2n, g3, 0), 0);
    };
    ggml_tensor * q_rot = apply_rotary(q_biased);
    ggml_tensor * k_rot = apply_rotary(k_biased);

    ggml_tensor * gamma      = ggml_mul(ctx, trap_post, dt);
    ggml_tensor * coeff_prev = ggml_mul(ctx, ggml_scale_bias(ctx, trap_post, -1.0f, 1.0f), dt);

    auto cont_if = [&](ggml_tensor * t) {
        return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
    };

    ggml_tensor * kv = ggml_mul_mat(ctx,
            cont_if(ggml_permute(ctx, k_rot,  1, 0, 2, 3)),
            cont_if(ggml_permute(ctx, v_proj, 1, 0, 2, 3)));
    ggml_tensor * kv_p = cont_if(ggml_permute(ctx, kv, 0, 1, 3, 2)); // (D_qk,D_v,1,H)

    ggml_tensor * state = ggml_mul(ctx, kv_p, ggml_reshape_4d(ctx, gamma, 1, 1, 1, H));

    if (state_in_4d != nullptr) {
        ggml_tensor * st_in = cont_if(ggml_permute(ctx, state_in_4d, 0, 1, 3, 2));
        ggml_tensor * prev_kv = ggml_mul_mat(ctx,
                cont_if(ggml_permute(ctx, K_in_4d, 1, 0, 2, 3)),
                cont_if(ggml_permute(ctx, V_in_4d, 1, 0, 2, 3)));
        ggml_tensor * prev_p = cont_if(ggml_permute(ctx, prev_kv, 0, 1, 3, 2));
        ggml_tensor * eff = ggml_fma(ctx, st_in, prev_p,
                ggml_reshape_4d(ctx, coeff_prev, 1, 1, 1, H));
        ggml_tensor * decay = ggml_reshape_4d(ctx, ggml_exp(ctx, log_alpha), 1, 1, 1, H);
        state = ggml_fma(ctx, state, eff, decay);
    }

    ggml_tensor * q_for_mm = cont_if(ggml_permute(ctx, q_rot, 0, 1, 3, 2)); // (D_qk,R,1,H)
    ggml_tensor * qstate = ggml_mul_mat(ctx, state, q_for_mm);

    ggml_tensor * state_last_flat = nullptr;
    if (state_out_packed != nullptr) {
        state_last_flat = ggml_reshape_2d(ctx, state, H*D_v*D_qk, 1);
    }

    ggml_tensor * o = ggml_fma(ctx, qstate,
            cont_if(ggml_permute(ctx, v_proj, 0, 1, 3, 2)),
            ggml_reshape_4d(ctx, D_skip, 1, 1, 1, H));
    o = ggml_fused_mul_unary(ctx, cont_if(ggml_permute(ctx, z_proj, 0, 1, 3, 2)), o, GGML_UNARY_OP_SILU);

    ggml_tensor * y = ggml_sum_rows(ctx, cont_if(ggml_permute(ctx,
            ggml_mul(ctx, o, ggml_reshape_4d(ctx, mimo_o, D_v, R, 1, H)), 1, 0, 2, 3)));
    y = ggml_reshape_3d(ctx, y, D_v, 1, H);

    if (state_out_packed != nullptr) {
        ggml_tensor * k_last = ggml_view_4d(ctx, k_rot, D_qk, R, H, 1,
                                            k_rot->nb[1], k_rot->nb[2], k_rot->nb[3], 0);
        ggml_tensor * v_last = ggml_view_4d(ctx, v_proj, D_v, R, H, 1,
                                            v_proj->nb[1], v_proj->nb[2], v_proj->nb[3], 0);
        ggml_tensor * a_last = ggml_view_4d(ctx, cum_lt, 1, num_angles, H, 1,
                                            cum_lt->nb[1], cum_lt->nb[2], cum_lt->nb[3], 0);
        auto flat = [&](ggml_tensor * t, int64_t n) {
            return ggml_reshape_2d(ctx, ggml_cont(ctx, t), n, 1);
        };
        *state_out_packed = ggml_concat(ctx,
                ggml_concat(ctx,
                    ggml_concat(ctx, state_last_flat, flat(k_last, H*R*D_qk), 0),
                    flat(v_last, H*R*D_v), 0),
                flat(a_last, H*num_angles), 0);
    }

    return cont_if(ggml_permute(ctx, y, 0, 2, 1, 3)); // (D_v, H, 1)
}

} // namespace

ggml_cgraph * llm_build_context::build_olala() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t H_ssm    = (int64_t) hparams.ssm_d_inner / (int64_t) hparams.ssm_dt_rank;
    const int64_t D_v      = (int64_t) hparams.ssm_dt_rank;
    const int64_t D_qk     = (int64_t) hparams.ssm_d_state;
    const int64_t R        = (int64_t) hparams.olala_mamba_mimo_dim;
    const int64_t G        = (int64_t) hparams.ssm_n_group;
    const int64_t angles_d = (int64_t) hparams.olala_num_rope_angles;
    const int64_t head_dim = (int64_t) hparams.n_embd_head_k_full;
    const int64_t n_head   = (int64_t) hparams.olala_n_signal_heads + (int64_t) hparams.olala_n_noise_heads;
    const int64_t n_kv_heads = (int64_t) hparams.olala_n_noise_heads;
    const int64_t n_signal = (int64_t) hparams.olala_n_signal_heads;
    const int64_t snr      = n_signal / n_kv_heads;
    const int64_t rank     = (int64_t) hparams.olala_tpa_rank;
    const int64_t n_embd_s = (int64_t) hparams.n_embd_v_s();
    const int64_t shift_sz = (int64_t) hparams.olala_shift_state_size();

    ggml_tensor * cur = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    cur = ggml_l2_norm(ctx0, cur, 1e-12f);
    cb(cur, "inp_norm", -1);
    ggml_build_forward_expand(gf, cur);

    ggml_tensor * inp_pos    = build_inp_pos();
    // I32->F32 cast is CPU-unsupported; gather from an F32 arange table.
    ggml_tensor * pos_table = ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0.0f, (float) n_ctx, 1.0f), 1, n_ctx);
    ggml_tensor * pos_f32   = ggml_get_rows(ctx0, pos_table, inp_pos); // (1, n_tokens)

    // ggml_clamp writes through a view: clamp a duplicate.
    const float slw = (float) std::max<uint32_t>(hparams.olala_slw_wsize, 1u);
    ggml_tensor * pos_mask = ggml_reshape_3d(ctx0,
            ggml_clamp(ctx0, ggml_dup(ctx0, pos_f32), 0.0f, 1.0f), 1, 1, n_tokens);
    ggml_tensor * pos1 = ggml_scale_bias(ctx0, pos_f32, 1.0f, 1.0f);
    ggml_tensor * log_pos = ggml_reshape_3d(ctx0, ggml_log(ctx0, ggml_clamp(ctx0, pos1, 1.0f, slw)), 1, 1, n_tokens);

    ggml_tensor * KQ_mask    = build_inp_KQ_mask();
    ggml_tensor * state_mask = build_inp_s_mask();
    ggml_tensor * inp_out_ids = (!llm_arch_requires_all_graph_output_rows(model.arch) &&
            n_tokens > 1 && n_outputs < n_tokens) ? build_inp_out_ids() : nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        const bool is_mamba = hparams.is_recurrent(il);

        ggml_tensor * residual = cur;
        ggml_tensor * mixer = nullptr;

        ggml_tensor * state_all = kv_self.s_l[il];
        GGML_ASSERT(state_all != nullptr);
        const int64_t state_width = is_mamba ? n_embd_s : shift_sz;
        GGML_ASSERT((int64_t) state_all->ne[0] == state_width);

        // One recurrent-state slot per context (n_seq_max == 1).
        ggml_tensor * state_view = ggml_view_2d(ctx0, state_all, state_width, 1, state_all->nb[1], 0);
        ggml_tensor * state_mask1 = ggml_view_1d(ctx0, state_mask, 1, 0);
        ggml_tensor * state_in = ggml_mul(ctx0, state_view, state_mask1);

        if (is_mamba) {
            ggml_tensor * in = cur->type == GGML_TYPE_F32 ? cur : ggml_cast(ctx0, cur, GGML_TYPE_F32);
            if (!ggml_is_contiguous(in)) in = ggml_cont(ctx0, in);

            ggml_tensor * proj_static = llm_build_lora_mm(lctx, ctx0, layer.ssm_in, in);
            ggml_tensor * proj_dyn    = llm_build_lora_mm(lctx, ctx0, layer.ssm_in_dyn, in);
            cb(proj_static, "m_in_proj", il);
            cb(proj_dyn, "m_in_proj_dyn", il);
            const int64_t per_head = 2*D_v + 3;
            ggml_tensor * per = ggml_reshape_3d(ctx0, proj_static, per_head, H_ssm, n_tokens);
            const int64_t fs = ggml_element_size(per);
            // 4D view with singleton rank dim: repeat reads it strided.
            auto mimo_in = [&](int64_t off) {
                return ggml_view_4d(ctx0, per, D_v, 1, H_ssm, n_tokens,
                                    D_v*fs, per->nb[1], per->nb[2], off*fs);
            };
            ggml_tensor * z_in = mimo_in(0);
            ggml_tensor * x_in = mimo_in(D_v);

            auto pick = [&](int64_t off) {
                ggml_tensor * v3 = ggml_view_3d(ctx0, per, 1, H_ssm, n_tokens, per->nb[1], per->nb[2], off*per->nb[0]);
                return ggml_reshape_2d(ctx0, ggml_cont(ctx0, v3), H_ssm, n_tokens);
            };
            ggml_tensor * dt_raw   = pick(2*D_v + 0);
            ggml_tensor * A_raw    = pick(2*D_v + 1);
            ggml_tensor * trap_raw = pick(2*D_v + 2);

            const int64_t BC_size = R*G*D_qk;
            ggml_tensor * B = nullptr;
            ggml_tensor * C = nullptr;
            ggml_tensor * ang_raw = nullptr;
            if (n_tokens == 1) {
                const size_t fs = ggml_element_size(proj_dyn);
                B = ggml_view_3d(ctx0, proj_dyn, D_qk, R, 1, D_qk*fs, D_qk*R*fs, 0);
                C = ggml_view_3d(ctx0, proj_dyn, D_qk, R, 1, D_qk*fs, D_qk*R*fs, BC_size*fs);
                ang_raw = ggml_view_1d(ctx0, proj_dyn, angles_d, 2*BC_size*fs);
            } else {
                ggml_tensor * B_flat = ggml_view_2d(ctx0, proj_dyn, BC_size, n_tokens, proj_dyn->nb[1], 0);
                ggml_tensor * C_flat = ggml_view_2d(ctx0, proj_dyn, BC_size, n_tokens, proj_dyn->nb[1], BC_size*proj_dyn->nb[0]);
                ang_raw = ggml_view_2d(ctx0, proj_dyn, angles_d, n_tokens, proj_dyn->nb[1], 2*BC_size*proj_dyn->nb[0]);
                B = ggml_reshape_3d(ctx0, ggml_cont(ctx0, B_flat), D_qk, R, n_tokens);
                C = ggml_reshape_3d(ctx0, ggml_cont(ctx0, C_flat), D_qk, R, n_tokens);
            }
            B = llm_build_norm(ctx0, B, hparams, layer.ssm_b_norm, nullptr, LLM_NORM_RMS, cb, il);
            C = llm_build_norm(ctx0, C, hparams, layer.ssm_c_norm, nullptr, LLM_NORM_RMS, cb, il);

            ggml_tensor * c_bias = layer.ssm_c_bias->type == GGML_TYPE_F32 ? layer.ssm_c_bias : ggml_cast(ctx0, layer.ssm_c_bias, GGML_TYPE_F32);
            ggml_tensor * b_bias = layer.ssm_b_bias->type == GGML_TYPE_F32 ? layer.ssm_b_bias : ggml_cast(ctx0, layer.ssm_b_bias, GGML_TYPE_F32);
            ggml_tensor * q_biased = nullptr;
            ggml_tensor * k_biased = nullptr;
            ggml_tensor * v_proj   = nullptr;
            ggml_tensor * z_proj   = nullptr;
            if (n_tokens == 1) {
                q_biased = ggml_add(ctx0, ggml_reshape_4d(ctx0, c_bias, D_qk, R, H_ssm, 1),
                                          ggml_reshape_4d(ctx0, C, D_qk, R, 1, 1));
                k_biased = ggml_add(ctx0, ggml_reshape_4d(ctx0, b_bias, D_qk, R, H_ssm, 1),
                                          ggml_reshape_4d(ctx0, B, D_qk, R, 1, 1));
                v_proj   = ggml_mul(ctx0, ggml_reshape_4d(ctx0, layer.ssm_mimo_x, D_v, R, H_ssm, 1), x_in);
                z_proj   = ggml_mul(ctx0, ggml_reshape_4d(ctx0, layer.ssm_mimo_z, D_v, R, H_ssm, 1), z_in);
            } else {
                q_biased = ggml_add(ctx0, ggml_repeat_4d(ctx0,
                        ggml_reshape_4d(ctx0, C, D_qk, R, 1, n_tokens), D_qk, R, H_ssm, n_tokens), c_bias);
                k_biased = ggml_add(ctx0, ggml_repeat_4d(ctx0,
                        ggml_reshape_4d(ctx0, B, D_qk, R, 1, n_tokens), D_qk, R, H_ssm, n_tokens), b_bias);
                v_proj   = ggml_mul(ctx0,
                        ggml_repeat_4d(ctx0, x_in, D_v, R, H_ssm, n_tokens), layer.ssm_mimo_x);
                z_proj   = ggml_mul(ctx0,
                        ggml_repeat_4d(ctx0, z_in, D_v, R, H_ssm, n_tokens), layer.ssm_mimo_z);
            }

            ggml_tensor * dt_sp = ggml_softplus(ctx0, ggml_add(ctx0, dt_raw, layer.ssm_dt_bias));
            ggml_tensor * a_neg = ggml_clamp(ctx0, ggml_scale(ctx0, ggml_softplus(ctx0, A_raw), -1.0f), -FLT_MAX, -1e-4f);
            // Feed ADT directly: exp() could underflow and log(0) give -inf.
            ggml_tensor * log_alpha = ggml_mul(ctx0, a_neg, dt_sp);
            ggml_tensor * trap  = ggml_sigmoid(ctx0, trap_raw);

            auto f32c = [&](ggml_tensor * t) {
                if (t->type != GGML_TYPE_F32) t = ggml_cast(ctx0, t, GGML_TYPE_F32);
                return ggml_is_contiguous(t) ? t : ggml_cont(ctx0, t);
            };

            ggml_tensor * q_f = f32c(q_biased);
            ggml_tensor * k_f = f32c(k_biased);
            ggml_tensor * v_f = f32c(v_proj);
            ggml_tensor * z_f = f32c(z_proj);
            ggml_tensor * al_f = f32c(log_alpha);
            ggml_tensor * dt_f = f32c(dt_sp);
            ggml_tensor * tr_f = f32c(trap);
            ggml_tensor * an_f = f32c(ang_raw);

            ggml_tensor * state_out = nullptr;
            ggml_tensor * y = mamba3_mimo(ctx0,
                    q_f, k_f, v_f, z_f, al_f, dt_f, tr_f, an_f,
                    f32c(layer.ssm_d), f32c(layer.ssm_mimo_o),
                    D_qk, D_v, R, H_ssm, n_tokens, angles_d,
                    state_in, &state_out, il);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, state_out, state_view));
            mixer = llm_build_lora_mm(lctx, ctx0, layer.wo, ggml_reshape_2d(ctx0, y, D_v*H_ssm, n_tokens));
            cb(mixer, "m_mixer_out", il);
        } else {
            ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, layer.wq, cur);
            q = ggml_reshape_3d(ctx0, q, head_dim, n_head, n_tokens);

            auto build_tpa = [&](ggml_tensor * wa, ggml_tensor * wb) {
                ggml_tensor * a = ggml_reshape_3d(ctx0, llm_build_lora_mm(lctx, ctx0, wa, cur), rank, n_kv_heads, n_tokens);
                ggml_tensor * b = ggml_reshape_3d(ctx0, llm_build_lora_mm(lctx, ctx0, wb, cur), head_dim, rank, n_tokens);
                b = ggml_cont(ctx0, ggml_permute(ctx0, b, 1, 0, 2, 3));
                return ggml_mul_mat(ctx0, b, a);
            };
            // k is followed by RMSNorm, so the 1/rank factor cancels
            ggml_tensor * k = build_tpa(layer.attn_wa_k, layer.attn_wb_k);
            ggml_tensor * v = ggml_scale(ctx0, build_tpa(layer.attn_wa_v, layer.attn_wb_v), 1.0f/(float) rank);

            const int64_t n_kp = head_dim*n_kv_heads;
            const size_t fs = sizeof(float);
            ggml_tensor * k_prev_in = ggml_reshape_4d(ctx0,
                    ggml_view_1d(ctx0, state_in, n_kp, 0), head_dim, n_kv_heads, 1, 1);
            ggml_tensor * v_prev_in = ggml_reshape_4d(ctx0,
                    ggml_view_1d(ctx0, state_in, n_kp, n_kp*fs), head_dim, n_kv_heads, 1, 1);

            auto shift = [&](ggml_tensor * x, ggml_tensor * proj, ggml_tensor * prev_first) {
                ggml_tensor * x4 = ggml_reshape_4d(ctx0, x, head_dim, n_kv_heads, n_tokens, 1);
                ggml_tensor * prev;
                if (n_tokens == 1) {
                    prev = prev_first;
                } else {
                    ggml_tensor * head = ggml_view_4d(ctx0, x4, head_dim, n_kv_heads, n_tokens - 1, 1,
                                                      x4->nb[1], x4->nb[2], x4->nb[3], 0);
                    prev = ggml_concat(ctx0, prev_first, head, 2);
                }
                if (!ggml_is_contiguous(prev)) prev = ggml_cont(ctx0, prev);
                ggml_tensor * prev_flat = ggml_reshape_3d(ctx0, prev, head_dim, n_kv_heads, n_tokens);
                ggml_tensor * a = ggml_sigmoid(ctx0, llm_build_lora_mm(lctx, ctx0, proj, cur));
                a = ggml_mul(ctx0, ggml_reshape_3d(ctx0, a, 1, n_kv_heads, n_tokens), pos_mask);
                return ggml_fma(ctx0, x, ggml_sub(ctx0, prev_flat, x), a);
            };
            ggml_tensor * k_new = shift(k, layer.attn_shift_k, k_prev_in);
            ggml_tensor * v_new = shift(v, layer.attn_shift_v, v_prev_in);

            ggml_tensor * k4 = ggml_reshape_4d(ctx0, k, head_dim, n_kv_heads, n_tokens, 1);
            ggml_tensor * v4 = ggml_reshape_4d(ctx0, v, head_dim, n_kv_heads, n_tokens, 1);
            ggml_tensor * k_last = ggml_view_4d(ctx0, k4, head_dim, n_kv_heads, 1, 1,
                                                k4->nb[1], k4->nb[2], k4->nb[3], (n_tokens - 1)*k4->nb[2]);
            ggml_tensor * v_last = ggml_view_4d(ctx0, v4, head_dim, n_kv_heads, 1, 1,
                                                v4->nb[1], v4->nb[2], v4->nb[3], (n_tokens - 1)*v4->nb[2]);
            if (!ggml_is_contiguous(k_last)) k_last = ggml_cont(ctx0, k_last);
            if (!ggml_is_contiguous(v_last)) v_last = ggml_cont(ctx0, v_last);
            ggml_tensor * packed = ggml_concat(ctx0,
                    ggml_reshape_2d(ctx0, k_last, n_kp, 1),
                    ggml_reshape_2d(ctx0, v_last, n_kp, 1), 0);

            q = llm_build_norm(ctx0, q, hparams, layer.attn_q_norm, nullptr, LLM_NORM_RMS, cb, il);
            k_new = llm_build_norm(ctx0, k_new, hparams, layer.attn_k_norm, nullptr, LLM_NORM_RMS, cb, il);

            ggml_tensor * scaler = ggml_reshape_3d(ctx0, layer.attn_softmax_scaler, 1, n_head, 1);
            ggml_tensor * q_scale = ggml_mul(ctx0,
                    ggml_repeat_4d(ctx0, scaler, 1, n_head, n_tokens, 1), log_pos);
            q = ggml_mul(ctx0, q, q_scale);

            const float kq_scale = 1.0f/(float) head_dim;
            ggml_tensor * attn = llm_build_kv(ctx0, lctx, kv_self, gf,
                    nullptr, nullptr, k_new, v_new, q, KQ_mask,
                    (int32_t) n_tokens, kv_head, (int32_t) n_kv, kq_scale,
                    cb, il, nullptr, 0, -1, nullptr, nullptr, swa_head, nullptr);
            // store the next token's shift carry after the state is consumed.
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, packed, state_view));
            ggml_tensor * attn3 = ggml_reshape_3d(ctx0, attn, head_dim, n_head, n_tokens);
            attn = ggml_reshape_4d(ctx0, attn3, head_dim, snr + 1, n_kv_heads, n_tokens);

            ggml_tensor * sig = ggml_view_4d(ctx0, attn, head_dim, snr, n_kv_heads, n_tokens,
                                             attn->nb[1], attn->nb[2], attn->nb[3], 0);
            ggml_tensor * noi = ggml_view_4d(ctx0, attn, head_dim, 1, n_kv_heads, n_tokens,
                                             attn->nb[1], attn->nb[2], attn->nb[3], (size_t) snr*attn->nb[1]);
            ggml_tensor * lam = ggml_sigmoid(ctx0, llm_build_lora_mm(lctx, ctx0, layer.attn_lambda, cur));
            lam = ggml_reshape_4d(ctx0, lam, 1, 1, n_kv_heads, n_tokens);
            // `noi` is strided: fused_mul_unary needs contiguous src1 on CUDA.
            ggml_tensor * combined = ggml_sub(ctx0, sig, ggml_mul(ctx0, noi, lam));
            ggml_tensor * y = ggml_reshape_2d(ctx0, combined, head_dim*n_signal, n_tokens);

            ggml_tensor * gate_in = ggml_scale_bias(ctx0,
                    llm_build_lora_mm(lctx, ctx0, layer.attn_gate, cur), 1.0f, hparams.olala_gate_bias);
            y = ggml_fused_mul_unary(ctx0, gate_in, y, GGML_UNARY_OP_SILU);
            mixer = llm_build_lora_mm(lctx, ctx0, layer.wo, y);
            cb(mixer, "v_mixer_out", il);
        }

        cur = geodesic(ctx0, residual, mixer, layer.geo_mixer_scale, layer.geo_mixer_bias, il);
        cb(cur, "post_mixer", il);

        // Last layer: only output rows need the MoE and second geodesic.
        if (il == n_layer - 1 && inp_out_ids != nullptr) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
        }

        residual = cur;

        ggml_tensor * y_mlp = llm_build_latent_moe(ctx0, lctx, cur, layer,
                n_expert, n_expert_used,
                LLM_FFN_RELU_SQR, /*norm_w=*/true, /*scale_w=*/true,
                hparams.expert_weights_scale,
                LLM_EXPERT_GATING_FUNC_SIGMOID,
                cb, il, gf);

        cur = geodesic(ctx0, residual, y_mlp, layer.geo_mlp_scale, layer.geo_mlp_bias, il);
        cb(cur, "post_mlp", il);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);
    }

    cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
