#include "iqk_config.h"
#include "iqk_cpu_ops.h"

#if defined IQK_IMPLEMENT

#include "iqk_common.h"

#include <cmath>
#include <cstring>

namespace {

struct kda_layout {
    size_t v[3];
    size_t g[4];
    size_t beta[4];
};

#ifdef __ARM_NEON
template <int head_dim>
void iqk_kda_neon_impl(int32_t n_heads, int32_t gqa_ratio, int32_t repeat_type, int32_t n_tokens, int32_t n_seqs,
        const kda_layout & layout,
        const float * q_data, const float * k_data, const float * v_data, const float * g_data, const float * beta_data,
        const float * state_in, float * out_data, float * state_out, float * saved_steps,
        int32_t state_step_stride, int32_t ith, int32_t nth) {
    const int32_t total_heads = n_heads * n_seqs;
    const int32_t heads_per_thread = (total_heads + nth - 1) / nth;
    const int32_t h_start = ith * heads_per_thread;
    const int32_t h_end = (h_start + heads_per_thread < total_heads) ? h_start + heads_per_thread : total_heads;

    static_assert(head_dim % 4 == 0);

    const float scale = 1.0f / sqrtf((float) head_dim);

    float v_new_buf[head_dim];
    float v_prime[head_dim], out_val[head_dim];
    float decay_buf[head_dim], kd[head_dim], qd[head_dim];

    float32x4x4_t vs4[4];

    for (int32_t h_idx = h_start; h_idx < h_end; ++h_idx) {
        const int32_t batch_idx = h_idx / n_heads;
        const int32_t head_idx  = h_idx % n_heads;
        const int32_t head_idx_kq = repeat_type == 0 ? head_idx / gqa_ratio : head_idx % (n_heads/gqa_ratio);

        const int32_t state_head_offset = batch_idx * (head_dim * head_dim * n_heads) + head_idx * (head_dim * head_dim);
        const int32_t out_head_offset = batch_idx * (head_dim * n_heads * n_tokens) + head_idx * head_dim;
        const int32_t out_token_stride = head_dim * n_heads;

        const int32_t qkv_head_offset_kq = batch_idx * (head_dim * n_tokens * n_heads/gqa_ratio) + head_idx_kq * (head_dim * n_tokens);
        const int32_t qkv_token_stride = head_dim;
        const float * v_head = v_data + batch_idx * layout.v[2] + head_idx * layout.v[1];
        const float * g_head = g_data + batch_idx * layout.g[3] + head_idx * layout.g[2];
        const float * beta_head = beta_data + batch_idx * layout.beta[3] + head_idx * layout.beta[2];

        float * state = state_out + state_head_offset;
        for (int32_t i = 0; i < head_dim * head_dim; ++i) {
            state[i] = state_in[state_head_offset + i];
        }

        for (int32_t t = 0; t < n_tokens; ++t) {
            const float * q_t = q_data + qkv_head_offset_kq + t * qkv_token_stride;
            const float * k_t = k_data + qkv_head_offset_kq + t * qkv_token_stride;
            const float * v_t = v_head + t * layout.v[0];
            const float * g_t = g_head + t * layout.g[0];

            const float beta_raw = beta_head[t * layout.beta[1]];

            float kq_sum = 0.0f;
            auto vqksum = vdupq_n_f32(0.0f);
            for (int32_t i = 0; i < head_dim; i += 4) {
                auto vq = vld1q_f32(q_t + i);
                auto vk = vld1q_f32(k_t + i);
                vqksum = vfmaq_f32(vqksum, vq, vk);
            }
            kq_sum = vaddvq_f32(vqksum);

            const float beta_val = 1.0f / (1.0f + expf(-beta_raw));
            for (int32_t col = 0; col < head_dim; ++col) {
                decay_buf[col] = expf(fminf(g_t[col * layout.g[1]], 50.0f));
                kd[col] = k_t[col] * decay_buf[col];
                qd[col] = q_t[col] * decay_buf[col];
            }

            const float attn_score = kq_sum * scale;

            float * out_t = out_data + out_head_offset + t * out_token_stride;

            std::memset(v_prime, 0, head_dim*sizeof(float));
            std::memset(out_val, 0, head_dim*sizeof(float));
            for (int32_t col = 0; col < head_dim; ++col) {
                const float k_col = kd[col];
                const float q_col = qd[col];
                for (int32_t row = 0; row < head_dim; ++row) {
                    const float s = state[row + col * head_dim];
                    v_prime[row] += s * k_col;
                    out_val[row] += s * q_col;
                }
            }
            for (int32_t row = 0; row < head_dim; ++row) {
                const float v_new = v_t[row] * beta_val - v_prime[row] * beta_val;
                v_new_buf[row] = v_new;
                out_t[row] = out_val[row] * scale + v_new * attn_score;
            }

            auto vmin = vdupq_n_f32(-1e6f);
            auto vmax = vdupq_n_f32( 1e6f);
            for (int32_t col = 0; col < head_dim; col += 4) {
                auto vk = vld1q_f32(k_t + col);
                for (int32_t row = 0; row < head_dim; row += 16) {
                    for (int32_t k = 0; k < 4; ++k) {
                        vs4[k] = vld1q_f32_x4(state + (col + k)*head_dim + row);
                        auto vd = vdupq_n_f32(decay_buf[col + k]);
                        for (int32_t j = 0; j < 4; ++j) {
                            vs4[k].val[j] = vmulq_f32(vs4[k].val[j], vd);
                        }
                    }
                    auto vn = vld1q_f32_x4(v_new_buf + row);
                    for (int32_t j = 0; j < 4; ++j) {
                        vs4[0].val[j] = vfmaq_laneq_f32(vs4[0].val[j], vn.val[j], vk, 0);
                        vs4[1].val[j] = vfmaq_laneq_f32(vs4[1].val[j], vn.val[j], vk, 1);
                        vs4[2].val[j] = vfmaq_laneq_f32(vs4[2].val[j], vn.val[j], vk, 2);
                        vs4[3].val[j] = vfmaq_laneq_f32(vs4[3].val[j], vn.val[j], vk, 3);
                    }
                    for (int32_t k = 0; k < 4; ++k) {
                        for (int32_t j = 0; j < 4; ++j) {
                            vs4[k].val[j] = vmaxq_f32(vminq_f32(vs4[k].val[j], vmax), vmin);
                        }
                        vst1q_f32_x4(state + (col + k)*head_dim + row, vs4[k]);
                    }
                }
            }

            if (saved_steps && t + 1 < n_tokens) {
                float * this_state = saved_steps + state_head_offset + t * state_step_stride;
                std::memcpy(this_state, state, head_dim * head_dim * sizeof(float));
            }
        }
    }
}
#endif

template <int head_dim>
void iqk_kda_impl(int32_t n_heads, int32_t gqa_ratio, int32_t repeat_type, int32_t n_tokens, int32_t n_seqs,
        const kda_layout & layout,
        const float * q_data, const float * k_data, const float * v_data, const float * g_data, const float * beta_data,
        const float * state_in, float * out_data, float * state_out, float * saved_steps,
        int32_t state_step_stride, int32_t ith, int32_t nth) {
#ifdef __ARM_NEON
    iqk_kda_neon_impl<head_dim>(n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, layout,
            q_data, k_data, v_data, g_data, beta_data, state_in, out_data, state_out,
            saved_steps, state_step_stride, ith, nth);
    return;
#endif
    const int32_t total_heads = n_heads * n_seqs;
    const int32_t heads_per_thread = (total_heads + nth - 1) / nth;
    const int32_t h_start = ith * heads_per_thread;
    const int32_t h_end = (h_start + heads_per_thread < total_heads) ? h_start + heads_per_thread : total_heads;

#ifdef __AVX2__
    static_assert(head_dim % 8 == 0);
#endif

    const float scale = 1.0f / sqrtf((float) head_dim);

#ifdef __AVX512F__
    __m512 v_prime[head_dim/16], out_val[head_dim/16];
#else
    float v_new_buf[head_dim];
    float v_prime[head_dim], out_val[head_dim];
#endif
    float decay_buf[head_dim];

    for (int32_t h_idx = h_start; h_idx < h_end; ++h_idx) {
        const int32_t batch_idx = h_idx / n_heads;
        const int32_t head_idx = h_idx % n_heads;
        const int32_t head_idx_kq = repeat_type == 0 ? head_idx / gqa_ratio : head_idx % (n_heads/gqa_ratio);

        const int32_t state_head_offset = batch_idx * (head_dim * head_dim * n_heads) + head_idx * (head_dim * head_dim);
        const int32_t out_head_offset = batch_idx * (head_dim * n_heads * n_tokens) + head_idx * head_dim;
        const int32_t out_token_stride = head_dim * n_heads;

        const int32_t qkv_head_offset_kq = batch_idx * (head_dim * n_tokens * n_heads/gqa_ratio) + head_idx_kq * (head_dim * n_tokens);
        const int32_t qkv_token_stride = head_dim;
        const float * v_head = v_data + batch_idx * layout.v[2] + head_idx * layout.v[1];
        const float * g_head = g_data + batch_idx * layout.g[3] + head_idx * layout.g[2];
        const float * beta_head = beta_data + batch_idx * layout.beta[3] + head_idx * layout.beta[2];

        float * state = state_out + state_head_offset;
        for (int32_t i = 0; i < head_dim * head_dim; ++i) {
            state[i] = state_in[state_head_offset + i];
        }

        for (int32_t t = 0; t < n_tokens; ++t) {
            const float * q_t = q_data + qkv_head_offset_kq + t * qkv_token_stride;
            const float * k_t = k_data + qkv_head_offset_kq + t * qkv_token_stride;
            const float * v_t = v_head + t * layout.v[0];
            const float * g_t = g_head + t * layout.g[0];

            const float beta_raw = beta_head[t * layout.beta[1]];

            float kq_sum = 0.0f;
#if defined __AVX512F__
            auto vqksum = _mm512_setzero_ps();
            for (int32_t i = 0; i < head_dim; i += 16) {
                auto vq = _mm512_loadu_ps(q_t + i);
                auto vk = _mm512_loadu_ps(k_t + i);
                vqksum = _mm512_fmadd_ps(vk, vq, vqksum);
            }
            kq_sum = _mm512_reduce_add_ps(vqksum);
#elif defined __AVX2__
            auto vqksum = _mm256_setzero_ps();
            for (int32_t i = 0; i < head_dim; i += 8) {
                auto vq = _mm256_loadu_ps(q_t + i);
                auto vk = _mm256_loadu_ps(k_t + i);
                vqksum = _mm256_fmadd_ps(vk, vq, vqksum);
            }
            kq_sum = hsum_float_8(vqksum);
#else
            for (int32_t i = 0; i < head_dim; ++i) {
                kq_sum += k_t[i] * q_t[i];
            }
#endif

            const float beta_val = 1.0f / (1.0f + expf(-beta_raw));
            for (int32_t col = 0; col < head_dim; ++col) {
                decay_buf[col] = expf(fminf(g_t[col * layout.g[1]], 50.0f));
            }

            const float attn_score = kq_sum * scale;

            float * out_t = out_data + out_head_offset + t * out_token_stride;

#ifdef __AVX512F__
            for (int32_t j = 0; j < head_dim/16; ++j) {
                v_prime[j] = out_val[j] = _mm512_setzero_ps();
            }
            for (int32_t col = 0; col < head_dim; ++col) {
                auto k_col = _mm512_set1_ps(k_t[col] * decay_buf[col]);
                auto q_col = _mm512_set1_ps(q_t[col] * decay_buf[col]);
                for (int32_t j = 0; j < head_dim/16; ++j) {
                    auto s = _mm512_loadu_ps(state + col * head_dim + 16*j);
                    v_prime[j] = _mm512_fmadd_ps(s, k_col, v_prime[j]);
                    out_val[j] = _mm512_fmadd_ps(s, q_col, out_val[j]);
                }
            }
            auto c1 = _mm512_set1_ps(beta_val);
            auto c2 = _mm512_set1_ps(beta_val);
            auto c3 = _mm512_set1_ps(scale);
            auto c4 = _mm512_set1_ps(attn_score);
            for (int32_t j = 0; j < head_dim/16; ++j) {
                auto v = _mm512_loadu_ps(v_t + 16*j);
                v_prime[j] = _mm512_sub_ps(_mm512_mul_ps(v, c1), _mm512_mul_ps(v_prime[j], c2));
                auto oval = _mm512_fmadd_ps(v_prime[j], c4, _mm512_mul_ps(out_val[j], c3));
                _mm512_storeu_ps(out_t + 16*j, oval);
            }
            auto vmin = _mm512_set1_ps(-1e6f);
            auto vmax = _mm512_set1_ps( 1e6f);
            for (int32_t col = 0; col < head_dim; ++col) {
                auto vk = _mm512_set1_ps(k_t[col]);
                auto vd = _mm512_set1_ps(decay_buf[col]);
                for (int32_t j = 0; j < head_dim/16; ++j) {
                    auto vs = _mm512_loadu_ps(state + col * head_dim + 16*j);
                    vs = _mm512_fmadd_ps(v_prime[j], vk, _mm512_mul_ps(vs, vd));
                    vs = _mm512_max_ps(vmin, _mm512_min_ps(vmax, vs));
                    _mm512_storeu_ps(state + col * head_dim + 16*j, vs);
                }
            }
#else
            std::memset(v_prime, 0, head_dim*sizeof(float));
            std::memset(out_val, 0, head_dim*sizeof(float));
            for (int32_t col = 0; col < head_dim; ++col) {
                const float k_col = k_t[col] * decay_buf[col];
                const float q_col = q_t[col] * decay_buf[col];
                for (int32_t row = 0; row < head_dim; ++row) {
                    const float s = state[row + col * head_dim];
                    v_prime[row] += s * k_col;
                    out_val[row] += s * q_col;
                }
            }
            for (int32_t row = 0; row < head_dim; ++row) {
                const float v_new = v_t[row] * beta_val - v_prime[row] * beta_val;
                v_new_buf[row] = v_new;
                out_t[row] = out_val[row] * scale + v_new * attn_score;
            }

#ifdef __AVX2__
            auto vmin = _mm256_set1_ps(-1e6f);
            auto vmax = _mm256_set1_ps( 1e6f);
            for (int32_t col = 0; col < head_dim; ++col) {
                auto vk = _mm256_set1_ps(k_t[col]);
                auto vd = _mm256_set1_ps(decay_buf[col]);
                for (int32_t row = 0; row < head_dim; row += 8) {
                    auto vs = _mm256_loadu_ps(state + col * head_dim + row);
                    auto vn = _mm256_loadu_ps(v_new_buf + row);
                    vs = _mm256_fmadd_ps(vn, vk, _mm256_mul_ps(vs, vd));
                    auto mask_l = _mm256_cmp_ps(vs, vmin, _CMP_LT_OQ);
                    auto mask_u = _mm256_cmp_ps(vs, vmax, _CMP_GT_OQ);
                    vs = _mm256_or_ps(_mm256_and_ps(mask_l, vmin), _mm256_andnot_ps(mask_l, vs));
                    vs = _mm256_or_ps(_mm256_and_ps(mask_u, vmax), _mm256_andnot_ps(mask_u, vs));
                    _mm256_storeu_ps(state + col * head_dim + row, vs);
                }
            }
#else
            for (int32_t col = 0; col < head_dim; ++col) {
                const float k_col = k_t[col];
                for (int32_t row = 0; row < head_dim; ++row) {
                    float s = state[row + col * head_dim];
                    s = decay_buf[col] * s + v_new_buf[row] * k_col;
                    state[row + col * head_dim] = fminf(fmaxf(s, -1e6f), 1e6f);
                }
            }
#endif
#endif

            if (saved_steps && t + 1 < n_tokens) {
                float * this_state = saved_steps + state_head_offset + t * state_step_stride;
                std::memcpy(this_state, state, head_dim * head_dim * sizeof(float));
            }
        }
    }
}

}

extern "C" IQK_API bool iqk_kda(int32_t head_dim, int32_t n_heads, int32_t gqa_ratio, int32_t repeat_type,
        int32_t n_tokens, int32_t n_seqs, size_t vnb1, size_t vnb2, size_t vnb3,
        const size_t gnb[4], const size_t bnb[4],
        const float * q_data, const float * k_data, const float * v_data, const float * g_data, const float * beta_data,
        const float * state_in, float * out_data, float * state_out, float * saved_steps,
        int32_t state_step_stride, int32_t ith, int32_t nth) {
    if (head_dim != 64 && head_dim != 128) {
        return false;
    }

    const kda_layout layout = {
        { vnb1/sizeof(float), vnb2/sizeof(float), vnb3/sizeof(float) },
        { gnb[0]/sizeof(float), gnb[1]/sizeof(float), gnb[2]/sizeof(float), gnb[3]/sizeof(float) },
        { bnb[0]/sizeof(float), bnb[1]/sizeof(float), bnb[2]/sizeof(float), bnb[3]/sizeof(float) },
    };

    if (head_dim == 64) {
        iqk_kda_impl<64>(n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, layout,
                q_data, k_data, v_data, g_data, beta_data, state_in,
                out_data, state_out, saved_steps, state_step_stride, ith, nth);
    } else {
        iqk_kda_impl<128>(n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, layout,
                q_data, k_data, v_data, g_data, beta_data, state_in,
                out_data, state_out, saved_steps, state_step_stride, ith, nth);
    }
    return true;
}

#else

extern "C" IQK_API bool iqk_kda(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
        size_t, size_t, size_t, const size_t[4], const size_t[4],
        const float *, const float *, const float *, const float *, const float *,
        const float *, float *, float *, float *, int32_t, int32_t, int32_t) {
    return false;
}

#endif
