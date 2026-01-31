#include "iqk/iqk_config.h"

#if defined IQK_IMPLEMENT && defined GGML_IQK_FLASH_ATTENTION

#include "iqk/fa/iqk_fa_templates.h"

namespace {

template <int step_k, typename KHelper, typename VHelper>
inline void iqk_deepseek_helper(KHelper& kh, VHelper& vh,
                        int nq1, int nk1, int stride_q, int stride_m, int stride_qkv,
                        const float * q, const char * mask, float scale, float softcap, float * qkv,
                        const float * sinkf, float * M, float * S) {
    auto update = [&nq1, &mask, &q, &qkv, &M, &S, stride_q, stride_m, stride_qkv] (int n) {
        nq1 -= n;
        if (nq1 == 0) return true;
        q    += n*stride_q;
        mask += n*stride_m;
        qkv  += n*stride_qkv;
        if (M && S) { M += n; S += n; }
        return false;
    };
    if (nq1 >= 16) {
        int n_step = nq1/16;
        FlashAttn<576, 512, 16, step_k> fa(scale, softcap, sinkf);
        fa.compute(kh, vh, 16*n_step, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
        if (update(16*n_step)) return;
    }
    if (nq1 >= 8) {
        int n_step = nq1/8;
        FlashAttn<576, 512, 8, step_k> fa(scale, softcap, sinkf);
        fa.compute(kh, vh, 8*n_step, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
        if (update(8*n_step)) return;
    }
    if (nq1 >= 4) {
        int n_step = nq1/4;
        FlashAttn<576, 512, 4, step_k> fa(scale, softcap, sinkf);
        fa.compute(kh, vh, 4*n_step, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
        if (update(4*n_step)) return;
    }
    if (nq1 == 3) {
        FlashAttn<576, 512, 3, step_k> fa(scale, softcap, sinkf);
        fa.compute(kh, vh, 3, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
    }
    else if (nq1 == 2) {
        FlashAttn<576, 512, 2, step_k> fa(scale, softcap, sinkf);
        fa.compute(kh, vh, 2, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
    } else {
        FlashAttn<576, 512, 1, step_k> fa(scale, softcap, sinkf);
        fa.compute(kh, vh, 1, nk1, stride_q, stride_m, stride_qkv, q, mask, qkv, M, S);
    }
}

template <int step_k>
inline bool iqk_deepseek_helper(ggml_type type_k,
                        int nq1, int nk1, int stride_q, int stride_k, int stride_v, int stride_m, int stride_qkv,
                        const float * q, const char * k, const char * v, const char * mask,
                        float scale, float softcap, float * qkv, const float * sinkf, float * M, float * S) {
    if (type_k == GGML_TYPE_Q8_0) {
        HelperQ80 kh((const char *)k, stride_k);
        HelperQ80 vh((const char *)v, stride_v);
        iqk_deepseek_helper<step_k>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        return true;
    }
    if (type_k == GGML_TYPE_Q8_0_R8) {
        HelperQ80R8<576> kh((const char *)k, stride_k);
        HelperQ80 vh((const char *)v, stride_v);
        iqk_deepseek_helper<step_k>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        return true;
    }
    if (type_k == GGML_TYPE_Q6_0) {
        HelperQ60 kh((const char *)k, stride_k);
        HelperQ60 vh((const char *)v, stride_v);
        iqk_deepseek_helper<step_k>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        return true;
    }
#if GGML_IQK_FA_ALL_QUANTS
    if (type_k == GGML_TYPE_Q8_KV) {
        HelperQ8KV<576> kh((const char *)k, stride_k);
        HelperQ8KV<512> vh((const char *)v, stride_v);
        iqk_deepseek_helper<step_k>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        return true;
    }
#endif
    if (type_k == GGML_TYPE_F16) {
        HelperF16 kh((const char *)k, stride_k);
        HelperF16 vh((const char *)v, stride_v);
        iqk_deepseek_helper<step_k>(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, mask, scale, softcap, qkv, sinkf, M, S);
        return true;
    }
#ifdef __AVX512BF16__
    if (type_k == GGML_TYPE_BF16) {
        HelperBF16<576, step_k> kh((const char *)k, stride_k);
        HelperBF16<512, step_k> vh((const char *)v, stride_v);
        if (nq1 % 8 == 0) {
            FlashAttnBF16<576, 512, 8, step_k> fa(scale, softcap, sinkf);
            fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
        } else {
            FlashAttnBF16<576, 512, 1, step_k> fa(scale, softcap, sinkf);
            fa.compute(kh, vh, nq1, nk1, stride_q, stride_m, stride_qkv, q, (const char *)mask, qkv, M, S);
        }
        return true;
    }
#endif
    return false;
}

}

IQK_FA_CASE(iqk_fa_576_512) {

    auto type_k = ggml_type(int_type_k);
    auto type_v = ggml_type(int_type_v);

    if (!(type_k == type_v || (type_k == GGML_TYPE_Q8_0_R8 && type_v == GGML_TYPE_Q8_0))) {
        return false;
    }
    stride_q /= sizeof(float); // q stride as float
    return iqk_deepseek_helper<32>(type_k, nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                        q, (const char *)k, (const char *)v, (const char *)mask, scale, softcap, qkv, sinkf, M, S);

}

#endif
