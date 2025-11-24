#include "iqk/iqk_config.h"

#if defined IQK_IMPLEMENT && defined GGML_IQK_FLASH_ATTENTION

#include "iqk/fa/iqk_fa_templates.h"

IQK_FA_CASE(iqk_fa_192_192) {

    auto type_k = ggml_type(int_type_k);
    auto type_v = ggml_type(int_type_v);

    stride_q /= sizeof(float); // q stride as float
    auto ck = (const char *)k;
    auto cv = (const char *)v;
    auto cm = (const char *)mask;

#ifdef __AVX512BF16__
    if (type_k == GGML_TYPE_BF16) {
        if (type_v != GGML_TYPE_BF16) return false; // we do not support mixing bf16 k-cache with other types
        if (nk%64 == 0) {
            iqk_flash_helper_T<192, 192, 64>(nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                    q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
            return true;
        }
        iqk_flash_helper_T<192, 192, 32>(nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                    q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
        return true;
    }
#endif

    if (nk%128 == 0) {
        return iqk_flash_helper_T<192, 192, 128>(type_k, type_v, nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
    }
    if (nk%64 == 0) {
        return iqk_flash_helper_T<192, 192, 64>(type_k, type_v, nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);
    }

    return iqk_flash_helper_T<192, 192, 32>(type_k, type_v, nq, nk, stride_q, stride_k, stride_v, stride_m, stride_qkv,
                q, ck, cv, cm, scale, softcap, qkv, sinkf, M, S);

}

#endif
