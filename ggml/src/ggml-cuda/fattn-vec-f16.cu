#include "fattn-vec-f16.cuh"
#include "fattn-vec-f16-interface.cuh"

#define FATTN_VEC_F16_CASE(D, type_K, type_V)                               \
    if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f16_case<D, D, type_K, type_V>(ctx, dst); \
        return;                                                             \
    }                                                                       \

#define FATTN_VEC_F16_CASE_DKDV(Dk, Dv, type_K, type_V)                               \
    if (Q->ne[0] == (Dk) && V->ne[0] == Dv && K->type == (type_K) && V->type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f16_case<Dk, Dv, type_K, type_V>(ctx, dst); \
        return;                                                             \
    }                                                                       \

void ggml_cuda_flash_attn_ext_vec_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16 )

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16)

    FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(256, GGML_TYPE_Q8_0,GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0,   GGML_TYPE_IQ4_NL)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q6_0,   GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q6_0,   GGML_TYPE_Q6_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0,   GGML_TYPE_Q6_0)

    FATTN_VEC_F16_CASE_DKDV(192, 128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE_DKDV(192, 128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#else
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(256, GGML_TYPE_Q8_0,GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_NL)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0,   GGML_TYPE_IQ4_NL)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q6_0,   GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0,   GGML_TYPE_Q6_0)

    FATTN_VEC_F16_CASE_DKDV(192, 128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE_DKDV(192, 128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

#endif // GGML_CUDA_FA_ALL_QUANTS

    on_no_fattn_vec_case(Q->ne[0], V->ne[0]);
}

bool ggml_cuda_fattn_vec_f16_is_supported([[maybe_unused]] ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    auto K = dst->src[1];
    auto V = dst->src[2];
    if (K->ne[0] != V->ne[0]) {
        if (K->ne[0] != 192 || V->ne[2] != 128) return false;
        if (K->type != V->type) return false;
        return K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0;
    }
#ifdef GGML_CUDA_FA_ALL_QUANTS
    if (K->ne[0] == 64) {
        return K->type == GGML_TYPE_F16 &&
              (V->type == GGML_TYPE_F16  || V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q4_1 ||
               V->type == GGML_TYPE_Q5_0 || V->type == GGML_TYPE_Q5_1 || V->type == GGML_TYPE_Q8_0);
    }
    if (K->ne[0] == 256) {
        return K->type == V->type && (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0);
    }
    if (K->ne[0] != 128 || V->ne[0] != 128) return false;
    if ((K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_Q4_1 || K->type == GGML_TYPE_Q5_0 || K->type == GGML_TYPE_Q5_1 ||
         K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_F16) &&
        (V->type == GGML_TYPE_Q4_0 || V->type == GGML_TYPE_Q4_1 || V->type == GGML_TYPE_Q5_0 || V->type == GGML_TYPE_Q5_1 ||
         V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_F16)) return true;
    return (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_IQ4_NL) ||
           (K->type == GGML_TYPE_Q6_0 && V->type == GGML_TYPE_Q5_0)   ||
           (K->type == GGML_TYPE_Q6_0 && V->type == GGML_TYPE_Q6_0)   ||
           (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q6_0)   ||
           (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_IQ4_NL);
#else
    if (K->ne[0] == 128) {
        if (K->type == V->type) {
            return K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_IQ4_NL;
        }
        return (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_IQ4_NL) ||
               (K->type == GGML_TYPE_Q6_0 && V->type == GGML_TYPE_Q5_0)   ||
               (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q6_0)   ||
               (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_IQ4_NL);
    }
    if (K->type != V->type) return false;
    if (K->ne[0] == 64) {
        return K->type == GGML_TYPE_F16;
    }
    if (K->ne[0] == 256) {
        return K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0;
    }
    return false;
#endif
}
