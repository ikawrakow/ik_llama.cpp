#pragma once
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

bool iqk_mul_mat_ext(int task_type, long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, float alpha, float beta, int ith, int nth);

static bool iqk_mul_mat(int task_type, long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth) {
    return iqk_mul_mat_ext(task_type, Nx, Ny, ne00, typeA, A, strideA, typeB, B, strideB, C, stride_C, 1.f, 0.f, ith, nth);
}

bool iqk_mul_mat_moe(long Nx, long Ny, long ne00, int ne11,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long nb1, long nb2, const void * vrow_mapping, int ith, int nth);

bool iqk_soft_max(int nc, const float * sp, float * dp, float * wp, const char * bias, float scale, float slope, bool bias_is_f16);

#ifdef __cplusplus
}
#endif
