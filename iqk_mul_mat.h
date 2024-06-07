#pragma once
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

bool iqk_mul_mat(long Nx, long Ny, long ne00, int typeA, const void * A, const void * B,
        float * C, long stride_C, int ith, int nth);

bool iqk_mul_mat_moe(long, long, long, int, int, const void *, const void *,
        float *, long, long, const void *, int, int);


#ifdef __cplusplus
}
#endif
