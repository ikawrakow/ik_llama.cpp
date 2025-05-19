#pragma once

#include "iqk_common.h"

#ifdef IQK_IMPLEMENT

#include <array>

bool iqk_set_kernels_float(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels);

void iqk_gemm_default_floats(int D, int nq, const char * vx, size_t bx, DataInfo& info, int k_step);

#endif
