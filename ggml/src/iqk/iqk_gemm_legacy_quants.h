#pragma once

#include "iqk_common.h"

#ifdef IQK_IMPLEMENT

#include <array>
#include <utility>

bool iqk_set_kernels_legacy_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);

void iqk_gemm_legacy_fa(int D, int nq, int type_k, const char * k, size_t stride_k, DataInfo& info, int k_step);

#endif
