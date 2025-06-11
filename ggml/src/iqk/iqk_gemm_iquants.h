#pragma once

#include "iqk_common.h"

#ifdef IQK_IMPLEMENT

#include <array>

bool iqk_set_kernels_iquants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);

bool iqk_convert_iquants_q80_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x);

#endif
