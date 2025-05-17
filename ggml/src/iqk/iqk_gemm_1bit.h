#pragma once

#include "iqk_common.h"

#ifdef IQK_IMPLEMENT

#include <array>

bool iqk_set_kernels_1bit(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16);

#endif
