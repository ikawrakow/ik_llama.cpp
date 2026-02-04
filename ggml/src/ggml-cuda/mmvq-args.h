#pragma once

#include "common.cuh"

struct mmvq_args {
    const void * vx_u;
    const void * vx_g;
    const void * bias_u;
    const void * bias_g;
    const void * vy;
    float      * dst;
    const char * ids_data;
    const int    ncols_x;
    const int    nrows_x;
    const int    nrows_y;
    const int    ncols_y;
    const int    nrows_dst;
    const int    ne2;
    const uint64_t nb02;
    const uint64_t nb12;
    const uint64_t nb2;
    const uint64_t ids_nb0;
    const uint64_t bias_nb1;
    ggml_unary_op  unary_op;
    float          limit;
};

