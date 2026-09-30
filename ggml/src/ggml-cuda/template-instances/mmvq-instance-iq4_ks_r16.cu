#include "../iqk_mmvq_templates.cuh"

// qk = 256
// qr = 2
// qi = 256 / 8 = 32
// -> blocks_per_row_x = nx/256
//    blocks_per_iter  = 2
//    kbx = tid/16 = 0 or 1
//    iqs = 2 * (tid % 16) = 0, 2, 4, ..., 30
//__device__ __forceinline__ void vec_dot_iq4_ks_r4_q8_1(
//    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {
//
//    const float * dptr = (const float *)vbq;
//    const block_iq4_ks_r4 * bq4 = (const block_iq4_ks_r4 *)(dptr + 4) + kbx;
//
//    // iqs is 0...30 in steps of 2
//    const int ib16 = iqs/2;
//    const float d8 = __low2float(bq8_1[ib16/2].ds);
//    const int32_t  * q8 = (const int *)bq8_1[ib16/2].qs + 4*(ib16%2);
//
//    int ib32 = ib16/2;
//    int is   = ib16%2;
//    const uint32_t * scales32 = (const uint32_t *)bq4->scales;
//    int scales = __vsub4(scales32[ib32] & 0xfefefefe, 0x7f7f7f7f);
//    const int8_t * s8 = (const int8_t *)&scales;
//    int2 val;
//    const int * q4 = (const int *)bq4->qs + 16*ib32;
//    for (int i = 0; i < 4; ++i) {
//        auto values = iq4k_values + ((bq4->scales[4*ib32+i] & 1) << 4);
//        int sumi = 0;
//        val  = get_int_from_table_16(q4[i+4*is+0], values);
//        sumi = ggml_cuda_dp4a(val.x, q8[0], ggml_cuda_dp4a(val.y, q8[2], sumi));
//        val  = get_int_from_table_16(q4[i+4*is+8], values);
//        sumi = ggml_cuda_dp4a(val.x, q8[1], ggml_cuda_dp4a(val.y, q8[3], sumi));
//        const float d = dptr[i] * d8;
//        result[i] += d * sumi * s8[i];
//    }
//}

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_KS_R16> {
    static constexpr int qk = QK8_0;   // 32
    static constexpr int qr = QR4_0;   //  2
    static constexpr int qi = 32; //QI4_0;   // 32/(4*2) = 4
};

//template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_y, int n_interleaved = 1, vec_dot_q_tail_cuda_t vec_dot_tail = nullptr>
//static __device__ void iqk_mul_mat_vec_q_kernel(
//    const void * __restrict__ vx, const void * __restrict__ vy,
//    const float * bias, float * __restrict__ dst,
//    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst, const int64_t row_size) {
//
//    constexpr int qk  = ggml_cuda_type_traits<type>::qk;  --> 32
//    constexpr int qi  = ggml_cuda_type_traits<type>::qi;  -->  4   suppose qi = 32
//
//#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
//    constexpr int nwarps              = 1;
//    constexpr int rows_per_cuda_block = n_interleaved;
//#else
//    constexpr int nwarps              = n_interleaved == 1 ? ncols_y <= 4 ? 4 : 2 : 1;                      ->  1
//    constexpr int rows_per_cuda_block = n_interleaved == 1 ? ncols_y == 1 ? 1 : 2 : n_interleaved;          -> 16
//#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)
//
//    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;            0...31
//    const     int row0 = rows_per_cuda_block*blockIdx.x;
//    const     int blocks_per_row_x = ncols_x / qk;                      nx / 32
//    const     int blocks_per_col_y = nrows_y / QK8_1;                   nx / 32
//    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;        2*1*32/4 = 16 for vdr = 2, 32 for vdr = 4, suppose vdr = 1, qi = 32 -> blocks_per_iter = 1
//
//// partial sum for each thread
//    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};
//
//    const block_q8_1 * y = (const block_q8_1 *) vy;
//
//    int kbx = tid / (qi/vdr); (0...31)/2  -> 0...15 for vdr = 2, 0...31 for vdr = 4  | suppose vdr = 1, qi = 32 -> kbx = 0, 1, ..., nx/32
//    for (; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
//        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx -> kby = kbx
//
//        // x block quant index when casting the quants to int
//        const int kqs = vdr * (tid % (qi/vdr));  2 * ((0...31) % 2)  0 or 2 for vdr = 2, always 0 for vdr = 4 | uppose vdr = 1, qi = 32 -> kqs = 1 * (tid % 32) = tid
//
//#pragma unroll
//        for (int j = 0; j < ncols_y; ++j) {
//            if constexpr (n_interleaved == 1) {
//#pragma unroll
//                for (int i = 0; i < rows_per_cuda_block; ++i) {
//                    vec_dot_q_cuda((const void *)((const char *)vx + (row0 + i)*row_size),
//                            &y[j*blocks_per_col_y + kby], kbx, kqs, &tmp[j][i]);
//                }
//            } else {
//                vec_dot_q_cuda((const void *)((const char *)vx + row0*row_size),
//                    &y[j*blocks_per_col_y + kby], kbx, kqs, tmp[j]);
//            }
//        }
//    }
//    suppose vdr = 8 -> blocks_per_iter = 8, kbx = tid/4 = 0...7, kqs = 8*((0...31) % 4) = 0, 8, 16, 24

// qi = 32, vdr = 2: pretty much the same as qi = 32, vdr = 1
// blocks_per_iter = 2, kbx = tid/16 = 0 or 1, iqs = 2 * (tid%16) = 0, 2, ..., 30
__device__ __forceinline__ void vec_dot_iq4_ks_r16_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {

    const float * dptr = (const float *)vbq;
    const auto bq4 = (const block_iq4_ks_r16 *)(dptr + 16) + kbx;

    //printf("kbx = %d, iqs = %d, threadIdx = %d, %d\n", kbx, iqs, threadIdx.x, threadIdx.y);
    //return;

    const int il = iqs / 16;  // 0 or 1
    const int ir = iqs % 16;  // 0, 2, 4, 6, 8, 10, 12, 14

    const float d8 = __low2float(bq8_1->ds);
    const int32_t * q8 = (const int *)bq8_1->qs + 2*il;

    for (int j = 0; j < 2; ++j) {
        auto values = iq4k_values + ((bq4->scales[ir+j] & 1) << 4);
        auto q4 = (const int *)bq4->qs + 32*il + ir + j;
        int sumi = 0;
        int2 val = get_int_from_table_16(q4[0], values);
        sumi = ggml_cuda_dp4a(val.x, q8[0], ggml_cuda_dp4a(val.y, q8[4], sumi));
        val = get_int_from_table_16(q4[16], values);
        sumi = ggml_cuda_dp4a(val.x, q8[1], ggml_cuda_dp4a(val.y, q8[5], sumi));

        float d4 = dptr[ir + j] * ((bq4->scales[ir + j] & 254) - 127);
        result[ir + j] += d4 * d8 * sumi;
    }
}

// qi = 32, vdr = 1
//__device__ __forceinline__ void vec_dot_iq4_ks_r16_q8_1(
//    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {
//
//    const float * dptr = (const float *)vbq;
//    const auto bq4 = (const block_iq4_ks_r16 *)(dptr + 16) + kbx;
//
//    //printf("kbx = %d, iqs = %d, threadIdx = %d, %d\n", kbx, iqs, threadIdx.x, threadIdx.y);
//    //return;
//
//    const int il = iqs / 16;  // 0 or 1
//    const int ir = iqs % 16;  // 0...15
//
//    const float d8 = __low2float(bq8_1->ds);
//    const int32_t * q8 = (const int *)bq8_1->qs + 2*il;
//
//    auto values = iq4k_values + ((bq4->scales[ir] & 1) << 4);
//    auto q4 = (const int *)bq4->qs + 32*il + ir;
//    int sumi = 0;
//    int2 val = get_int_from_table_16(q4[0], values);
//    sumi = ggml_cuda_dp4a(val.x, q8[0], ggml_cuda_dp4a(val.y, q8[4], sumi));
//    val = get_int_from_table_16(q4[16], values);
//    sumi = ggml_cuda_dp4a(val.x, q8[1], ggml_cuda_dp4a(val.y, q8[5], sumi));
//
//    float d4 = dptr[ir] * ((bq4->scales[ir] & 254) - 127);
//    result[ir] += d4 * d8 * sumi;
//}
// q4_0: qi = 4, vdr = 2 -> we should use qi = 64, vdr = 32
// -> blocks_per_iter = 16, kbx = tid / 2 = 0...15, iqs = 32*(tid%2) = 0 or 32
// what if vdr = 64 and qi = 64
// -> blocks_per_iter = 32, kbx = tid, iqs = 0
// qi = 32, vdr = 8
//__device__ __forceinline__ void vec_dot_iq4_ks_r16_q8_1(
//    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs, float * result) {
//
//    const float * dptr = (const float *)vbq;
//    const auto bq4 = (const block_iq4_ks_r16 *)(dptr + 16) + kbx;
//
//    const float d8 = __low2float(bq8_1[kbx].ds);
//    const int32_t * q8 = (const int *)bq8_1[kbx].qs;
//
//    auto q4 = (const int *)bq4->qs;
//#pragma unroll
//    for (int i = 0; i < 16; ++i) {
//        auto values = iq4k_values + ((bq4->scales[i] & 1) << 4);
//        int sumi = 0;
//        for (int j = 0; j < 4; ++j) {
//            int2 val = get_int_from_table_16(q4[16*j + i], values);
//            sumi = ggml_cuda_dp4a(val.x, q8[j], ggml_cuda_dp4a(val.y, q8[j+4], sumi));
//        }
//        float d4 = dptr[i] * ((bq4->scales[i] & 254) - 127);
//        result[i] += d4 * d8 * sumi;
//    }
//}

void mul_mat_vec_iq4_ks_r16_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    //printf("=========================== %s: ncols_x = %d\n", __func__, args.ncols_x);
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KS_R16, 2, vec_dot_iq4_ks_r16_q8_1, 16>(args, stream);
}

