//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "iqk_config.h"
#include "iqk_mul_mat.h"
#include "iqk_flash_impl.h"

#if defined IQK_IMPLEMENT && defined GGML_IQK_FLASH_ATTENTION

#include <algorithm>
#include <cstdio>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>

namespace {
inline uint32_t simple_gcd(uint32_t a, uint32_t b) {
    while (a != b) {
        if (a > b) a -= b;
        else b -= a;
    }
    return a;
}
inline void accumulate_qkv(int Dv, float& M, float& S, float Mj, float Sj, float * Racc, const float * R) {
    if (Mj == -INFINITY) return;
    if (Mj > M) {
        if (M == -INFINITY) {
            std::memcpy(Racc, R, Dv*sizeof(float));
            S = Sj;
        } else {
            float c = exp(M - Mj);
            S = c*S + Sj;
            for (int i = 0; i < Dv; ++i) Racc[i] = c*Racc[i] + R[i];
        }
        M = Mj;
    } else {
        float c = exp(Mj - M);
        S += c*Sj;
        for (int i = 0; i < Dv; ++i) Racc[i] += c*R[i];
    }
}
}

// TODO: get the ggml_type enum here without polution
//
extern "C" IQK_API bool iqk_flash_attn_noalibi(int type_q, int type_mask, float max_bias,
                            int neq3, int neq2, long nbq3, long nbq2,
                            int nek3, int nek2, long nbk3, long nbk2,
                            int nev3, int nev2, long nbv3, long nbv2,
                            int ne2,  int ne1,  long nb1,
                            int int_type_k_in,      // type of k
                            int int_type_v,         // type of v
                            int Dk,                 // K head size
                            int Dv,                 // V head size
                            int neq1,               // number of columns in q
                            int nek1,               // number of rows in k
                            int stride_q,           // distance between q columns in bytes
                            int stride_k,           // distance between k rows in bytes
                            int stride_v,           // distance between v rows in bytes
                            int stride_m,           // distance between mask rows (in bytes
                            const void  * q,        // q matrix.
                            const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            float         scale,    // scale applied before softmax
                            float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            float       * qkv,      // v*softmax(scale*(k*q))
                            [[maybe_unused]] void * work_buffer_in, [[maybe_unused]] barrier_t barrier, [[maybe_unused]] void * barrier_data,
                            int ith, int nth) {

    if (type_q != 0 || type_mask != 1 || max_bias > 0) return false;

    int rk2 = neq2/nek2;
    int rv2 = neq2/nev2;
    int rk3 = neq3/nek3;
    int rv3 = neq3/nev3;

    int int_type_k = int_type_k_in;
    auto work_buffer = work_buffer_in;
    if (neq1 >= 8 || (rk2 >= 8 && nek2 > 1)) {
        uint64_t row_size = 0;
        work_buffer = iqk_repack_k(int_type_k, Dk, nek1, nek2, nek3, stride_k, nbk2, nbk3, k, work_buffer_in, ith, nth, int_type_k, row_size);
        if (int_type_k != int_type_k_in) {
            stride_k = row_size;
            nbk2 = stride_k*nek1;
            nbk3 = nbk2*nek2;
            k = work_buffer_in;
            barrier(barrier_data);
        }
    }
    //uint64_t row_size = 0;
    //auto work_buffer = iqk_repack_k(int_type_k, Dk, nek1, nek2, nek3, stride_k, nbk2, nbk3, k, work_buffer_in, ith, nth, int_type_k, row_size);
    //if (int_type_k != int_type_k_in) {
    //    stride_k = row_size;
    //    nbk2 = stride_k*nek1;
    //    nbk3 = nbk2*nek2;
    //    k = work_buffer_in;
    //    barrier(barrier_data);
    //}

    // Getting confused all the time about where to load data from and store the results to
    // (especially when combining the results from the threads).
    // So, for now, making it work just for MLA (nek2 = 1).
    // I think it would also speed up things for GQA, but I'm leaving this for another day.
    if (neq3 == 1 && rk2 > 1 && neq1 == 1 && nth >= 1 && nek1/32 > 1 && nek2 == 1) {
        int nstep_k = nek1/32;
        int gcd_k   = simple_gcd(nstep_k, nth);
        if (gcd_k >= 1) {
            int nth_k = nth/gcd_k;
            int ith_k = ith%gcd_k;
            int ith_q = ith/gcd_k;
            int nq_per_thread = (rk2 + nth_k - 1)/nth_k;
            if (nq_per_thread > 1) {
                int ith_mid = nth_k;
                int nq_this_thread = nq_per_thread;
                if (nq_per_thread*nth_k > rk2) {
                    ith_mid = rk2 - nth_k*(nq_per_thread - 1);
                    if (ith_q >= ith_mid) --nq_this_thread;
                }
                int j_mid = ith_mid*nq_per_thread;
                auto work = (char *)work_buffer;
                auto size_thread = (Dv + 16)*nq_per_thread*sizeof(float);
                auto result_buffer = work;

                auto kth = (const char *)k + ith_k*(nek1/gcd_k)*stride_k;
                auto vth = (const char *)v + ith_k*(nek1/gcd_k)*stride_v;
                auto q_offset = ith_q < ith_mid ? ith_q*nq_per_thread*nbq2 : (ith_mid*nq_per_thread + (ith_q - ith_mid)*nq_this_thread)*nbq2;
                auto qth = (const char *)q + q_offset;
                auto mth = (const char *)mask + ith_k*(nek1/gcd_k)*sizeof(uint16_t); // we don't have ggml_half available here

                // Each thread will produce a result of size Dv*nq_this_thread*sizeof(float)
                // In addition, we need M, S for the nq_this_thread rows the thread is processing
                // => (Dv + 2)*nq_per_thread*sizeof(float). We use (Dv + 16) instead to make sure threads are not
                // writing onto the same cache line.
                auto work_this_thread = (float *)(result_buffer + ith*size_thread);
                if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                            Dk, Dv, nq_this_thread, nek1/gcd_k, nbq2, stride_k, stride_v, 0, Dv, //Dk*sizeof(uint16_t), Dv,
                            (const float *)qth, (const void *)kth, (const void *)vth, (const void *)mth,
                            scale, softcap,
                            work_this_thread, work_this_thread + (Dv+0)*nq_this_thread, work_this_thread + (Dv+1)*nq_this_thread)) return false;

                barrier(barrier_data);

                // There are nek1/gcd_k contributions for each j that we need to sum up
                // Thread i computed k/v (i%gcd_k)*(nek1/gcd_k) for j (i/gcd_k)*(rk2/nth_k)...((i/gcd_k)+1)*(rk2/nth_k) and results at offset i*size_thread

                // TODO: simdify this
                // TODO: if nth > rk2, have threads process portions of the rows instead of entire rows as it is now
                for (int j = ith; j < rk2; j += nth) {
                    auto Racc = qkv + j*nb1/sizeof(float);
                    float M = -INFINITY, S = 0;
                    int jth_first, jj, nq_this_j;
                    if (j < j_mid) {
                        jth_first = j/nq_per_thread;
                        jj = j%nq_per_thread;
                        nq_this_j = nq_per_thread;
                    } else {
                        jth_first = ith_mid + (j - j_mid)/(nq_per_thread-1);
                        jj = (j - j_mid)%(nq_per_thread-1);
                        nq_this_j = nq_per_thread - 1;
                    }
                    jth_first *= gcd_k;
                    for (int jth = jth_first; jth < jth_first + gcd_k; ++jth) {
                        auto R = (const float *)(result_buffer + jth*size_thread);
                        auto Mj = R + Dv*nq_this_j;
                        auto Sj = Mj + nq_this_j;
                        R += jj*Dv;
                        accumulate_qkv(Dv, M, S, Mj[jj], Sj[jj], Racc, R);
                    }
                    float norm = S > 0 ? 1/S : 1;
                    for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
                }
                return true;
            }
        }
    }

    if (neq3 == 1 && rk2 > 1 && rk2 == rv2 && neq1 == 1 && nth >= 1 && nek2*nek1 >= 32*nth) {
        auto result_size = (Dv + 16)*rk2*sizeof(float);
        int gcd = simple_gcd(nek2, nth);
        if (false && gcd > 1) {
            int nth_g = nth/gcd;
            int ith_g = ith%nth_g;
            int nek1_32 = nek1/32;
            int nek1_pt = (nek1_32 + nth_g - 1)/nth_g;
            int ith_mid = nth_g;
            if (nek1_pt*nth_g > nek1_32) {
                ith_mid = nek1_32 - nth_g*(nek1_pt - 1);
            }
            nek1_pt *= 32;
            int nek1_mid = ith_mid*nek1_pt;
            int nek1_thread = ith_g < ith_mid ? nek1_pt : nek1_pt - 32;
            for (int ik02 = ith/nth_g; ik02 < nek2; ik02 += gcd) {
                int ik01 = ith_g < ith_mid ? ith_g*nek1_pt : nek1_mid + (ith_g - ith_mid)*nek1_thread;
                auto this_result = (float *)((char *)work_buffer + (ik02*nth_g + ith_g)*result_size);
                auto this_q = (const float *)((const char *)q + ik02*rk2*nbq2);
                auto this_k = (const char *)k + ik01*stride_k + ik02*nbk2;
                auto this_v = (const char *)v + ik01*stride_v + ik02*nbv2;
                auto this_m = (const char *)mask + ik01*sizeof(uint16_t); // we don't have ggml_half available here
                if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                            Dk, Dv, rk2, nek1_thread, nbq2, stride_k, stride_v, 0, Dv,
                            this_q, (const void *)this_k, (const void *)this_v, (const void *)this_m,
                            scale, softcap, this_result, this_result + (Dv+0)*rk2, this_result + (Dv+1)*rk2)) return false;
            }

            barrier(barrier_data);

            for (int iq2 = ith; iq2 < neq2; iq2 += nth) {
                int ik02 = iq2/rk2;
                int il = iq2 - ik02*rk2;
                auto Racc = qkv + iq2*nb1/sizeof(float);
                float M = -INFINITY, S = 0;
                for (int ig = 0; ig < nth_g; ++ig) {
                    int istep_k = ik02*nth_g + ig;
                    auto this_result = (float *)((char *)work_buffer + istep_k*result_size);
                    const float * R  = this_result + il*Dv;
                    const float * Mj = this_result + Dv*rk2;
                    const float * Sj = Mj + rk2;
                    accumulate_qkv(Dv, M, S, Mj[il], Sj[il], Racc, R);
                }
                float norm = S > 0 ? 1/S : 1;
                for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
            }
            return true;
        }
        int nth_k  = nth/gcd;
        int nek2_k = nek2/gcd;
        int nchunk = nek2_k*nek1/32;
        int npt = (nchunk + nth_k - 1)/nth_k;
        int nk;
        if (npt*nth_k == nchunk) {
            nk = 32 * (nek2*nek1/(32*nth));
        } else {
            //int nm = std::max(1, npt/8);
            int nm = 1;
            while (true) {
                if (nm*4 >= npt) break;
                nm *= 2;
            }
            nk = 32*nm;
        }
        //int nk = 32 * (nek2*nek1/(32*nth));
        int nkk = (nek1 + nk - 1)/nk;
        int nstep_k = nek2*nkk;
        //if (ith == 0) printf("rk2 = %d, nek1 = %d, nek2 = %d, nk = %d, nkk = %d, nstep_k = %d\n", (int)rk2, (int)nek1, (int)nek2, nk, nkk, nstep_k);
        for (int istep_k = ith; istep_k < nstep_k; istep_k += nth) {
            int ik02 = istep_k/nkk;
            int ik01 = nk*(istep_k - ik02*nkk);
            int this_nk = ik01 + nk <= nek1 ? nk : nek1 - ik01;
            if (this_nk <= 0) break;
            auto this_result = (float *)((char *)work_buffer + istep_k*result_size);
            auto this_q = (const float *)((const char *)q + ik02*rk2*nbq2);
            auto this_k = (const char *)k + ik01*stride_k + ik02*nbk2;
            auto this_v = (const char *)v + ik01*stride_v + ik02*nbv2;
            auto this_m = (const char *)mask + ik01*sizeof(uint16_t); // we don't have ggml_half available here
            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                     Dk, Dv, rk2, this_nk, nbq2, stride_k, stride_v, 0, Dv,
                     this_q, (const void *)this_k, (const void *)this_v, (const void *)this_m,
                     scale, softcap, this_result, this_result + (Dv+0)*rk2, this_result + (Dv+1)*rk2)) return false;
        }

        barrier(barrier_data);

        // We have nkk results for each head
        for (int iq2 = ith; iq2 < neq2; iq2 += nth) {
            // ik02*rk2 + il = iq2 (il = 0...rk2-1) => ik02 = iq2/rk2, il = iq2%rk2;
            int ik02 = iq2/rk2;
            int il = iq2 - ik02*rk2;
            auto Racc = qkv + iq2*nb1/sizeof(float);
            //std::memset(Racc, 0, Dv*sizeof(float));
            float M = -INFINITY, S = 0;
            for (int ikk = 0; ikk < nkk; ++ikk) {
                int istep_k = ik02*nkk + ikk;
                auto this_result = (float *)((char *)work_buffer + istep_k*result_size);
                const float * R  = this_result + il*Dv;
                const float * Mj = this_result + Dv*rk2;
                const float * Sj = Mj + rk2;
                accumulate_qkv(Dv, M, S, Mj[il], Sj[il], Racc, R);
            }
            float norm = S > 0 ? 1/S : 1;
            for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
        }
        return true;
    }

    // I keep changing my mind what is the best strategy to split the threads when processing
    // multiple heads. This is my current thinking, the commented out code below was the previous.
    int ntg = nth/simple_gcd(neq2*neq3, nth);
    int neq1g = (neq1 + ntg - 1)/ntg;
    //int64_t work_per_slice = D*nek1*neq1;
    //int ntg = 1;
    //
    // When neq1 is large, it is better to have more than one thread process one (iq2,iq3) matrix
    // But we also want each thread to process the same amount of rows, so neq1 must be a multiple of
    // the number of threads processing the (iq2, iq3) matrix.
    //
    //if (neq1 >= 8*nth) {
    //    if      (nth%8 == 0 && neq1%8 == 0 && work_per_slice >= (1 << 23)) ntg = 8;
    //    else if (nth%4 == 0 && neq1%4 == 0 && work_per_slice >= (1 << 21)) ntg = 4;
    //    else if (nth%2 == 0 && neq1%2 == 0 && work_per_slice >= (1 << 19)) ntg = 2;
    //}
    int counter = 0;
    for (int64_t iq3 = 0; iq3 < neq3; iq3++) {
        for (int64_t iq2 = 0; iq2 < neq2; iq2++) {
            if (counter++ % (nth/ntg) == ith/ntg) {
                int iq1 = (ith%ntg)*neq1g;
                int this_neq1 = std::min(neq1g, neq1-iq1);
                if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                        Dk, Dv, this_neq1, nek1, stride_q, stride_k, stride_v, stride_m, ne1*nb1/sizeof(float),
                        (const float *)((const char *)q + iq2*nbq2 + iq3*nbq3 + iq1*stride_q),
                        (const void  *)((const char *)k + iq2/rk2*nbk2 + iq3/rk3*nbk3),
                        (const void  *)((const char *)v + iq2/rv2*nbv2 + iq3/rv3*nbv3),
                        (const void  *)((const char *)mask + iq1*stride_m),
                        scale, softcap,
                        (float *)((char *)qkv + (iq3*ne2*ne1 + iq2 + iq1*ne1)*nb1), nullptr, nullptr)) return false;
            }
        }
    }

    return true;
}

#else

bool iqk_flash_attn_noalibi([[maybe_unused]] int type_q, [[maybe_unused]] int type_mask, [[maybe_unused]] float max_bias,
                            [[maybe_unused]] int neq3, [[maybe_unused]] int neq2, [[maybe_unused]] long nbq3, [[maybe_unused]] long nbq2,
                            [[maybe_unused]] int nek3, [[maybe_unused]] int nek2, [[maybe_unused]] long nbk3, [[maybe_unused]] long nbk2,
                            [[maybe_unused]] int nev3, [[maybe_unused]] int nev2, [[maybe_unused]] long nbv3, [[maybe_unused]] long nbv2,
                            [[maybe_unused]] int ne2,  [[maybe_unused]] int ne1,  [[maybe_unused]] long nb1,
                            [[maybe_unused]] int type_k,             // type of k
                            [[maybe_unused]] int type_v,             // type of v
                            [[maybe_unused]] int Dk,                 // K head size
                            [[maybe_unused]] int Dv,                 // V head size
                            [[maybe_unused]] int nq,                 // number of columns in q
                            [[maybe_unused]] int nk,                 // number of rows in k
                            [[maybe_unused]] int stride_q,           // distance between q columns in bytes
                            [[maybe_unused]] int stride_k,           // distance between k rows in bytes
                            [[maybe_unused]] int stride_v,           // distance between v rows in bytes
                            [[maybe_unused]] int stride_m,           // distance between mask rows (in bytes
                            [[maybe_unused]] const void  * q,        // q matrix.
                            [[maybe_unused]] const void  * k,        // k matrix. Assumed to be fp16, nq x nk elements
                            [[maybe_unused]] const void  * v,        // v matrix. Assumed to be fp16, nq x nk elements
                            [[maybe_unused]] const void  * mask,     // mask. If not null, assumed to be fp16. nq x nk elements
                            [[maybe_unused]] float         scale,    // scale applied before softmax
                            [[maybe_unused]] float         softcap,  // if > 0, a "soft-cap" operation is applied before softmax
                            [[maybe_unused]] float       * qkv,      // v*softmax(scale*(k*q))
                            [[maybe_unused]] void * work_buffer, [[maybe_unused]] barrier_t barrier, [[maybe_unused]] void * barrier_data,
                            [[maybe_unused]] int ith, [[maybe_unused]] int nth) {
    return false;
}

#endif

