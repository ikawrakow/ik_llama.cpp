#include "iqk_config.h"
#include "iqk_mul_mat.h"
#include "iqk_flash_impl.h"

#ifdef IQK_IMPLEMENT

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
}

// TODO: get the ggml_type enum here without polution
//
bool iqk_flash_attn_noalibi(int type_q, int type_mask, float max_bias,
                            int neq3, int neq2, long nbq3, long nbq2,
                            int nek3, int nek2, long nbk3, long nbk2,
                            int nev3, int nev2, long nbv3, long nbv2,
                            int ne2,  int ne1,  long nb1,
                            int int_type_k,         // type of k
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
                            [[maybe_unused]] void * work_buffer, [[maybe_unused]] barrier_t barrier, [[maybe_unused]] void * barrier_data,
                            int ith, int nth) {

    if (type_q != 0 || type_mask != 1 || max_bias > 0) return false;

    int rk2 = neq2/nek2;
    int rv2 = neq2/nev2;
    int rk3 = neq3/nek3;
    int rv3 = neq3/nev3;

    if (neq3 == 1 && rk2 > 1 && neq1 == 1 && nth >= 1 && nek1/32 > 1) {
        int nstep_k = nek1/32;
        int gcd_k   = simple_gcd(nstep_k, nth);
        if (gcd_k >= 1) {
            int nth_k = nth/gcd_k;
            if (rk2%nth_k == 0) {
                int ith_k = ith%gcd_k;
                int ith_q = ith/gcd_k;
                //if (ith == 0) {
                //    printf("==================== %s: nek1 = %d, rk2 = %d -> each thread will do %d k's with %d q's\n", __func__, nek1, rk2,
                //        nek1/gcd_k, rk2/nth_k);
                //    printf("stride_k = %d, stride_v = %d, nbk2 = %ld nbv2 = %ld nb1 = %ld\n", stride_k, stride_v, nbk2, nbv2, nb1);
                //    const uint16_t * m = (const uint16_t *)mask;
                //    int non = 0, last = -1;
                //    for (int j = 0; j < nek1; ++j) if (m[j] == 0) { ++non; last = j; }
                //    printf("There are %d entries set in mask, last was %d\n", non, last);
                ////    printf("Mask:\n");
                ////    for (int j = 0; j < nek1; ++j) printf(" 0x%04x", m[j]);
                ////    printf("\n");
                //}
                //printf("ith_k = %d, ith_q = %d\n", ith_k, ith_q);
                auto kth = (const char *)k + ith_k*(nek1/gcd_k)*stride_k;
                auto vth = (const char *)v + ith_k*(nek1/gcd_k)*stride_v;
                auto qth = (const char *)q + ith_q*(rk2/nth_k)*nbq2;
                auto mth = (const char *)mask + ith_k*(nek1/gcd_k)*sizeof(uint16_t); // we don't have ggml_half available here
                auto work = (char *)work_buffer;

                // Each thread will produce a result of size Dv*(rk2/nth_k)*sizeof(float)
                // In addition, we need M, S for the rk2/nth_k rows the thread is processing
                // => (Dv + 2)*rk2/nth_k*sizeof(float). We use (Dv + 16) instead to make sure threads are not
                // writing onto the same cache line.
                auto size_thread = (Dv + 16)*rk2/nth_k*sizeof(float);
                auto result_buffer = work;
                auto work_this_thread = (float *)(result_buffer + ith*size_thread);
                //printf("Thread %d: computing k,v = %d, q = %d...%d\n", ith, ith_k*(nek1/gcd_k), ith_q*(rk2/nth_k), ith_q*(rk2/nth_k)+rk2/nth_k-1);
                if (!iqk_flash_attn_impl(int_type_k, int_type_v,
                            Dk, Dv, rk2/nth_k, nek1/gcd_k, nbq2, stride_k, stride_v, 0, Dv, //Dk*sizeof(uint16_t), Dv,
                            (const float *)qth, (const void *)kth, (const void *)vth, (const void *)mth,
                            scale, softcap,
                            work_this_thread, work_this_thread + (Dv+0)*rk2/nth_k, work_this_thread + (Dv+1)*rk2/nth_k)) return false;

                //if (ith == 0) {
                //    for (int jth = 0; jth < nth; ++jth) {
                //        auto R = (const float *)(result_buffer + jth*size_thread);
                //        auto M = R + Dv*rk2/nth_k;
                //        auto S = M + rk2/nth_k;
                //        printf("--- Thread %d\n", jth);
                //        for (int j = 0; j < rk2/nth_k; ++j) printf("    %2d  %g  %g\n", j, M[j], S[j]);
                //    }
                //}
                barrier(barrier_data);

                for (int j = ith; j < rk2; j += nth) {
                    auto Racc = qkv + j*nb1/sizeof(float);
                    float M = -INFINITY, S = 0;
                    int jth_q = j/(rk2/nth_k);
                    int jj = j%(rk2/nth_k);
                    for (int j1 = 0; j1 < rk2/nth_k; ++j1) {
                        auto R = (const float *)(result_buffer + (jth_q*(rk2/nth_k) + j1)*size_thread);
                        auto Mj = R + Dv*rk2/nth_k;
                        auto Sj = Mj + rk2/nth_k;
                        R += jj*Dv;
                        if (Mj[jj] == -INFINITY) continue;
                        if (Mj[jj] > M) {
                            if (M == -INFINITY) {
                                std::memcpy(Racc, R, Dv*sizeof(float));
                                S = Sj[jj];
                            } else {
                                float c = exp(M - Mj[jj]);
                                S = c*S + Sj[jj];
                                for (int i = 0; i < Dv; ++i) Racc[i] = c*Racc[i] + R[i];
                            }
                            M = Mj[jj];
                        } else {
                            float c = exp(Mj[jj] - M);
                            S += c*Sj[jj];
                            for (int i = 0; i < Dv; ++i) Racc[i] += c*R[i];
                        }
                    }
                    float norm = S > 0 ? 1/S : 1;
                    for (int i = 0; i < Dv; ++i) Racc[i] *= norm;
                }
                return true;

                int nv = (Dv/16 + nth - 1)/nth;
                int mv1 = nth;
                if (nv*nth != Dv/16) mv1 = nv*nth - Dv/16;
                int start_i;
                if (ith < mv1) start_i = ith*nv*16;
                else {
                    if (nv == 1) return true;
                    start_i = (mv1*nv + (ith - mv1)*(nv-1))*16;
                    --nv;
                }
                for (int j = 0; j < rk2; ++j) {
                    auto Racc = qkv + j*nb1/sizeof(float) + start_i;
                    float M = -INFINITY, S = 0;
                    int jth_q = j/(rk2/nth_k);
                    int jj = j%(rk2/nth_k);
                    for (int j1 = 0; j1 < rk2/nth_k; ++j1) {
                        auto R = (const float *)(result_buffer + (jth_q*(rk2/nth_k) + j1)*size_thread);
                        auto Mj = R + Dv*rk2/nth_k;
                        auto Sj = Mj + rk2/nth_k;
                        R += jj*Dv + start_i;
                        //printf("        result from thread %d for row %d: M = %g, Mj[j] = %g, S = %g, Sj[j] = %g\n", jth_q*gcd_k + j1, j, M, Mj[jj], S, Sj[jj]);
                        if (Mj[jj] == -INFINITY) continue;
                        if (Mj[jj] > M) {
                            if (M == -INFINITY) {
                                std::memcpy(Racc, R, 16*nv*sizeof(float));
                                S = Sj[jj];
                            } else {
                                float c = exp(M - Mj[jj]);
                                S = c*S + Sj[jj];
                                for (int i = 0; i < 16*nv; ++i) Racc[i] = c*Racc[i] + R[i];
                            }
                            M = Mj[jj];
                        } else {
                            float c = exp(Mj[jj] - M);
                            S += c*Sj[jj];
                            for (int i = 0; i < 16*nv; ++i) Racc[i] += c*R[i];
                        }
                    }
                    //printf("Row %d: M = %g, S = %g\n", j, M, S);
                    float norm = S > 0 ? 1/S : 1;
                    for (int i = 0; i < 16*nv; ++i) Racc[i] *= norm;
                    //for (int i = 0; i < Dv; ++i) {
                    //    Racc[i] *= norm;
                    //    printf("  %3d  %g\n", i, Racc[i]);
                    //}
                }
                return true;

                //// So, now we have gcd_k results for each of the rk2 rows.
                //// Each thread will compute Dv/gcd_v elements.
                //// To ciompute the Dv/gcd_v elements, each thread needs to sum over the gcd_k partial results.
                //// Dv = 512, nth = 16 => gcd_v = 16 -> each thread processes 512/16 = 32 elements
                //// Dv = 192, nth = 32 => gcd_v = 4  -> each thread processes 48 elements
                //int nv = ((Dv/16) + gcd_k - 1)/gcd_k;
                //int mv1 = gcd_k;
                //if (nv*gcd_k != Dv/16) mv1 = nv*gcd_k - Dv/16;
                //// The first mv1 threads will process 16*nv elements, the remaining gcd - mv1 will do (nv - 1)*16.
                //int start_i;
                //if ((ith%gcd_k) < mv1) start_i = (ith%gcd_k)*nv*16;
                //else {
                //    if (--nv == 0) return true;
                //    start_i = mv1*nv + ((ith%gcd_k) - mv1)*nv*16;
                //}
                ////printf("Thread %d: computing %d elements in %d rows starting at %d by summing %d rows\n", ith, 16*nv, rk2/nth_k, start_i, gcd_k);
                ////if (ith == 0) printf("Computing %d rows by summing %d thread results\n", rk2/nth_k, gcd_k);
                //for (int j = 0; j < rk2/nth_k; ++j) {
                //    auto offset = start_i + Dv*j;
                //    float S = 0, M = -INFINITY;
                //    auto Racc = qkv + (ith_q*rk2/nth_k + j)*nb1/sizeof(float) + start_i;
                //    //if (ith == 0) printf("    working on row %d. Storing into qkv at %ld, reading from offset %d\n", j, (ith_q*rk2/nth_k + j)*nb1/sizeof(float), offset);
                //    for (int j1 = 0; j1 < gcd_k; ++j1) {
                //        auto R = (const float *)(result_buffer + (ith_q*gcd_k + j1)*size_thread);
                //        auto Mj = R + Dv*rk2/nth_k;
                //        auto Sj = Mj + rk2/nth_k;
                //        R += offset;
                //        //if (ith == 0) printf("        thread %d: M = %g, Mj[j] = %g, S = %g, Sj[j] = %g\n", j1, M, Mj[j], S, Sj[j]);
                //        printf("        thread %d: result from thread %d for row %d: M = %g, Mj[j] = %g, S = %g, Sj[j] = %g\n", ith, ith_q*gcd_k + j1, j, M, Mj[j], S, Sj[j]);
                //        if (Mj[j] == -INFINITY) continue;
                //        if (Mj[j] > M) {
                //            float c = M > -INFINITY ? exp(M - Mj[j]) : 0;
                //            S = c*S + Sj[j];
                //            M = Mj[j];
                //            for (int i = 0; i < 16*nv; ++i) Racc[i] = c*Racc[i] + R[i];
                //        } else {
                //            float c = exp(Mj[j] - M);
                //            S += c*Sj[j];
                //            for (int i = 0; i < 16*nv; ++i) Racc[i] += c*R[i];
                //        }
                //    }
                //    if (!(S > 0)) printf("Oops: S = %g\n", S);
                //    float norm = S > 0 ? 1/S : 1;
                //    //if (S <= 0) {
                //    //    printf("Oops: S = %g\n", S);
                //    //}
                //    //float norm = 1/S;
                //    for (int i = 0; i < 16*nv; ++i) Racc[i] *= norm;
                //}
                //return true;
            }
        }
    }
    //if (neq3 == 1 && rk2%4 == 0 && neq1 == 1 && nth > 1 && nek1/32 > 1) {
    //    int nstep_q = rk2/4;
    //    if (ith < nstep_q) {
    //        thread_local std::vector<uint16_t> mask_copy;
    //        if (int(mask_copy.size()) < 4*Dk) {
    //            mask_copy.resize(4*Dk);
    //        }
    //        for (int i = 0; i < 4; ++i) std::memcpy(mask_copy.data() + i*Dk, mask, Dk*sizeof(uint16_t));
    //        for (int istep = ith; istep < nstep_q; istep += nth) {
    //            if (!iqk_flash_attn_impl(int_type_k, int_type_v,
    //                        Dk, Dv, 4, nek1, nbq2, stride_k, stride_v, Dk*sizeof(uint16_t), nb1/sizeof(float),
    //                        (const float *)((const char *)q + 4*istep*nbq2),
    //                        (const void  *)((const char *)k),
    //                        (const void  *)((const char *)v),
    //                        (const void  *)((const char *)mask_copy.data()),
    //                        scale, softcap,
    //                        (float *)((char *)qkv + 4*istep*nb1), nullptr, nullptr)) return false;
    //        }
    //    }
    //    return true;
    //    //int nstep_k = nek1/32;
    //    //int gcd_q = simple_gcd(nstep_q, nth);
    //    //int nth_q = nth/gcd_q;
    //    //int gcd_k = simple_gcd(nstep_k, nth_q);
    //    //if (ith == 0) printf("%s: neq2=%d nek2=%d nev2=%d nek1=%d gcd_k=%d gcd_q=%d\n", __func__, neq2, nek2, nev2, nek1, gcd_k, gcd_q);
    //    //////if (ith == 0) printf("%s: neq2=%d nek2=%d nev2=%d nek1=%d nstep_k=%d nstep_q=%d\n", __func__, neq2, nek2, nev2, nek1, nstep_k, nstep_q);
    //    ////int gcd_k = simple_gcd(nstep_k, nth);
    //    ////if (gcd_k > 1) {
    //    ////    int nth_k = nth/gcd_k;
    //    ////    int gcd_q = simple_gcd(nstep_q, nth_k);
    //    ////    if (ith == 0) printf("%s: neq2=%d nek2=%d nev2=%d nek1=%d gcd_k=%d gcd_q=%d\n", __func__, neq2, nek2, nev2, nek1, gcd_k, gcd_q);
    //    ////}
    //}

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

    //return iqk_flash_attn_impl(int_type_k, int_type_v, Dk, Dv, nq1, nk1, stride_q, stride_k, stride_v, stride_m,
    //        ne1*nb1/sizeof(float),
    //        q, k, v, mask, scale, softcap, qkv, nullptr, nullptr);
}

#else

bool iqk_flash_attn_noalibi([[maybe_unused]] int type_q, [[maybe_unused]] int type_mask, [[maybe_unused]] float max_bias,
                            [[maybe_unused]] int neq3, [[maybe_unused]] int neq2, [[maybe_unused]] long nbq3, [[maybe_unused]] long nbq2,
                            [[maybe_unused]] int nek3, [[maybe_unused]] int nek2, [[maybe_unused]] long nbk3, [[maybe_unused]] long nbk2,
                            [[maybe_unused]] int nev3, [[maybe_unused]] int nev2, [[maybe_unused]] long nbv3, [[maybe_unused]] long nbv2,
                            [[maybe_unused]] int ne2,  [[maybe_unused]] int ne1,  [[maybe_unused]] long nb1,
                            [[maybe_unused]] int int_type_k,         // type of k
                            [[maybe_unused]] int int_type_v,         // type of v
                            [[maybe_unused]] int D,                  // head size
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

