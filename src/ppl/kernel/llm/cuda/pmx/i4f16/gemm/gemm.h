// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <cuda_fp16.h>

#include "gemm_iters.h"
#include "shape.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

template <typename BlockShape_, typename WarpShape_, int Stages, bool SplitK, bool Bias, int GroupSize>
struct Gemm {
    using BlockShape = BlockShape_;
    using WarpShape = WarpShape_;
    using MmaShape = GemmShape<16, 8, 16>;
    using IterShape =
        GemmShape<WarpShape::kM / MmaShape::kM, WarpShape::kN / MmaShape::kN, WarpShape::kK / MmaShape::kK>;

    using Mma = DefaultMma<BlockShape, WarpShape, MmaShape, Stages, SplitK, Bias, GroupSize>;

    static_assert(WarpShape::kM % Mma::MmaShape::kM == 0);
    static_assert(WarpShape::kN % Mma::MmaShape::kN == 0);
    static_assert(WarpShape::kK % Mma::MmaShape::kK == 0);
    static_assert(Mma::kSliceK % WarpShape::kK == 0);
    static constexpr int kWarpSize = Mma::kWarpSize;

    using RegTypeA = Array<half, 8>;
    using RegTypeB = Array<half, 4>;
    using RegTypeS = half;

    using IteratorA = typename Mma::IteratorA;
    using IteratorB = typename Mma::IteratorB;
    using IteratorS = typename Mma::IteratorS;
    using BatchShape = GemmShape<(IteratorA::IterCount::kCount + IterShape::kK - 1) / IterShape::kK,
                                 (IteratorB::IterCount::kCount + IterShape::kK - 1) / IterShape::kK,
                                 (IteratorS::IterCount::kCount + IterShape::kK - 1) / IterShape::kK>;

    using WarpIterA = typename Mma::WarpIterA;
    using WarpIterB = typename Mma::WarpIterB;
    using WarpIterS = typename Mma::WarpIterS;

    RegTypeA reg_a_[2][IterShape::kM];
    RegTypeB reg_b_[2][IterShape::kN];
    RegTypeS reg_s_[(IterShape::kN) * 8];
    float reg_u_c_[(IterShape::kN) * (IterShape::kM) * 4][4];
    float reg_s_c_[(IterShape::kN) * (IterShape::kM) * 4][4];

    mutable int warp_idx_{};
    mutable int lane_idx_{};
    mutable bool is_store_thread_{};

    template <typename T, int N, int M>
    __device__ static void clear(T (&dst)[N][M]) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
#pragma unroll
            for (int j = 0; j < M; ++j) {
                dst[i][j] = T{};
            }
        }
    }

    __device__ void sync_slice(int partial_idx) {
        if constexpr (Mma::kSlices == 1) {
            __syncthreads();
        } else {
            constexpr int kGroups = (Mma::kSlices + 7) / 8;
            constexpr uint32_t num_threads = Mma::WarpCount::kMN * kWarpSize;
            const uint32_t barrier_id = partial_idx / kGroups + 1;
            asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "n"(num_threads));
        }
    }

    __inline__ __device__ void store(const float4& data, int m, int n, half* C, int M, int N) {
        float4* c_f4 = reinterpret_cast<float4*>(C);
        const bool guard = ((n) < (N) && m < M);
        global_store(data, (void*)(&c_f4[(m * N + n) / 2]), guard);
    }

    __device__ void operator()(void* __restrict__ C, void* __restrict__ temp_C, const void* __restrict__ A,
                               const void* __restrict__ B, const void* __restrict__ S, const int M, const int N,
                               const int K, const int batchs, const int k_nums, const void* __restrict__ bias) {
        float* reg_u_c_ptr = reinterpret_cast<float*>(reg_u_c_);
        float* reg_s_c_ptr = reinterpret_cast<float*>(reg_s_c_);

        extern __shared__ uint8_t smem[];

        const int warp_idx = threadIdx.x / kWarpSize;
        const int lane_idx = threadIdx.x % kWarpSize;

        const int warp_idx_m = warp_idx % Mma::WarpCount::kM;
        const int warp_idx_nk = warp_idx / Mma::WarpCount::kM;
        const int warp_idx_n = warp_idx_nk % Mma::WarpCount::kN;
        const int warp_idx_k = warp_idx / (Mma::WarpCount::kM * Mma::WarpCount::kN);
        const int warp_idx_mn = warp_idx % (Mma::WarpCount::kM * Mma::WarpCount::kN);

        const int partial_idx = warp_idx_k;

        warp_idx_ = threadIdx.x / kWarpSize;
        lane_idx_ = threadIdx.x % kWarpSize;
        is_store_thread_ = warp_idx_k == 0 ? true : false;

        const int split_cta_k = batchs * blockIdx.z;
        const int cta_k = partial_idx * Mma::kSliceK + split_cta_k;
        const int cta_m = blockIdx.y * BlockShape::kM;
        const int cta_n = blockIdx.x * BlockShape::kN;

        float* const cache_c = reinterpret_cast<float*>(smem);
        half* const cache_a = reinterpret_cast<half*>(smem + IteratorA::kSmemByteSize * partial_idx);
        half* const cache_b = reinterpret_cast<half*>(smem + IteratorA::kSmemByteSize * Mma::kSlices +
                                                      IteratorB::kSmemByteSize * partial_idx);
        half* const cache_s =
            reinterpret_cast<half*>(smem + IteratorA::kSmemByteSize * Mma::kSlices +
                                    IteratorB::kSmemByteSize * Mma::kSlices + IteratorS::kSmemByteSize * partial_idx);

        IteratorA iterator_A{(const half*)A, smem_cast(cache_a), K, M, cta_m, cta_k, warp_idx_mn, lane_idx};
        IteratorB iterator_B{(const half*)B, smem_cast(cache_b), K, N, cta_n, cta_k, warp_idx_mn, lane_idx};
        IteratorS iterator_S{(const half*)S, smem_cast(cache_s), N * 4, K, cta_n * 4, cta_k, warp_idx_mn, lane_idx};
        int gemm_iterator = (k_nums + BlockShape::kK - 1) / BlockShape::kK;

#pragma unroll
        for (int stage = 0; stage < Stages - 1; ++stage, --gemm_iterator) {
            iterator_B.advance(gemm_iterator > 0);
            iterator_S.advance(gemm_iterator > 0);
            iterator_A.advance(gemm_iterator > 0);
            cp_async_fence();
        }

        WarpIterA warp_iterator_A(iterator_A.smem_u32_, warp_idx_m, lane_idx, 0);
        WarpIterB warp_iterator_B(iterator_B.smem_u32_, warp_idx_n, lane_idx, 0);
        WarpIterS warp_iterator_S(iterator_S.smem_u32_, warp_idx_n, lane_idx, 0);

        clear(reg_u_c_);

        cp_async_wait<Stages - 2>();
        sync_slice(partial_idx);

        warp_iterator_A.load(reg_a_[0], 0);
        warp_iterator_B.load(reg_b_[0], 0);

        PRAGMA_NO_UNROLL
        for (; gemm_iterator > Mma::kIter;) {
            warp_iterator_S.load_scale(0, reg_s_);
            mma(iterator_A, iterator_B, iterator_S, warp_iterator_A, warp_iterator_B, warp_iterator_S, reg_u_c_ptr,
                partial_idx, gemm_iterator);
        }

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        if constexpr (Mma::kSlices > 1) {
            reduce_sum(reg_u_c_ptr, cache_c, warp_idx_m, warp_idx_n, partial_idx);
        }
        post_process(reg_u_c_ptr, reg_s_c_ptr);

        if constexpr (!Mma::kSplitK) {
            store_result(reg_s_c_ptr, bias, cache_c, (half*)C, M, N, cta_m, cta_n, warp_idx_m, warp_idx_n);
        } else {
            if (gridDim.z > 1) {
                store_result(reg_s_c_ptr, bias, cache_c, (half*)temp_C, M, N, cta_m, cta_n, warp_idx_m, warp_idx_n);
            } else {
                store_result(reg_s_c_ptr, bias, cache_c, (half*)C, M, N, cta_m, cta_n, warp_idx_m, warp_idx_n);
            }
        }
    }

    __device__ void mma(IteratorA& iterator_A, IteratorB& iterator_B, IteratorS& iterator_S, WarpIterA& warp_iterator_A,
                        WarpIterB& warp_iterator_B, WarpIterS& warp_iterator_S, float* accum_unpack, int partial_idx,
                        int& gemm_iterator) {
        auto frag_unpack_C_ptr = reinterpret_cast<Array<float, 4>(*)[4]>(accum_unpack);
        RegTypeB frag_unpack_bs[2][IterShape::kN][4];

        float scale_float[IterShape::kN * 8];
        float temp_c[IterShape::kM][IterShape::kN][4][4];

#pragma unroll
        for (int iter_m = 0; iter_m < IterShape::kM; ++iter_m) {
#pragma unroll
            for (int iter_n = 0; iter_n < IterShape::kN; ++iter_n) {
#pragma unroll
                for (int inner = 0; inner < 4; inner++) {
                    temp_c[iter_m][iter_n][inner][0] = 0;
                    temp_c[iter_m][iter_n][inner][1] = 0;
                    temp_c[iter_m][iter_n][inner][2] = 0;
                    temp_c[iter_m][iter_n][inner][3] = 0;
                }
            }
        }

#pragma unroll
        for (int iter_k = 0; iter_k < IterShape::kK; ++iter_k) {
            if (iter_k <= IterShape::kK - 1) {
                iterator_A.next(iter_k, BatchShape::kM, gemm_iterator > 0);
                iterator_B.next(iter_k, BatchShape::kN, gemm_iterator > 0);
                iterator_S.next(iter_k, BatchShape::kK, gemm_iterator > 0);
            }

            if (iter_k == IterShape::kK - 1) {
                cp_async_fence();
                cp_async_wait<Stages - 2>();
                sync_slice(partial_idx);

                iterator_A.nexts();
                iterator_B.nexts();
                iterator_S.nexts();

                warp_iterator_A.nexts();
                warp_iterator_B.nexts();
                warp_iterator_S.nexts();
                --gemm_iterator;
            }

            warp_iterator_A.load(reg_a_[(iter_k + 1) % 2], (iter_k + 1) % IterShape::kK);
            warp_iterator_B.load(reg_b_[(iter_k + 1) % 2], (iter_k + 1) % IterShape::kK);

            auto& frag_unpack_b = frag_unpack_bs[iter_k % 2];
            unpack(reg_b_[iter_k % 2], frag_unpack_b);

#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int iter_n = 0; iter_n < IterShape::kN; ++iter_n) {
#pragma unroll
                    for (int iter_m = 0; iter_m < IterShape::kM; ++iter_m) {
                        auto& frag_a = reg_a_[iter_k % 2][iter_m];
                        auto& frag_b = frag_unpack_b[iter_n];
                        uint32_t* reg_a = reinterpret_cast<uint32_t*>(&frag_a);
                        uint32_t* reg_b = reinterpret_cast<uint32_t*>(frag_b);
                        mma_s16816_half(reg_a[0], reg_a[1], reg_a[2], reg_a[3], reg_b[i], reg_b[4 + i],
                                        temp_c[iter_m][iter_n][i][0], temp_c[iter_m][iter_n][i][1],
                                        temp_c[iter_m][iter_n][i][2], temp_c[iter_m][iter_n][i][3]);
                    }
                }
            }
        }

#pragma unroll
        for (int iter_n = 0; iter_n < IterShape::kN; ++iter_n) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                scale_float[iter_n * 8 + i] = __half2float(reg_s_[iter_n * 8 + i]);
            }
        }

#pragma unroll
        for (int iter_m = 0; iter_m < IterShape::kM; ++iter_m) {
#pragma unroll
            for (int iter_n = 0; iter_n < IterShape::kN; ++iter_n) {
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    auto& frag_c = frag_unpack_C_ptr[iter_m * IterShape::kN + iter_n][i];
                    float* reg_c = reinterpret_cast<float*>(&frag_c);
                    reg_c[0] += temp_c[iter_m][iter_n][i][0] * scale_float[iter_n * 8 + i];
                    reg_c[1] += temp_c[iter_m][iter_n][i][1] * scale_float[iter_n * 8 + i + 4];
                    reg_c[2] += temp_c[iter_m][iter_n][i][2] * scale_float[iter_n * 8 + i];
                    reg_c[3] += temp_c[iter_m][iter_n][i][3] * scale_float[iter_n * 8 + i + 4];
                }
            }
        }
    }

    __device__ void unpack(const RegTypeB* warp_frag_b, RegTypeB (*frag_unpack_b)[4]) {
#pragma unroll
        for (int iter_n = 0; iter_n < IterShape::kN; iter_n++) {
            const RegTypeB& b_raw = warp_frag_b[iter_n];
            const uint32_t* b = reinterpret_cast<const uint32_t*>(&b_raw);
            uint32_t(*u)[4] = reinterpret_cast<uint32_t(*)[4]>(frag_unpack_b[iter_n]);
#pragma unroll
            for (int i = 0; i < 2; i++) {
                dequantilize(b[i], u[i]);
            }
        }
    }

    __device__ void reduce_sum(float* reg_c, float* cache_c, const int warp_idx_m, const int warp_idx_n,
                               const int partial_idx) {
        float4* reg_c_f4 = reinterpret_cast<float4*>(reg_c);
        float4* cache_c_f4 = reinterpret_cast<float4*>(cache_c);
        const int offset_m = warp_idx_m * IterShape::kM;
        const int offset_n = warp_idx_n * IterShape::kN;

#pragma unroll
        for (int z = 0; z < Mma::kSlices; ++z) {
            if (partial_idx == z) {
#pragma unroll
                for (int i = 0; i < IterShape::kM; ++i) {
#pragma unroll
                    for (int j = 0; j < IterShape::kN; ++j) {
#pragma unroll
                        for (int x = 0; x < 4; ++x) {
                            const int src = (i * IterShape::kN + j) * 4 + x;
                            const int dst =
                                ((i + offset_m) * BlockShape::kN / Mma::MmaShape::kN + j + offset_n) * 4 + x;
                            if (z > 0) {
                                float* frag_c = reinterpret_cast<float*>(&reg_c_f4[src]);
                                float* ele = reinterpret_cast<float*>(&cache_c_f4[dst * kWarpSize + lane_idx_]);
#pragma unroll
                                for (int e = 0; e < 4; e++) {
                                    frag_c[e] = ele[e] + frag_c[e];
                                }
                            }
                            cache_c_f4[dst * kWarpSize + lane_idx_] = reg_c_f4[src];
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (partial_idx == 0) {
#pragma unroll
            for (int i = 0; i < IterShape::kM; ++i) {
#pragma unroll
                for (int j = 0; j < IterShape::kN; ++j) {
#pragma unroll
                    for (int x = 0; x < 4; ++x) {
                        int src = ((i + offset_m) * BlockShape::kN / Mma::MmaShape::kN + j + offset_n) * 4 + x;
                        int dst = (i * IterShape::kN + j) * 4 + x;
                        reg_c_f4[dst] = cache_c_f4[src * kWarpSize + lane_idx_];
                    }
                }
            }
        }
    }

    __device__ void post_process(const float* reg_c, float* post_frag_c) {
        if (!is_store_thread_) {
            return;
        }

#pragma unroll
        for (int i = 0; i < IterShape::kM; ++i) {
#pragma unroll
            for (int j = 0; j < IterShape::kN; ++j) {
                float temp[2][8];
#pragma unroll
                for (int x = 0; x < 2; ++x) {
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        temp[x][k] = reg_c[i * IterShape::kN * 16 + j * 16 + k / 2 * 4 + k % 2 + x * 2];
                    }
                }

                float post[2][8];
#pragma unroll
                for (int x = 0; x < 2; ++x) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        post[x][k] = temp[x][k * 2];
                    }
#pragma unroll
                    for (int k = 4; k < 8; k++) {
                        post[x][k] = temp[x][(k - 4) * 2 + 1];
                    }
                }

#pragma unroll
                for (int x = 0; x < 2; ++x) {
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        post_frag_c[i * IterShape::kN * 16 + j * 16 + k / 2 * 4 + k % 2 + x * 2] = post[x][k];
                    }
                }
            }
        }
    }

    __device__ void loadb(const void* bias, int m, int n, int M, int N, uint4& data) {
        const float4* bias_f4 = reinterpret_cast<const float4*>(bias);
        const bool guard = ((n) < (N) && m < M);
        const void* addr = (const void*)(&bias_f4[(m * N + n) / 2]);
        loadg(data.x, data.y, data.z, data.w, addr, guard);
    }

    __device__ void store_result(const float* reg_c, const void* bias, float* cache_c, half* C, int m, int n, int cta_m,
                                 int cta_n, int warp_idx_m, int warp_idx_n) {
        if (!is_store_thread_) {
            return;
        }

        float4 temp_b[IterShape::kM][2][IterShape::kN];
        if constexpr (Mma::kBias) {
            if (blockIdx.z == 0) {
#pragma unroll
                for (int i = 0; i < IterShape::kM; ++i) {
                    const int mm = cta_m + warp_idx_m * WarpShape::kM + i * Mma::MmaShape::kM + lane_idx_ / 4;
#pragma unroll
                    for (int x = 0; x < 2; ++x) {
#pragma unroll
                        for (int j = 0; j < IterShape::kN; ++j) {
                            const int nn =
                                cta_n + warp_idx_n * WarpShape::kN + j * Mma::MmaShape::kN + lane_idx_ % 4 * 2;
                            float4& frag_b_f4 = temp_b[i][x][j];
                            loadb(bias, (mm + x * 8), nn, m, n, reinterpret_cast<uint4&>(frag_b_f4));
                        }
                    }
                }
            }
        }

        float4 temp_c[IterShape::kM][2][IterShape::kN];

#pragma unroll
        for (int i = 0; i < IterShape::kM; ++i) {
            const float2* frag_c = (float2*)&reg_c[i * IterShape::kN * 4 * 4];
#pragma unroll
            for (int x = 0; x < 2; ++x) {
#pragma unroll
                for (int j = 0; j < IterShape::kN; ++j) {
                    float4& frag_c_f4 = temp_c[i][x][j];
                    half2* half_c = reinterpret_cast<half2*>(&frag_c_f4);
                    if constexpr (Mma::kBias) {
                        float4& frag_b_f4 = temp_b[i][x][j];
                        half2* half_b = reinterpret_cast<half2*>(&frag_b_f4);
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            half2 temp = __float22half2_rn(frag_c[j * 8 + e * 2 + x]);
                            if (blockIdx.z == 0) {
                                half_c[e] = temp + half_b[e];
                            } else {
                                half_c[e] = temp;
                            }
                        }
                    } else {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            half_c[e] = __float22half2_rn(frag_c[j * 8 + e * 2 + x]);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int i = 0; i < IterShape::kM; ++i) {
            const int mm = cta_m + warp_idx_m * WarpShape::kM + i * Mma::MmaShape::kM + lane_idx_ / 4;
#pragma unroll
            for (int x = 0; x < 2; ++x) {
#pragma unroll
                for (int j = 0; j < IterShape::kN; ++j) {
                    const int nn = cta_n + warp_idx_n * WarpShape::kN + j * Mma::MmaShape::kN + lane_idx_ % 4 * 2;
                    float4& frag_c_f4 = temp_c[i][x][j];
                    store(reinterpret_cast<float4&>(frag_c_f4), (mm + x * 8), nn, C, m, n);
                }
            }
        }
    }
};

template <typename Gemm>
__global__ __launch_bounds__(Gemm::Mma::kThreads) void gemm_kernel(void* __restrict__ C, const void* __restrict__ A,
                                                                   const void* __restrict__ B,
                                                                   const void* __restrict__ S, int M, int N, int K,
                                                                   int batchs, void* __restrict__ workspace,
                                                                   const void* __restrict__ bias = nullptr) {
    Gemm gemm;
    const int k_base = blockIdx.z * batchs;
    const int k_nums = (blockIdx.z == (gridDim.z - 1)) ? K - k_base : batchs;

    half* temp_c_base = reinterpret_cast<half*>(workspace);
    half* temp_c = temp_c_base + M * N * 4 * blockIdx.z;
    gemm((void*)C, (void*)temp_c, (void*)A, (void*)B, (void*)S, M, N, K, batchs, k_nums, bias);
}

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16