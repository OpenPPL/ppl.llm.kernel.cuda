#include "gemv.h"

#include <cuda_fp16.h>

#include "utils.h"


namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {


template <int ThreadNums>
__device__ __forceinline__ float sum(float val, volatile float* smem) {
    const int row = threadIdx.x / ThreadNums;
    const int col = threadIdx.x % ThreadNums;

    if constexpr (ThreadNums == 32) {
#pragma unroll
        for (int mask = ThreadNums >> 1; mask > 0; mask >>= 1) {
            val += __shfl_xor_sync(0xFFFFFFFF, val, mask, 32);
        }
        return val;
    }

    smem[row * ThreadNums + col] = val;
    __syncthreads();

#pragma unroll
    for (int mask = ThreadNums >> 1; mask >= 32; mask >>= 1) {
        if (col < mask) {
            smem[row * ThreadNums + col] += smem[row * ThreadNums + col + mask];
        }
        __syncthreads();
    }

    float sum = smem[row * ThreadNums + col];
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, mask, 32);
    }

    return sum;
}

template <int Threads, int TBatch, int AccessBatch, int BATCH>
__device__ __forceinline__ void compute(const half* ptr_b, const half* ptr_a[BATCH], const half* ptr_s, const int N,
                                        const int K, half (&accum_c)[BATCH][4], const int iter_nums,
                                        const int block_idx) {
    constexpr static int kTBatch = TBatch;
    constexpr static int kAccessBatch = AccessBatch;
    const int thread_x = threadIdx.x % TBatch;
    const int offset_x = thread_x * kAccessBatch;

    for (int it = 0; it < iter_nums; it += 8) {
        float4 reg_as[8][BATCH];
        float4 reg_b[8];
        float2 reg_s[8];

#pragma unroll
        for (int m = 0; m < 8; m++) {
#pragma unroll
            for (int r = 0; r < BATCH; r++) {
                reg_as[m][r] = {0.0, 0.0, 0.0, 0.0};
            }
            reg_b[m] = {0.0, 0.0, 0.0, 0.0};
            reg_s[m] = {0.0, 0.0};
        }

#pragma unroll
        for (int m = 0; m < 8; m++) {
            const int offset_iter = offset_x + (it + m) * kTBatch * kAccessBatch;
            const half* ptr_iter_b = ptr_b + (it + m) * kTBatch * kAccessBatch;
            const float4* ptr_f4_b = reinterpret_cast<const float4*>(ptr_iter_b);
            if (offset_iter < K) {
                reg_b[m] = __ldg(ptr_f4_b);
            }
            const int offset_s = (offset_iter / 128) * N * 4;
            const half* ptr_iter_s = ptr_s + offset_s;
            const float2* ptr_f2_s = reinterpret_cast<const float2*>(ptr_iter_s);
            if (offset_iter < K) {
                reg_s[m] = __ldg(ptr_f2_s);
            }
        }

#pragma unroll
        for (int b = 0; b < BATCH; b++) {
#pragma unroll
            for (int m = 0; m < 8; m++) {
                const int offset_iter = offset_x + (it + m) * kTBatch * kAccessBatch;
                const half* ptr_iter_a = ptr_a[b] + (it + m) * kTBatch * kAccessBatch;
                const float4* ptr_f4_a = reinterpret_cast<const float4*>(ptr_iter_a);
                if (offset_iter < K) {
                    reg_as[m][b] = __ldca(ptr_f4_a);
                }
            }
        }

        uint32_t reg_bu[8][4][4];
        uint32_t reg_ut[8][4][4];
#pragma unroll
        for (int m = 0; m < 8; m++) {
            uint32_t* reg_b_u32 = reinterpret_cast<uint32_t*>(&reg_b[m]);
#pragma unroll
            for (int p = 0; p < 4; p++) {
                dequantilize(reg_b_u32[p], reg_bu[m][p]);
            }

#pragma unroll
            for (int r = 0; r < 4; r++) {
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    reg_ut[m][r][c] = reg_bu[m][c][r];
                }
            }
        }

#pragma unroll
        for (int m = 0; m < 8; m++) {
#pragma unroll
            for (int p = 0; p < 4; p++) {
#pragma unroll
                for (int b = 0; b < BATCH; b++) {
                    const half* reg_sh = reinterpret_cast<const half*>(&reg_s[m]);
                    const half2* reg_ah = reinterpret_cast<const half2*>(&reg_as[m][b]);
                    const half2* reg_bh = reinterpret_cast<const half2*>(&reg_ut[m][p]);
                    half2 temp_sum = {0.0, 0.0};
#pragma unroll
                    for (int i = 0; i < (kAccessBatch >> 1); i++) {
                        temp_sum = __hfma2(reg_ah[i], reg_bh[i], temp_sum);
                    }
                    const half temp_s = __hadd(temp_sum.x, temp_sum.y);
                    const half temp_v = __hmul(temp_s, reg_sh[p]);
                    accum_c[b][p] = __hadd(temp_v, accum_c[b][p]);
                }
            }
        }
    }
}

template <int Threads, int TBatch, int BATCH>
__global__ __launch_bounds__(Threads) void gemv_basic(const void* __restrict__ A, const void* __restrict__ B,
                                                      const void* __restrict__ S, void* __restrict__ C, int N, int K) {
    const int block_idx = blockIdx.x * gridDim.y + blockIdx.y;

    constexpr static int kThreads = Threads;
    constexpr static int kTBatch = TBatch;
    constexpr static int kAccessBatch = 8;
    constexpr static int kK = kTBatch * kAccessBatch;
    constexpr static int kN = kThreads / kTBatch;
    constexpr static int kBatch = BATCH;

    const half* a_ptr = reinterpret_cast<const half*>(A);

    const int cta_n = (block_idx * kN) * K;
    const half* b_ptr = reinterpret_cast<const half*>(B) + cta_n;

    const int c_offset = (block_idx * kN + threadIdx.x / kTBatch) * 4;
    half* c_ptr[kBatch];
    float2* c_ptr_f4[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; b++) {
        c_ptr[b] = reinterpret_cast<half*>(C) + b * N * 4 + c_offset;
        c_ptr_f4[b] = reinterpret_cast<float2*>(c_ptr[b]);
    }

    const int thread_x = threadIdx.x % TBatch;
    const int thread_y = threadIdx.x / TBatch;
    const int offset_x = thread_x * kAccessBatch;
    const int offset_y = thread_y;
    const int offset_b = offset_y * K + offset_x;
    const half* ptr_b = b_ptr + offset_b;
    const half* ptr_a[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; b++) {
        ptr_a[b] = a_ptr + b * K + offset_x;
    }
    const half* ptr_s = reinterpret_cast<const half*>(S) + block_idx * kN * 4 + offset_y * 4;

    const bool is_write_thread = thread_x == 0;

    half accum_c[kBatch][4];

#pragma unroll
    for (int p = 0; p < 4; p++) {
#pragma unroll
        for (int b = 0; b < kBatch; b++) {
            accum_c[b][p] = {0.0};
        }
    }

    const int iter_nums = (K + kK - 1) / kK;
    compute<kThreads, kTBatch, kAccessBatch, kBatch>(ptr_b, ptr_a, ptr_s, N, K, accum_c, iter_nums, block_idx);

    __shared__ float smem[kThreads];

#pragma unroll
    for (int p = 0; p < 4; p++) {
#pragma unroll
        for (int b = 0; b < kBatch; b++) {
            const float val_sum = sum<kTBatch>(__half2float(accum_c[b][p]), smem);
            accum_c[b][p] = __float2half(val_sum);
        }
    }

    if (is_write_thread && c_offset < N * 4) {
        float2* accum_pack[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; b++) {
            accum_pack[b] = reinterpret_cast<float2*>(accum_c[b]);
        }

#pragma unroll
        for (int b = 0; b < kBatch; b++) {
            *(c_ptr_f4[b]) = *(accum_pack[b]);
        }
    }
}

template <int Threads, int TBatch, int BATCH>
__global__ __launch_bounds__(Threads) void gemv_bias(const void* __restrict__ A, const void* __restrict__ B,
                                                     const void* __restrict__ S, const void* __restrict__ BS,
                                                     void* __restrict__ C, int N, int K) {
    const int block_idx = blockIdx.x * gridDim.y + blockIdx.y;

    constexpr static int kThreads = Threads;
    constexpr static int kTBatch = TBatch;
    constexpr static int kAccessBatch = 8;
    constexpr static int kK = kTBatch * kAccessBatch;
    constexpr static int kN = kThreads / kTBatch;
    constexpr static int kBatch = BATCH;

    const half* a_ptr = reinterpret_cast<const half*>(A);

    const int cta_n = (block_idx * kN) * K;
    const half* b_ptr = reinterpret_cast<const half*>(B) + cta_n;

    const int c_offset = (block_idx * kN + threadIdx.x / kTBatch) * 4;
    half* c_ptr[kBatch];
    float2* c_ptr_f4[kBatch];
    float2 bias[kBatch];
    const half* bs_ptr[kBatch];
    const float2* bs_ptr_f4[kBatch];

#pragma unroll
    for (int b = 0; b < kBatch; b++) {
        c_ptr[b] = reinterpret_cast<half*>(C) + b * N * 4 + c_offset;
        c_ptr_f4[b] = reinterpret_cast<float2*>(c_ptr[b]);
        // bs_ptr[b] = reinterpret_cast<const half*>(BS) + b * N * 4 + c_offset;
        // bs_ptr_f4[b] = reinterpret_cast<const float2*>(bs_ptr[b]);
    }
    bs_ptr[0] = reinterpret_cast<const half*>(BS) + 0 * N * 4 + c_offset;
    bs_ptr_f4[0] = reinterpret_cast<const float2*>(bs_ptr[0]);

    const int thread_x = threadIdx.x % TBatch;
    const int thread_y = threadIdx.x / TBatch;
    const int offset_x = thread_x * kAccessBatch;
    const int offset_y = thread_y;
    const int offset_b = offset_y * K + offset_x;
    const half* ptr_b = b_ptr + offset_b;
    const half* ptr_a[kBatch];
#pragma unroll
    for (int b = 0; b < kBatch; b++) {
        ptr_a[b] = a_ptr + b * K + offset_x;
    }
    const half* ptr_s = reinterpret_cast<const half*>(S) + block_idx * kN * 4 + offset_y * 4;

    const bool is_write_thread = thread_x == 0;

    half accum_c[kBatch][4];

#pragma unroll
    for (int p = 0; p < 4; p++) {
#pragma unroll
        for (int b = 0; b < kBatch; b++) {
            accum_c[b][p] = {0.0};
        }
    }

    const int iter_nums = (K + kK - 1) / kK;
    compute<kThreads, kTBatch, kAccessBatch, kBatch>(ptr_b, ptr_a, ptr_s, N, K, accum_c, iter_nums, block_idx);

    if (is_write_thread && c_offset < N * 4) {
// #pragma unroll
//         for (int b = 0; b < kBatch; b++) {
//             bias[b] = *(bs_ptr_f4[b]);
//         }
        bias[0] = *(bs_ptr_f4[0]);
    }

    __shared__ float smem[kThreads];

#pragma unroll
    for (int p = 0; p < 4; p++) {
#pragma unroll
        for (int b = 0; b < kBatch; b++) {
            const float val_sum = sum<kTBatch>(__half2float(accum_c[b][p]), smem);
            accum_c[b][p] = __float2half(val_sum);
        }
    }

    if (is_write_thread && c_offset < N * 4) {
        float2* accum_pack[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; b++) {
            accum_pack[b] = reinterpret_cast<float2*>(accum_c[b]);
        }

        const half2* bs_h = reinterpret_cast<const half2*>(&bias[0]);
#pragma unroll
        for (int b = 0; b < kBatch; b++) {
            half2* accum_h = reinterpret_cast<half2*>(accum_pack[b]);
            // const half2* bs_h = reinterpret_cast<const half2*>(&bias[b]);
#pragma unroll
            for (int it = 0; it < 2; it++) {
                accum_h[it] = __hadd2(bs_h[it], accum_h[it]);
            }
        }

#pragma unroll
        for (int b = 0; b < kBatch; b++) {
            *(c_ptr_f4[b]) = *(accum_pack[b]);
        }
    }
}

template <int Threads, int TBatch, int BATCH>
struct Launcher {
    static void run(const void* __restrict__ A, const void* __restrict__ B, const void* __restrict__ S,
                    void* __restrict__ C, int M, int N, int K, cudaStream_t stream) {
        dim3 block(Threads, 1);
        int grid_nums = (N + (Threads / TBatch) - 1) / (Threads / TBatch);
        dim3 grid(grid_nums, 1);
        switch (M) {
            case 1:
                gemv_basic<Threads, TBatch, BATCH><<<grid, block, 0, stream>>>(A, B, S, C, N, K);
                break;
            case 2:
                gemv_basic<Threads, TBatch, BATCH><<<grid, block, 0, stream>>>(A, B, S, C, N, K);
                break;
            case 3:
                gemv_basic<Threads, TBatch, BATCH><<<grid, block, 0, stream>>>(A, B, S, C, N, K);
                break;
            case 4:
                gemv_basic<Threads, TBatch, BATCH><<<grid, block, 0, stream>>>(A, B, S, C, N, K);
                break;
        }
    }

    static void run_bias(const void* __restrict__ A, const void* __restrict__ B, const void* __restrict__ S,
                         const void* __restrict__ BS, void* __restrict__ C, int M, int N, int K,
                         cudaStream_t stream) {
        dim3 block(Threads, 1);
        int grid_nums = (N + (Threads / TBatch) - 1) / (Threads / TBatch);
        dim3 grid(grid_nums, 1);
        switch (M) {
            case 1:
                gemv_bias<Threads, TBatch, BATCH><<<grid, block, 0, stream>>>(A, B, S, BS, C, N, K);
                break;
            case 2:
                gemv_bias<Threads, TBatch, BATCH><<<grid, block, 0, stream>>>(A, B, S, BS, C, N, K);
                break;
            case 3:
                gemv_bias<Threads, TBatch, BATCH><<<grid, block, 0, stream>>>(A, B, S, BS, C, N, K);
                break;
            case 4:
                gemv_bias<Threads, TBatch, BATCH><<<grid, block, 0, stream>>>(A, B, S, BS, C, N, K);
                break;
        }
    }
};

void Gemv::run(const void* __restrict__ A, const void* __restrict__ B, const void* __restrict__ S, void* __restrict__ C,
               int M, int N, int K, cudaStream_t stream) const {
    if (K > (N * 4) * 1.2) {
        switch (M) {
            case 1:
                Launcher<128, 64, 1>::run(A, B, S, C, M, N, K, stream);
                break;
            case 2:
                Launcher<128, 64, 2>::run(A, B, S, C, M, N, K, stream);
                break;
            case 3:
                Launcher<128, 64, 3>::run(A, B, S, C, M, N, K, stream);
                break;
            case 4:
                Launcher<128, 64, 4>::run(A, B, S, C, M, N, K, stream);
                break;
            default:
                break;
        }
    } else {
        switch (M) {
            case 1:
                Launcher<128, 32, 1>::run(A, B, S, C, M, N, K, stream);
                break;
            case 2:
                Launcher<128, 32, 2>::run(A, B, S, C, M, N, K, stream);
                break;
            case 3:
                Launcher<128, 32, 3>::run(A, B, S, C, M, N, K, stream);
                break;
            case 4:
                Launcher<128, 32, 4>::run(A, B, S, C, M, N, K, stream);
                break;
            default:
                break;
        }
    }
}

void Gemv::run_bias(const void* __restrict__ A, const void* __restrict__ B, const void* __restrict__ S,
                    const void* __restrict__ BS, void* __restrict__ C, int M, int N, int K, cudaStream_t stream) const {
    if (K > (N * 4) * 1.2) {
        switch (M) {
            case 1:
                Launcher<128, 64, 1>::run_bias(A, B, S, BS, C, M, N, K, stream);
                break;
            case 2:
                Launcher<128, 64, 2>::run_bias(A, B, S, BS, C, M, N, K, stream);
                break;
            case 3:
                Launcher<128, 64, 3>::run_bias(A, B, S, BS, C, M, N, K, stream);
                break;
            case 4:
                Launcher<128, 64, 4>::run_bias(A, B, S, BS, C, M, N, K, stream);
                break;
            default:
                break;
        }
    } else {
        switch (M) {
            case 1:
                Launcher<128, 32, 1>::run_bias(A, B, S, BS, C, M, N, K, stream);
                break;
            case 2:
                Launcher<128, 32, 2>::run_bias(A, B, S, BS, C, M, N, K, stream);
                break;
            case 3:
                Launcher<128, 32, 3>::run_bias(A, B, S, BS, C, M, N, K, stream);
                break;
            case 4:
                Launcher<128, 32, 4>::run_bias(A, B, S, BS, C, M, N, K, stream);
                break;
            default:
                break;
        }
    }
}


}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16
