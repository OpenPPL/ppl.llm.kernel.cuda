// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except input compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to input writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/kernel/llm/cuda/pmx/i4f16/quantize.h"

#include "ppl/common/log.h"
#include "cudakernel/common/common.cuh"

#include "../quant_utils.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

template<int32_t group_size, int32_t TPB>
__global__
void minmax_scale_fp16_kernel(
    const fp16_t* input, // [N, K]
    const int32_t N,
    const int32_t K,
    fp16_t* scale_ptr // [N, K / 128]
) {
    // Firstly we take element from x in row major order for int4 quant
    // One Warp process one group:
    const int32_t tid = threadIdx.x;

    const int32_t k_block_id = blockIdx.x;
    const int32_t n_block_id = blockIdx.y;

    constexpr int32_t INT4_PACK_SIZE = 8;

    fp32_t local_x[INT4_PACK_SIZE];
    fp32_t local_max = 0.0f;

    for (int32_t i = 0; i < INT4_PACK_SIZE; i++) {
        local_x[i] = 0.0;
    }
    for (int32_t i = 0; i < INT4_PACK_SIZE; i++) {
        int32_t index = k_block_id * group_size + i + tid * INT4_PACK_SIZE;
        if(index < K) {
            local_x[i] = __half2float(input[index + n_block_id * K]);
            if(abs(local_x[i]) > local_max) {
                local_max = abs(local_x[i]);
            }
        }
    }
    // reduction inside a thread warp
    for (int32_t mask = TPB / 2; mask >= 1; mask /= 2) {
        local_max = max(__shfl_xor_sync(uint32_t(-1), local_max, mask), local_max);
    }
    // scale_ptr is stored as row major
    fp32_t scale = min_max_range_to_scale(local_max, QTHRESHOLD, INT4_QLEVEL);
    if(threadIdx.x == 0) {
        scale_ptr[k_block_id * N + n_block_id] = __float2half(scale);
    }
}

template<int32_t group_size>
__global__
void minmax_quantize_fp16_kernel(
    const fp16_t* input, // [N, K]
    const fp16_t* scale_ptr, // [N, K / 128]
    const int32_t N,
    const int32_t K,
    int32_t *output // [N / 8, K]
) {
    constexpr int32_t INT4_PACK_SIZE = 8;
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    fp16_t local_x[INT4_PACK_SIZE];
    fp16_t local_s[INT4_PACK_SIZE];

    if (tid * INT4_PACK_SIZE < N * K){
        # pragma unroll
        for(int32_t i = 0; i < INT4_PACK_SIZE; i++){
            const int32_t index_in_row_major = tid * INT4_PACK_SIZE + i;
            const int32_t index_in_col_major = (index_in_row_major % N) * K + index_in_row_major / N;
            local_x[i] = input[index_in_col_major];
            local_s[i] = scale_ptr[index_in_row_major % N + (index_in_row_major / N / 128) * N];
        }
    }

    // convert fp16 to int4
    int32_t packed_q_val = 0;

    # pragma unroll
    for(int32_t i = 0; i < INT4_PACK_SIZE; i++){
        int32_t temp = quant_scalar<fp32_t, fp32_t, false>(
            __half2float(local_x[i]), __half2float(local_s[i]), INT4_QMIN, INT4_QMAX);
        packed_q_val += (temp - INT4_QMIN) << (i * 4);
    }

    // output is stored as col major, transpose it in 16-bit order(not 32-bit).
    // output matrix [N, K] in 16 bit will behave like [2 * N, K]
    const int32_t pack_offset = (tid * INT4_PACK_SIZE) >> 2;
    const int32_t KK = K; const int32_t NN = N / 8 * 2;
    int32_t pack_offset_in_col_major[2] = {
        (pack_offset % NN) * KK + pack_offset / NN,
        ((pack_offset + 1) % NN) * KK + (pack_offset + 1) / NN
    };

    int16_t *packed_int4x8_16bit = reinterpret_cast<int16_t *>(&packed_q_val);
    int16_t *output_16bit        = reinterpret_cast<int16_t *>(output);

    output_16bit[pack_offset_in_col_major[0]] = packed_int4x8_16bit[0];
    output_16bit[pack_offset_in_col_major[1]] = packed_int4x8_16bit[1];
}

ppl::common::RetCode minmax_quantize_fp16(
    cudaStream_t stream,
    const void* input, // [N, K] fp16
    const int64_t N,
    const int64_t K, // must aligned 128 now
    const int64_t group_size,
    void* quantized, // [N / 8, K]
    void* scale  // [num_of_element / group_size]
)
{
    constexpr int32_t INT4_PACK_SIZE = 8;
    constexpr int32_t TPB = 256;

    const dim3 scale_block = {
        (unsigned int)((K + group_size - 1) / group_size),
        (unsigned int)N,
        1};
    const int32_t quantize_block = (N * K / INT4_PACK_SIZE + TPB - 1) / TPB;
    
    switch(group_size){
        case 128:
            minmax_scale_fp16_kernel<128, 128 / INT4_PACK_SIZE>
                <<<scale_block, 128 / INT4_PACK_SIZE, 0, stream>>>(
                    (fp16_t*)(input), N, K, (fp16_t*)(scale)
                );
            minmax_quantize_fp16_kernel<128>
                <<<quantize_block, TPB, 0, stream>>>(
                    (fp16_t*)(input), (fp16_t*)(scale), N, K, (int32_t*)(quantized)
                );
            break;
        default:
            LOG(ERROR) << "only support group_size == 128";
            return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

}}}}}}

