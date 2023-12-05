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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_QUANT_UTILS_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_QUANT_UTILS_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

#include "type.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

const fp32_t  QTHRESHOLD = 1e-7;
const int32_t INT8_QMIN       = -127;
const int32_t INT8_QMAX       = 127;
const int32_t INT8_QLEVEL     = 127;
const int32_t INT4_QMIN       = -8;
const int32_t INT4_QMAX       = 7;
const int32_t INT4_QLEVEL     = 8;

__host__ __device__ inline
bool check_power_of_2(const int32_t x) {
    return (x > 0) && ((x & (x - 1)) == 0);
}

template<int32_t WRAPS_PER_THREAD_BLOCK>
inline __device__
fp32_t block_reduce_max(
    const fp32_t reducing,
    fp32_t *shared_mem
) {
    constexpr int32_t WRAP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WRAP_SIZE;
    const int32_t wrap_id = threadIdx.x / WRAP_SIZE;

    fp32_t reduced = reducing;

    // step 1. reduction in warp
#pragma unroll
    for (int32_t mask = WRAP_SIZE / 2; mask >= 1; mask /= 2) {
        reduced = max(__shfl_xor_sync(uint32_t(-1), reduced, mask), reduced);
    }

    // step 2. reduction in thread block (via shared mem)
    if (lane_id == 0) shared_mem[wrap_id] = reduced;
    __syncthreads();

    if (lane_id < WRAPS_PER_THREAD_BLOCK) {
        reduced = shared_mem[lane_id];
    }

#pragma unroll
    for (int32_t mask = WRAP_SIZE / 2; mask >= 1; mask /= 2) {
        reduced = max(__shfl_xor_sync(uint32_t(-1), reduced, mask), reduced);
    }

    // Broadcast to other threads.
    return __shfl_sync(uint32_t(-1), reduced, 0);
}

template<typename VType, typename SType, bool use_scale_reciprocal>
__device__ inline
int32_t quant_scalar(
    const VType value, const SType scale, 
    const int32_t clip_min, const int32_t clip_max
) {
    fp32_t _f_value;
    if (use_scale_reciprocal){
        _f_value = value * scale;
    }
    else{
        _f_value = value * (1 / scale);
    }
    int32_t _i_value = round(_f_value);
    _i_value = _i_value > clip_max ? clip_max : _i_value;
    _i_value = _i_value < clip_min ? clip_min : _i_value;
    return _i_value;
}

__device__ inline
fp32_t dequant_scalar(
    const int32_t value, 
    const fp32_t scale
) {
    return value * scale;
}

__device__ inline
fp32_t dequant_scalar(
    const int32_t value, 
    const fp16_t scale
) {
    return value * __half2float(scale);
}

__device__ inline
fp32_t min_max_range_to_scale(
    const fp32_t range, 
    const fp32_t scale_threshold, 
    const int32_t quant_level
) {
    return max(range / quant_level, scale_threshold);
}

__device__ inline
fp32_t min_max_range_to_scale(
    const fp32_t range,
    const int32_t quant_level
) {
    return max(range / quant_level, QTHRESHOLD);
}

}}}}}

#endif
