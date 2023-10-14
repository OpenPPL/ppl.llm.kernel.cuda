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

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

/*
Following code is coming from
https://github.com/openppl-public/ppq/blob/master/ppq/csrc/cuda/common.cuh
*/
constexpr int32_t ROUND_HALF_EVEN          = 0;
constexpr int32_t ROUND_HALF_UP            = 1;
constexpr int32_t ROUND_HALF_DOWN          = 2;
constexpr int32_t ROUND_HALF_TOWARDS_ZERO  = 3;
constexpr int32_t ROUND_HALF_FAR_FORM_ZERO = 4;
constexpr int32_t ROUND_TO_NEAR_INT        = 5;
constexpr int32_t ROUND_UP                 = 6;
constexpr int32_t ROUND_DOWN               = 7;

constexpr float MINIMUM_QUANT_SCALE = 1e-7; // 量化最小 scale，改了也会起飞
constexpr int32_t INT8_QMIN = -127;          // 量化最小值，改了就等着起飞
constexpr int32_t INT8_QMAX = +127;          // 量化最小值，改了就等着起飞

template<typename Dtype>
__device__ inline
Dtype CLIP(const Dtype v, const Dtype min, const Dtype max)
{
    if(v > max) return max;
    if(v < min) return min;
    return v;
}

template<int32_t rounding>
__device__ inline
int32_t ROUND2INT(const float value)
{
    if (false) { }
    else if (rounding == ROUND_HALF_EVEN) {
        return std::nearbyint(value);
    }
    else if (rounding == ROUND_HALF_UP) {
        return floor(value + .5);
    }
    else if (rounding == ROUND_HALF_DOWN) {
        return ceil(value - .5);
    }
    else if (rounding == ROUND_HALF_TOWARDS_ZERO) {
        if (value > 0)
            return ROUND2INT<ROUND_HALF_DOWN>(value);
        else
            return ROUND2INT<ROUND_HALF_UP>(value);
    }
    else if (rounding == ROUND_HALF_FAR_FORM_ZERO) {
        if (value > 0)
            return ROUND2INT<ROUND_HALF_UP>(value);
        else
            return ROUND2INT<ROUND_HALF_DOWN>(value);
    }
    else if (rounding == ROUND_UP) {
        return ceil(value);
    }
    else if (rounding == ROUND_DOWN) {
        return floor(value);
    }
    else {
        return round(value);
    }
}

/*
根据 scale 把 fp32 的数量化到 int8，这个东西似乎可以调用一个 __saturatef 指令加速运算，虽然它现在还没有
https://github.com/openppl-public/ppq/blob/master/ppq/csrc/cuda/common.cuh#L118
*/
__device__ inline
int8_t QUANT_FP32_TO_INT8(const float value, const float scale) {
    float _f_value = value / scale;
    int32_t _i_value = CLIP<int32_t>(
        __float2int_rn(_f_value),
        INT8_QMIN, INT8_QMAX
    );
    return (int8_t)(_i_value);
}

/*
根据 scale的倒数 把 fp32 的数量化到 int8，这个东西似乎可以调用一个 __saturatef 指令加速运算，虽然它现在还没有
https://github.com/openppl-public/ppq/blob/master/ppq/csrc/cuda/common.cuh#L118
*/
__device__ inline
int8_t QUANT_FP32_TO_INT8_RCP(const float value, const float r_scale) {
    float _f_value = value * r_scale;
    int32_t _i_value = CLIP<int32_t>(
        ROUND2INT<ROUND_HALF_EVEN>(_f_value),
        INT8_QMIN, INT8_QMAX
    );
    return (int8_t)(_i_value);
}

/*
根据 min-max 范围确定 scale，这玩意跟 ppq 里面的实现很像，但它更简单一些
https://github.com/openppl-public/ppq/blob/master/ppq/quantization/observer/range.py#L23
*/
__device__ inline
float MIN_MAX_RANGE_TO_SCALE(const float range)
{
    return max(range / 127.0f, MINIMUM_QUANT_SCALE);
}

/* 快速求倒数，调一条原生指令 */
__device__ inline
float RCP(const float value)
{
    return __frcp_rz(value);
}

template<int32_t WPT>
__device__ inline
float BLOCK_REDUCE_MAX(float reducing, float *shared_mem)
{
    // Helper function for reduce softmax qkmax.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

    for (int32_t mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    if (lane_id == 0) {
        shared_mem[warp_id] = reducing;
    }
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];
    else reducing = -FLT_MAX;

#pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

}}}}}

#endif
