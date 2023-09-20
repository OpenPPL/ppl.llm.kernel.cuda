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

# ifndef __PPL_KERNEL_LLM_CUDA_QUANT_H__
# define __PPL_KERNEL_LLM_CUDA_QUANT_H__

# define __HOST_DEVICE_FUNCTION__ __host__ __device__
# include "ppl/kernel/llm/cuda/common/general_include.h"
# include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace quant {

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

// type define for quant
using fp16_t   = half;
using fp16x2_t = half2;

using fp32_t   = float;
using fp32x2_t = float2;
using fp32x4_t = float4;

using int16_t  = short int;

using int8x2_t = char2;
using int8x4_t = char4;

using uint8x2_t = uchar2;
using uint8x4_t = uchar4;

constexpr fp32_t MINIMUM_QUANT_SCALE = 1e-7; // 量化最小 scale，改了也会起飞
constexpr int32_t INT8_QMIN = -127;          // 量化最小值，改了就等着起飞
constexpr int32_t INT8_QMAX = +127;          // 量化最小值，改了就等着起飞

template<typename Dtype>
__HOST_DEVICE_FUNCTION__ inline
Dtype CLIP(const Dtype v, const Dtype min, const Dtype max){
    if(v > max) return max;
    if(v < min) return min;
    return v;
}

__HOST_DEVICE_FUNCTION__ inline
int _round2int(
    const float value,
    const int rounding
){
    switch(rounding){
        case ROUND_HALF_EVEN:
            return std::nearbyint(value);
        case ROUND_HALF_UP:
            return floor(value + .5);
        case ROUND_HALF_DOWN:
            return ceil(value - .5);
        case ROUND_HALF_TOWARDS_ZERO:
            if (value > 0) return _round2int(value, ROUND_HALF_DOWN);
            else return _round2int(value, ROUND_HALF_UP);
        case ROUND_HALF_FAR_FORM_ZERO:
            if (value > 0) return _round2int(value, ROUND_HALF_UP);
            else return _round2int(value, ROUND_HALF_DOWN);
        case ROUND_UP:
            return ceil(value);
        case ROUND_DOWN:
            return floor(value);
        default:
            return round(value);
    }
    return 0;
}

union FPConvertHelper {
    float value;
    uint32_t data;
};

template<typename Dtype, typename Stype, typename Otype>
__HOST_DEVICE_FUNCTION__ inline
float QuantizeScalarFloating(
    const Dtype value, const Stype scale, const Otype offset,
    const int exponent, const int mantissa,
    const float clip_min, const float clip_max, 
    const Rounding rounding){
    /**
     * PPQ Quantization Function implementation.
     * This function convert an float value to low-precision float
     */
    FPConvertHelper helper; FPConvertHelper rounding_helper;
    float Unscaled_FP32 = static_cast<float>(value) / scale;
    
    helper.value = Unscaled_FP32;
	int32_t exponent_min  = -(1 << (exponent - 1)) + 1;
    int32_t exponent_max  = (1 << (exponent - 1));

    // Following code will process exponent overflow
    /* For FP8 E4M3, the maximum exponent value should be 8.                                  */
    /* The Maximum number of FP8 E4M3 should be (0 1111 111) = 480                            */
    /* We call it as theoretical_maximum, FP8 E4M3 can not represent a number larger than it. */
    uint32_t fp32_sign    = 0;
    int32_t fp32_exp      = (exponent_max + 127) << 23;
    int32_t fp32_mantissa = ~(0x007FFFFF >> mantissa) & 0x007FFFFF;
    helper.data = fp32_sign + fp32_mantissa + fp32_exp;
    float theoretical_maximum = helper.value;

    if (Unscaled_FP32 > min(clip_max, theoretical_maximum)) 
        return min(clip_max, theoretical_maximum);
    if (Unscaled_FP32 < max(clip_min, -theoretical_maximum)) 
        return max(clip_min, -theoretical_maximum);

    // Code start from here will convert number within fp8 range.
    // Following code will Split float32 into sign, exp, mantissa
    /* IEEE 754 Standard: 1 bit sign, 8 bit exponent, 23 bit mantissa */

    /* In binary 10000000 00000000 00000000 00000000 = 0x80000000 in Hex */
    /* In binary 01111111 10000000 00000000 00000000 = 0x7F800000 in Hex */
    /* In binary 00000000 01111111 11111111 11111111 = 0x007FFFFF in Hex */

    /* Tool: https://www.h-schmidt.net/FloatConverter/IEEE754.html */
    helper.value  = Unscaled_FP32;
    fp32_sign     = helper.data & 0x80000000;
    fp32_exp      = helper.data & 0x7F800000;
    fp32_mantissa = helper.data & 0x007FFFFF;

    // Following code will process exponent underflow
    /* Float underflow means fp32_exp is smaller than exponent_min          */
    /* Where exponent_min is the minimum exponent value of quantized float. */
    /* For FP8 E4M3, the minimum exponent value should be -7.               */
    /* The Min Subnormal value of FP8 E4M3 should be (0 0000 001) = 2^-9    */
    /* The Min normal value of FP8 E4M3 should be (0 0001 000) = 2^-6       */
	if (((fp32_exp >> 23) - 127) < exponent_min + 1){
        // following divide might have some problems
        // but it is the simplest method with very limited error.
        float min_subnormal = 1.0f / (1 << ((1 << (exponent - 1)) + mantissa - 2));
        return _round2int(Unscaled_FP32 / min_subnormal, rounding) * min_subnormal;
	}

    /* high precision mantissa convert to low precision mantissa requires rounding                         */
    /* Here we apply a tricky method to round mantissa:                                                    */
    /* We create another float, which sign = 0, exponent = 127, mantissa = fp32_mantissa << (23 - mantissa) */
    /* Then we directly round this float to int, result here is what we want, you can prove it by yourself */
    rounding_helper.data = ((fp32_mantissa << (mantissa)) & 0x007FFFFF) + 0x3F800000;
    uint32_t round_bit = _round2int(rounding_helper.value - 1, rounding);

    // process mantissa
    fp32_mantissa = ((fp32_mantissa >> (23 - mantissa)) + round_bit) << (23 - mantissa);
    helper.data = fp32_sign + fp32_mantissa + fp32_exp;

    return CLIP<float>(helper.value, clip_min, clip_max);

}

/*
根据 scale 把 fp32 的数量化到 int8，这个东西似乎可以调用一个 __saturatef 指令加速运算，虽然它现在还没有
https://github.com/openppl-public/ppq/blob/master/ppq/csrc/cuda/common.cuh#L118
*/
__HOST_DEVICE_FUNCTION__ inline
int8_t QUANT_FP32_TO_INT8(const fp32_t value, const fp32_t scale) {
    fp32_t _f_value = value / scale;
    int32_t _i_value = CLIP<int32_t>(
        _round2int(_f_value, ROUND_HALF_EVEN),
        INT8_QMIN, INT8_QMAX
    );
    return (int8_t)(_i_value);
}

/*
根据 scale的倒数 把 fp32 的数量化到 int8，这个东西似乎可以调用一个 __saturatef 指令加速运算，虽然它现在还没有
https://github.com/openppl-public/ppq/blob/master/ppq/csrc/cuda/common.cuh#L118
*/
__HOST_DEVICE_FUNCTION__ inline
int8_t QUANT_FP32_TO_INT8_RCP(const fp32_t value, const fp32_t r_scale) {
    fp32_t _f_value = value * r_scale;
    int32_t _i_value = CLIP<int32_t>(
        _round2int(_f_value, ROUND_HALF_EVEN),
        INT8_QMIN, INT8_QMAX
    );
    return (int8_t)(_i_value);
}

/*
根据 min-max 范围确定 scale，这玩意跟 ppq 里面的实现很像，但它更简单一些
https://github.com/openppl-public/ppq/blob/master/ppq/quantization/observer/range.py#L23
*/
__HOST_DEVICE_FUNCTION__ inline
fp32_t MIN_MAX_RANGE_TO_SCALE(const fp32_t range){
    assert(range > 0);
    return max(range / 127, MINIMUM_QUANT_SCALE);
}

/* 快速求倒数，调一条原生指令 */
__HOST_DEVICE_FUNCTION__ inline
fp32_t RCP(const fp32_t value){
    return __frcp_rz(value);
}

}}}}}

# endif