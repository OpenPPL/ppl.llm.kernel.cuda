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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_TYPE_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_TYPE_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

using bf16_t   = nv_bfloat16;
using bf16x2_t = nv_bfloat162;

using fp16_t   = half;
using fp16x2_t = half2;

using fp32_t   = float;
using fp32x2_t = float2;
using fp32x4_t = float4;

using int8x2_t = char2;
using int8x4_t = char4;

using uint8x2_t = uchar2;
using uint8x4_t = uchar4;

using int32x4_t = int4;
using int32x2_t = int2;

using int4x8_t = int32_t;
using int4x4_t = int16_t;
using int4x2_t = int8_t;




template<typename T>
struct ToType2 {};

template<>
struct ToType2<fp16_t> {
    typedef fp16x2_t type;
};

template<>
struct ToType2<bf16_t> {
    typedef bf16x2_t type;
};




template<typename T>
struct FromType2 {};

template<>
struct FromType2<fp16x2_t> {
    typedef fp16_t type ;
};

template<>
struct FromType2<bf16x2_t> {
    typedef bf16_t type;
};




template<typename T>
inline __host__ __device__ fp32_t tofp32(T val);

template<typename T>
inline __host__ __device__ fp32x2_t tofp32x2(T val);

template<>
inline __host__ __device__ fp32_t tofp32<fp16_t>(fp16_t val) {
    return __half2float(val);
}

template<>
inline __host__ __device__ fp32x2_t tofp32x2<fp16x2_t>(fp16x2_t val) {
    return __half22float2(val);
}

template<>
inline __host__ __device__ fp32_t tofp32<bf16_t>(bf16_t val) {
    return __bfloat162float(val);
}

template<>
inline __host__ __device__ fp32x2_t tofp32x2<bf16x2_t>(bf16x2_t val) {
    return __bfloat1622float2(val);
}




template<typename T>
inline __host__ __device__ T fromfp32(fp32_t val);

template<typename T>
inline __host__ __device__ T fromfp32x2(fp32x2_t val);

template<>
inline __host__ __device__ fp16_t fromfp32<fp16_t>(fp32_t val) {
    return __float2half(val);
}

template<>
inline __host__  __device__ fp16x2_t fromfp32x2<fp16x2_t>(fp32x2_t val) {
    return __float22half2_rn(val);
}

template<>
inline __host__ __device__ bf16_t fromfp32<bf16_t>(fp32_t val) {
    return __float2bfloat16(val);
}

template<>
inline __host__ __device__ bf16x2_t fromfp32x2<bf16x2_t>(fp32x2_t val) {
    return __float22bfloat162_rn(val);
}

}}}}}

#endif
