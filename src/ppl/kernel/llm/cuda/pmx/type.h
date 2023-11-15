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
#include <math.h>

__device__ inline float2 operator+(const float2& a, const float2& b) {
    return {a.x + b.x, a.y + b.y};
}
namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

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

}}}}}

#endif
