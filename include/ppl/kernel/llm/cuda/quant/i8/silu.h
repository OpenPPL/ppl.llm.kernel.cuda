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

# ifndef __PPL_KERNEL_LLM_CUDA_QUANT_I8_SILU_H__
# define __PPL_KERNEL_LLM_CUDA_QUANT_I8_SILU_H__

# include "ppl/kernel/llm/cuda/common/general_include.h"
# include "ppl/kernel/llm/cuda/quant/common.h"
# include "ppl/kernel/llm/cuda/quant/layout.h"

using namespace ppl::kernel::llm::cuda::quant;

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace quant { namespace i8 {

__DEVICE_INLINE_FUNCTION__
fp32_t __silu_fp32(const fp32_t val){
    return val / (1.f + __expf(-val));
}

__DEVICE_INLINE_FUNCTION__
fp32_t __silu_fp16(const fp16_t val){
    auto fp32 = __half2float(val);
    return fp32 / (1.f + __expf(-fp32));
}

/*
 * Silu 与 token-channel 解量化 与 token 量化 的融合算子
 * 这个算子会执行 Silu, Mul, Dequantize, Quantize 三个操作
 * 其两个输入 input, gate 将首先被解量化到 fp16
 * 而后执行 output_fp16 = silu(input) * gate
 * 最后执行 output_int8 = dynamic per token quantize(output_fp16)
 * 这个函数要求其输入输出排布均是 col32 的
 * 
 * 这个函数执行动态 per token 量化，
*/
template<bool GATED>
ppl::common::RetCode silu_i32_i8_col32(
    const int32_t *input,                // [num_of_token, hidden_dim] layout: col32
    const int32_t *gate,                 // [num_of_token, hidden_dim] layout: col32, 没有就直接传空指针
    const int64_t num_of_token,
    const int64_t hidden_dim,
    fp16_t        *input_token_scale     // [num_of_token]
    const fp16_t  *input_channel_scale   // [hidden_dim]
    fp16_t        *gate_token_scale      // [num_of_token], 没有就直接传空指针
    const fp16_t  *gate_channel_scale    // [hidden_dim], 没有就直接传空指针
    fp16_t        *workspace             // [num_of_token, hidden_dim]
    int8_t        *int8_output           // [num_of_token, hidden_dim]
);

/*
 * Silu 与 token-channel 解量化 的融合算子
 * 这个算子会执行 Silu, Mul, Dequantize 三个操作
 * 其两个输入 input, gate 将首先被解量化到 fp16
 * 而后执行 silu(input) * gate
 * 这个函数要求其输入排布是 col32 的，可由参数 ConvertLayoutToRowMajor 决定在输出时是否使用 row major 排布
*/
template<bool GATED, bool ConvertLayoutToRowMajor>
ppl::common::RetCode _silu_i32_fp16_col32(
    const int32_t *input,                // [num_of_token, hidden_dim] layout: col32
    const int32_t *gate,                 // [num_of_token, hidden_dim] layout: col32
    const int64_t num_of_token,
    const int64_t hidden_dim,
    fp16_t        *input_token_scale     // [num_of_token]
    const fp16_t  *input_channel_scale   // [hidden_dim]
    fp16_t        *gate_token_scale      // [num_of_token]
    const fp16_t  *gate_channel_scale    // [hidden_dim]
    fp16_t        *fp16_output           // [num_of_token, hidden_dim]
);

}}}}}}

# endif