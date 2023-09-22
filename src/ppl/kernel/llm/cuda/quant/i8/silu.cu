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

# include "ppl/kernel/llm/cuda/quant/i8/silu.h"
# include "ppl/common/log.h"

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
 * Silu 与 token-channel 解量化 的融合算子
 * 这个算子会执行 Silu, Mul, Dequantize 三个操作
 * 其两个输入 input, gate 将首先被解量化到 fp16
 * 而后执行 silu(input) * gate
 * 这个函数要求其输入排布是 col32 的，可由参数 ConvertLayoutToRowMajor 决定在输出时是否使用 row major 排布
*/
template<bool GATED, int32_t TPB, bool ConvertLayoutToRowMajor>
__global__ 
void _silu_i32_fp16_col32(
    const int32_t *input,                // [num_of_token, hidden_dim] layout: col32
    const int32_t *gate,                 // [num_of_token, hidden_dim] layout: col32
    const int64_t num_of_token,
    const int64_t hidden_dim,
    fp16_t        *input_token_scale     // [num_of_token]
    const fp16_t  *input_channel_scale   // [hidden_dim]
    fp16_t        *gate_token_scale      // [num_of_token]
    const fp16_t  *gate_channel_scale    // [hidden_dim]
    fp16_t        *fp16_output            // [num_of_token, hidden_dim]
) {
    const int32_t token_id = blockIdx.x;
    auto layout_converter = LayoutConverter(num_of_token, hidden_dim);
    for(int32_t i = threadIdx.x; i < hidden_dim; i += TPB) {
        const int32_t col32_offset = layout_converter.RowColToOffset(token_id, i);

        const int32_t input = input[col32_offset];
        const fp32_t  dq_input = input * (
            __half2float(input_token_scale[token_id]) * 
            __half2float(input_channel_scale[i])
        );

        fp32_t ret = __silu_fp32(de_input);

        if(GATED) {
            int32_t gate = gate[col32_offset];
            fp32_t  dq_gate = gate * (
                __half2float(gate_token_scale[token_id]) * 
                __half2float(gate_channel_scale[i])
            );
            ret *= gate;
        }
        
        if (ConvertLayoutToRowMajor){
            fp16_output[layout_converter.Col32ToRowMajor(col32_offset)] = __float2half(ret);
        } else{
            fp16_output[col32_offset] = __float2half(ret);
        }
    }
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
template<bool GATED, int32_t TPB>
__global__ 
void _silu_i32_i8_col32(
    const int32_t *input,                // [num_of_token, hidden_dim] layout: col32
    const int32_t *gate,                 // [num_of_token, hidden_dim] layout: col32
    const int64_t num_of_token,
    const int64_t hidden_dim,
    fp16_t        *input_token_scale     // [num_of_token]
    const fp16_t  *input_channel_scale   // [hidden_dim]
    fp16_t        *gate_token_scale      // [num_of_token]
    const fp16_t  *gate_channel_scale    // [hidden_dim]
    fp16_t        *workspace             // [num_of_token, hidden_dim]
    int8_t        *int8_output)          // [num_of_token, hidden_dim]
{
    constexpr int32_t WARP_SIZE = 32;
    constexpr int32_t WPT       = 32;    // warp per thread block.
    const int32_t token_id = blockIdx.x;
    auto layout_converter = LayoutConverter(num_of_token, hidden_dim);

    fp32_t local_max = 0.0;
    __shared__ fp32_t red_smem[WPT];

    for(int32_t i = threadIdx.x; i < hidden_dim; i += TPB) {
        const int32_t col32_offset = layout_converter.RowColToOffset(token_id, i);

        const int32_t input = input[col32_offset];
        const fp32_t  dq_input = input * (
            __half2float(input_token_scale[token_id]) * 
            __half2float(input_channel_scale[i])
        );

        fp32_t ret = __silu_fp32(de_input);

        if(GATED) {
            int32_t gate = gate[col32_offset];
            fp32_t  dq_gate = gate * (
                __half2float(gate_token_scale[token_id]) * 
                __half2float(gate_channel_scale[i])
            );
            ret *= gate;
        }

        local_max = max(abs(ret), local_max);
        workspace[col32_offset] = __float2half(ret);
    }

    local_max = __BlockReduceMax<WPT>(local_max, red_smem);
    // quantize from workspace(fp16) to output(int8)
    fp32_t scale_rcp = RCP(local_max / 127);
    
    for(int32_t i = threadIdx.x; i < hidden_dim; i += TPB) {
        const int32_t col32_offset = layout_converter.RowColToOffset(token_id, i);
        const fp16_t input = workspace[col32_offset];

        int8_output[col32_offset] = QUANT_FP32_TO_INT8_RCP(input, scale_crp);
    }
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
) {
    const int64_t TPB = 256;
    // 这个 kernel 需要沿着 token 维度统计量化信息，所以每一个 thread block 只负责处理一个 token 的内容
    const int32_t num_of_thread_block = num_of_token; 

    _silu_i32_i8_col32<GATED, TPB>
    <<<num_of_thread_block, TPB, 0, stream>>> (
        input, gate, num_of_token, hidden_dim,
        input_token_scale, input_channel_scale,
        gate_token_scale, gate_channel_scale,
        workspace, int8_output
    );

    return ppl::common::RC_SUCCESS;
}


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
) {
    const int64_t TPB = 256;
    // 这个 kernel 虽然不需要沿着 token 维度统计量化信息
    // 但我懒得改了，希望有缘人把它改写成另外的写法
    const int32_t num_of_thread_block = num_of_token;

    _silu_i32_fp16_col32<GATED, TPB, ConvertLayoutToRowMajor>
    <<<num_of_thread_block, TPB, 0, stream>>> (
        input, gate, num_of_token, hidden_dim,
        input_token_scale, input_channel_scale,
        gate_token_scale, gate_channel_scale,
        fp16_output
    );

    return ppl::common::RC_SUCCESS;
}

}}}}}}