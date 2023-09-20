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

# ifndef __PPL_KERNEL_LLM_CUDA_QUANT_I8_CORE_H__
# define __PPL_KERNEL_LLM_CUDA_QUANT_I8_CORE_H__

# include "ppl/kernel/llm/cuda/common/general_include.h"
# include "ppl/kernel/llm/cuda/quant/common.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace quant { namespace i8 {

/*
The function performs group-wise dynamic de-quantization, 
    which de-quantizes the int8 to fp16. 

group-wise quantization means that the quantization granularity of this function is extremely small. 
    Typically, we perform per-tensor or per-channel quantization because only these two quantization schemes have 
    high-performance implementations for matrix multiplication. 

However, this function is designed for the temporary storage of kv and does not involve any calculation issues.
Therefore, we can design a new quantization scheme to obtain lower quantization bit widths or higher quantization precision. 

We define a group as k(k=8 in most cases) adjacent elements, and the elements in the group share a set of quantization parameters. 
The quantization granularity of this quantization scheme is extremely small, thus it has extremely high quantization precision.

The so-called dynamic quantization means that each input quantization parameter is calculated in real-time, 
    rather than being given by a calibration function.
*/
ppl::common::RetCode groupwise_minmax_dequantize_int8i_fp16o(
    cudaStream_t stream,
    const int8_t* input,
    const fp16_t* group_scale,
    fp16_t* output,
    const int64_t num_of_elements
);

/*
The function performs group-wise dynamic quantization, 
    which quantizes the fp16 to int8. 

group-wise quantization means that the quantization granularity of this function is extremely small. 
    Typically, we perform per-tensor or per-channel quantization because only these two quantization schemes have 
    high-performance implementations for matrix multiplication. 

However, this function is designed for the temporary storage of kv and does not involve any calculation issues.
Therefore, we can design a new quantization scheme to obtain lower quantization bit widths or higher quantization precision. 

We define a group as k(k=8 in most cases) adjacent elements, and the elements in the group share a set of quantization parameters. 
The quantization granularity of this quantization scheme is extremely small, thus it has extremely high quantization precision.

The so-called dynamic quantization means that each input quantization parameter is calculated in real-time, 
    rather than being given by a calibration function.
*/
ppl::common::RetCode groupwise_minmax_quantize_int8i_fp16o(
    cudaStream_t stream,
    const fp16_t* input,
    fp16_t* group_scale,
    int8_t* output,
    int64_t num_of_elements
);

/*
_channelwise_minmax_quantize_fp16i_i8o 用于执行 channelwise 的矩阵量化
    输入矩阵必须形如 [input channel, output channel]，其 layout 的最后一维必须是 output channel
    这个函数将生成一个形如 [output channel] 的 scale 向量用于量化
    
    该函数沿着 input channel 的方向统计通道最大值(取绝对值)，并使得 scale = max_value / 127
    该函数要求 scale 必须大于 1e-7，否则将以该值进行覆盖，因此不会出现 scale = 0 的问题
    该函数会使用生成的 scale 完成对输入矩阵的量化，限制其表示范围在 [-127, 127] 之间。

    该函数返回两个值，分别是量化后的矩阵 [input channel, output channel] 与解量化所需的 scale 向量 [output channel]

_channelwise_minmax_quantize_fp16i_i8o 与 _tokenwise_minmax_quantize_fp16i_i8o 的区别其实就是一个沿着第一维统计量化，另一个沿着最后一维统计量化。
    channelwise, tokenwise 这样的命名方式更符合量化领域的规范

    _channelwise_minmax_quantize_fp16i_i8o 访存不连续，性能很烂，但它通常是个预处理过程，没啥影响。
*/
ppl::common::RetCode channelwise_minmax_quantize_fp16i_i8o(
    cudaStream_t stream,
    const fp16_t *input // [dim or input channel, num_of_channel] 
    const int64_t dim,
    const int64_t num_of_channel,
    fp16_t *scale_out, // [num_of_channel] 
    int8_t *output
);
/*
_tokenwise_minmax_quantize_fp16i_i8o 用于执行 tokenwise 的矩阵量化
    输入矩阵必须形如 [num of tokens, hidden dim]，其 layout 的第一维必须是 num of tokens
    这个函数将生成一个形如 [num of tokens] 的 scale 向量用于量化
    
    该函数沿着 input channel 的方向统计通道最大值(取绝对值)，并使得 scale = max_value / 127
    该函数要求 scale 必须大于 1e-7，否则将以该值进行覆盖，因此不会出现 scale = 0 的问题
    该函数会使用生成的 scale 完成对输入矩阵的量化，限制其表示范围在 [-127, 127] 之间。

    该函数返回两个值，分别是量化后的矩阵 [num of tokens, hidden dim] 与解量化所需的 scale 向量 [num of tokens]

_channelwise_minmax_quantize_fp16i_i8o 与 _tokenwise_minmax_quantize_fp16i_i8o 的区别其实就是一个沿着第一维统计量化，另一个沿着最后一维统计量化。
    channelwise, tokenwise 这样的命名方式更符合量化领域的规范
*/
template<bool ConvertRowMajorToCol32>
ppl::common::RetCode tokenwise_minmax_quantize_fp16i_i8o(
    cudaStream_t stream,
    const fp16_t *input, // [num of tokens, hidden dim]
    const int64_t hidden_dim,
    fp16_t *scale_out,
    int8_t *output
);

/*
token_channel_dequantize_i32i_f16o 用于执行 token + channel 的双重解量化
    大语言模型的推理过程中参与矩阵乘的 num of tokens 会一直变化，我们可不想当它太大的时候造成量化误差过高
    所以我们的矩阵乘法不仅仅是沿着 gemm output channel 量化的，其激活也会沿着 token 的方向量化

    进行解量化时，我们要使用 output channel 与 token 的两种量化参数同时进行该操作。

    输入矩阵必须形如 [num of tokens, hidden dim]，其 layout 的第一维必须是 num of tokens
    参数矩阵必须同时传入 scale_per_token, scale_per_channel 两个

    该函数将执行 output[i][j] = input[i][j] * scale_per_token[i] * scale_per_channel[j] 进行解量化
    该函数在 batch 大约为 512 时可以打满访存带宽，好像 batch 大了反而菜一些
*/
template<bool ConvertCol32ToRowMajor>
ppl::common::RetCode token_channel_dequantize_i32i_f16o(
    cudaStream_t stream,
    int32_t *input,    // [num_of_token, hidden_dim] or [M, N]
    const int64_t num_of_token,
    const int64_t hidden_dim,
    const fp16_t *scale_per_token,   // [num_of_token]
    const fp16_t *scale_per_channel, // [hidden_dim]
    fp16_t *output     // [num_of_token, hidden_dim]
);

}}}}}}

# endif