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

# include "ppl/kernel/llm/cuda/quant/common.h"
# include "ppl/kernel/llm/cuda/quant/core.h"
# include "ppl/kernel/llm/cuda/quant/layout.h"
# include "ppl/common/log.h"

using namespace ppl::kernel::llm::cuda::quant;

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace quant { namespace i8 {

/*
    Function List:
        __BlockReduceMax - 一个临时工，用来在 thread block 里面求最大值

        groupwise_minmax_quantize_fp16i_int8o - 动态分组量化，fp16 输入，int8 输出

        tokenwise_minmax_quantize_fp16i_i8o - 动态Token量化，fp16 输入，int8 输出
        channelwise_minmax_quantize_fp16i_i8o - 静态通道量化，fp16 输入，int8 输出

        token_channel_dequantize_i32i_f16o - token-channel 双重解量化，int32 输入， fp16 输出
        groupwise_minmax_dequantize_int8i_fp16o - 动态分组解量化，int8 输入，fp16 输出
*/

/* Helper Function */
template<int32_t WPT>
__device__ inline
fp32_t __BlockReduceMax(fp32_t reducing, fp32_t *shared_mem){
    // Helper function for reduce softmax qkmax.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    if (lane_id == 0) {
        shared_mem[warp_id] = reducing;
    }
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];
    else reducing = -FLT_MAX;

# pragma unroll
    for (int mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

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
template<int TPB, int VPT> // 8 fp16 occupy 128 bytes, which can be loaded by a single thread at once.
__global__
void _groupwise_minmax_quantize_fp16i_int8o(
    const fp16_t  *in,
    const int32_t num_of_elements,
    int8_t        *out,
    fp16_t        *scale_out
){
    const int32_t idx = (blockIdx.x * TPB + threadIdx.x) * VPT;
    fp16_t local_in[VPT]; int8_t local_out[VPT];

    if (idx < num_of_elements){
        copy<sizeof(fp16_t) * VPT>(&in[idx], local_in);
        const fp16_t eps = __float2half(1e-5f);
        fp16_t scale = __float2half(0.0f);

    #pragma unroll
        for(int i = 0; i < VPT; i ++){
            scale = scale > abs(local_in[i]) ? scale : abs(local_in[i]);
        }
        scale = scale / __float2half(127.0f); 
        scale = scale > eps ? scale : eps;
    
        scale_out[blockIdx.x * TPB + threadIdx.x] = scale;
        fp16_t inv_s = __float2half(1.0f) / scale;
    
    #pragma unroll
        for(int i = 0; i < VPT; i ++){
            local_out[i] = (int8_t)__half2short_rn(local_in[i] * inv_s);
        }
        copy<sizeof(int8_t) * VPT>(local_out, &out[idx]);
    }
}


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
template<int TPB, int VPT>
__global__
void _groupwise_minmax_dequantize_int8i_fp16o(
    const int8_t  *in,
    const int32_t  num_of_elements,
    fp16_t        *out,
    const fp16_t  *scale
){
    const int32_t idx = (blockIdx.x * TPB + threadIdx.x) * VPT;
    int8_t local_in[VPT]; fp16_t local_out[VPT];

    if (idx < num_of_elements){
        copy<sizeof(int8_t) * VPT>(&in[idx], local_in);

    #pragma unroll
        for(int i = 0; i < VPT; i ++){
            local_out[i] = __float2half((float)local_in[i]) * scale[blockIdx.x * TPB + threadIdx.x];
        }
        copy<sizeof(fp16_t) * VPT>(local_out, &out[idx]);
    }
}

/*
_channelwise_minmax_quantize_fp16i_i8o 用于执行 channelwise 的矩阵量化
    输入矩阵必须形如 [input channel, output channel]，其 layout 的最后一维必须是 output channel
    这个函数将生成一个形如 [output channel] 的 scale 向量用于量化
    
    该函数沿着 input channel 的方向统计通道最大值(取绝对值)，并使得 scale = max_value / 127
    该函数要求 scale 必须大于 1e-7，否则将以该值进行覆盖，因此不会出现 scale = 0 的问题
    该函数会使用生成的 scale 完成对输入矩阵的量化，限制其表示范围在 [-127, 127] 之间。

    该函数返回两个值，分别是量化后的矩阵 [num_of_token, num_of_channel] 与解量化所需的 scale 向量 [output channel]

_channelwise_minmax_quantize_fp16i_i8o 与 _tokenwise_minmax_quantize_fp16i_i8o 的区别其实就是一个沿着第一维统计量化，另一个沿着最后一维统计量化。
    channelwise, tokenwise 这样的命名方式更符合量化领域的规范

    _channelwise_minmax_quantize_fp16i_i8o 访存不连续，性能很烂，但它通常是个预处理过程，没啥影响。
*/
template<int32_t TPB>
__global__
void _channelwise_minmax_quantize_fp16i_i8o(
    const fp16_t *in, // [num_of_token, num_of_channel]
    const int32_t dim,
    const int32_t num_of_channel,
    fp16_t *scale_out, int8_t *out
){
    const int32_t tile_idx    = blockIdx.x;
    const int32_t tile_offset = tile_idx * TPB;
    const int32_t local_idx   = tile_offset + threadIdx.x;

    if (local_idx < num_of_channel){
        fp32_t channel_max = 0.0f;
        for(int32_t i = 0; i < dim; i++){
            fp32_t value = __half2float(in[i * num_of_channel + local_idx]);
            channel_max = max(abs(value), channel_max);
        }

        fp32_t scale = MIN_MAX_RANGE_TO_SCALE(channel_max);
        for(int32_t i = 0; i < dim; i++){
            out[i * num_of_channel + local_idx] = QUANT_FP32_TO_INT8(
                __half2float(in[i * num_of_channel + local_idx]), scale
            );
        }

        scale_out[local_idx] = __float2half(scale);
    }
}

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
template<int32_t TPB, bool ConvertRowMajorToCol32>
__global__
void _tokenwise_minmax_quantize_fp16i_i8o(
    const fp16_t *in, // [token, hidden_dim]
    const int32_t hidden_dim,
    fp16_t *scale_out,
    int8_t *out
) {
    constexpr int32_t WPT      = TPB / 32; // warp per thread block
    const int32_t batch_id     = blockIdx.x;
    const int32_t batch_offset = batch_id * hidden_dim;
    __shared__ fp32_t red_smem[WPT];

    fp32_t local_max = 0.0f;
    for(int32_t i = threadIdx.x; i < hidden_dim; i += TPB){
        fp32_t value = __half2float(in[batch_offset + i]);
        local_max = max(abs(value), local_max);
    }
    local_max = __BlockReduceMax<WPT>(local_max, red_smem);
    fp32_t scale = MIN_MAX_RANGE_TO_SCALE(local_max);

    for(int32_t i = threadIdx.x; i < hidden_dim; i += TPB) {
        fp32_t fp_value = __half2float(in[batch_offset + i]);
        // convert layout
        if (ConvertRowMajorToCol32) {
            // convert data layout from rowmajor to col32
            // col32 layout is necessary for int8 gemm
            // check here: https://blog.speechmatics.com/gpu-quantisation
            const int32_t num_of_row = blockDim.x;
            const int32_t num_of_col = hidden_dim;
            int32_t col32_idx = LayoutConverter(
                num_of_row, num_of_col).RowMajorToCol32(batch_offset + i);
            out[col32_idx] = QUANT_FP32_TO_INT8(fp_value, scale);
        }
        else 
        {
            out[batch_offset + i] = QUANT_FP32_TO_INT8(fp_value, scale);
        }
    }
    if(threadIdx.x == 0){
        scale_out[batch_id] = __float2half(scale);
    }
}

/*
_token_channel_dequantize_i32i_f16o 用于执行 token + channel 的双重解量化
    大语言模型的推理过程中参与矩阵乘的 num of tokens 会一直变化，我们可不想当它太大的时候造成量化误差过高
    所以我们的矩阵乘法不仅仅是沿着 gemm output channel 量化的，其激活也会沿着 token 的方向量化

    进行解量化时，我们要使用 output channel 与 token 的两种量化参数同时进行该操作。

    输入矩阵必须形如 [num of tokens, hidden dim]，其 layout 的第一维必须是 num of tokens
    参数矩阵必须同时传入 scale_per_token, scale_per_channel 两个

    该函数将执行 output[i][j] = input[i][j] * scale_per_token[i] * scale_per_channel[j] 进行解量化
    该函数在 batch 大约为 512 时可以打满访存带宽，好像 batch 大了反而菜一些
*/
template<int32_t TPB, int32_t VPT, bool ConvertCol32ToRowMajor>
__global__
void _token_channel_dequantize_i32i_f16o(
    int32_t *in,    // [num_of_token, hidden_dim] or [M, N]
    const int32_t batch_size,
    const int32_t hidden_dim,
    const fp16_t *scale_per_batch,
    const fp16_t *scale_per_channel,
    fp16_t *out
) {
    const int32_t batch_id = blockIdx.y;
    const int32_t batch_offset = batch_id * hidden_dim;
    const int32_t tile_id  = blockIdx.x;
    const int32_t tile_offset = tile_id * TPB * VPT;

    int32_t local_in[VPT]; fp16_t local_w[VPT];
    int32_t input_index = batch_offset + tile_offset + threadIdx.x * VPT;
    if (ConvertCol32ToRowMajor){
        input_index = LayoutConverter(
            num_of_row, num_of_col).Col32ToRowMajor(input_index);
    }
    copy<sizeof(int32_t) * VPT>(&in[input_index], local_in);
    copy<sizeof(fp16_t) * VPT>(&scale_per_channel[tile_offset + threadIdx.x * VPT], local_w);
    
    fp32_t batch_scale = __half2float(scale_per_batch[batch_id]);
    fp16_t local_out[VPT];
    # pragma unroll
    for(int32_t i = 0; i < VPT; i++){
        local_out[i] = __float2half(local_in[i] * __half2float(local_w[i]) * batch_scale);
    }
    copy<sizeof(fp16_t) * VPT>(local_out, &out[batch_offset + tile_offset + threadIdx.x * VPT]);
    
}

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
){
    constexpr int32_t VPT = 16 / sizeof(fp16_t);
    constexpr int32_t TPB = 256;
    const int32_t grid_size = num_of_elements / VPT / TPB + (num_of_elements % (VPT * TPB) != 0);

    if (num_of_elements % 8 != 0){
        LOG(ERROR) << "Input data length must be a multiple of 8.";
        return ppl::common::RC_UNSUPPORTED;
    }

    _groupwise_minmax_dequantize_int8i_fp16o<TPB, VPT>
    <<<grid_size, TPB, 0, stream>>>(
        input, num_of_elements, 
        output, group_scale
    );
    return ppl::common::RC_SUCCESS;
}

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
){
    constexpr int32_t VPT = 16 / sizeof(fp16_t);
    constexpr int32_t TPB = 256;
    const int32_t grid_size = x.numel() / VPT / TPB + (x.numel() % (VPT * TPB) != 0);

    if (num_of_elements % 8 != 0){
        LOG(ERROR) << "Input data length must be a multiple of 8.";
        return ppl::common::RC_UNSUPPORTED;
    }

    _groupwise_minmax_quantize_int8i_fp16o<TPB, VPT>
    <<<grid_size, TPB, 0, stream>>>(
        input, num_of_elements, 
        output, group_scale
    );

    return ppl::common::RC_SUCCESS;
}


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
) {
    constexpr int32_t TPB = 256;
    const int32_t num_of_thread_block = num_of_out_channel / TPB + (num_of_out_channel % TPB != 0);

    _channelwise_minmax_quantize_fp16i_i8o<TPB>
    <<<num_of_thread_block, TPB, 0, stream>>>(
        input, dim, num_of_channel,
        scale_out, output
    );

    return ppl::common::RC_SUCCESS;
}

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
) {
    constexpr int32_t TPB = 256;
    _tokenwise_minmax_quantize_fp16i_i8o<TPB, ConvertRowMajorToCol32>
    <<<num_of_batch, TPB, 0, stream>>>(
        input, hidden_dim,
        scale_out, output
    );

    return ppl::common::RC_SUCCESS;
}

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
) {
    constexpr int32_t TPB = 256;
    constexpr int32_t VPT = 4;
    const dim3 block_grid {hidden_dim / (TPB*VPT), num_of_token, 1};
    
    _token_channel_dequantize_i32i_f16o<TPB, VPT, ConvertCol32ToRowMajor>
    <<<block_grid, TPB, 0, stream>>> (
        input, num_of_token, hidden_dim,
        scale_per_token, scale_per_channel,
        output
    );

    return ppl::common::RC_SUCCESS;
}

}}}}}}
