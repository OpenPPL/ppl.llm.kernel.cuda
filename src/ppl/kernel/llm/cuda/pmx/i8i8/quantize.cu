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

#include "ppl/kernel/llm/cuda/pmx/i8i8/quantize.h"

#include "ppl/common/log.h"
#include "cudakernel/common/common.cuh"

#include "../quant_utils.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i8i8 {

template<int32_t TPB, matrix_layout_t TO_LAYOUT>
__global__
void minmax_quantize_fp16_kernel(
    const half* input, // [batch, quant_dim]
    const int64_t batch,
    const int64_t quant_dim,
    const float up_scale, // scale[i] = scale_val * up_scale for precision
    int8_t* output, // [batch, quant_dim]
    half* scale_output // [batch]
)
{
    constexpr int32_t WPT      = TPB / 32; // warp per thread block
    const int64_t batch_id     = blockIdx.x;
    const int64_t batch_offset = batch_id * quant_dim;
    __shared__ float red_smem[WPT];

    float local_max = 0.0f;
    for(int64_t i = threadIdx.x; i < quant_dim; i += TPB){
        float value = __half2float(input[batch_offset + i]);
        local_max = max(abs(value), local_max);
    }
    local_max = BLOCK_REDUCE_MAX<WPT>(local_max, red_smem);
    float scale = MIN_MAX_RANGE_TO_SCALE(local_max);

    MatrixLayoutHelper<TO_LAYOUT> idx_hlp;
    idx_hlp.Init(batch, quant_dim);

    for(int64_t i = threadIdx.x; i < quant_dim; i += TPB){
        float fp_value = __half2float(input[batch_offset + i]);
        output[idx_hlp.GetOffset(batch_id, i)] = (int8_t)__float2int_rn(fp_value / scale);
    }
    if(threadIdx.x == 0){
        scale_output[batch_id] = __float2half(scale * up_scale);
    }
}

ppl::common::RetCode minmax_quantize_fp16(
    cudaStream_t stream,
    const void* input, // fp16, [batch, quant_dim]
    const int64_t batch,
    const int64_t quant_dim,
    const float up_scale, // scale[i] = scale * up_scale for precision
    const matrix_layout_t to_layout,
    void* quantized, // int8, [batch, quant_dim]
    void* scale // fp16, [batch]
)
{
    constexpr int32_t TPB = 256;
    if (to_layout == MATRIX_LAYOUT_ROW_MAJOR) {
        minmax_quantize_fp16_kernel<TPB, MATRIX_LAYOUT_ROW_MAJOR>
        <<<batch, TPB, 0, stream>>>(
            (const half*)input, batch, quant_dim,
            up_scale, (int8_t*)quantized, (half*)scale
        );
    } else if (to_layout == MATRIX_LAYOUT_COL_MAJOR) {
        minmax_quantize_fp16_kernel<TPB, MATRIX_LAYOUT_COL_MAJOR>
        <<<batch, TPB, 0, stream>>>(
            (const half*)input, batch, quant_dim,
            up_scale, (int8_t*)quantized, (half*)scale
        );
    } else if (to_layout == MATRIX_LAYOUT_COL32) {
        minmax_quantize_fp16_kernel<TPB, MATRIX_LAYOUT_COL32>
        <<<batch, TPB, 0, stream>>>(
            (const half*)input, batch, quant_dim,
            up_scale, (int8_t*)quantized, (half*)scale
        );
    } else if (to_layout == MATRIX_LAYOUT_COL32_2R_4R4) {
        minmax_quantize_fp16_kernel<TPB, MATRIX_LAYOUT_COL32_2R_4R4>
        <<<batch, TPB, 0, stream>>>(
            (const half*)input, batch, quant_dim,
            up_scale, (int8_t*)quantized, (half*)scale
        );
    } else if (to_layout == MATRIX_LAYOUT_COL4_4R2_8C) {
        minmax_quantize_fp16_kernel<TPB, MATRIX_LAYOUT_COL4_4R2_8C>
        <<<batch, TPB, 0, stream>>>(
            (const half*)input, batch, quant_dim,
            up_scale, (int8_t*)quantized, (half*)scale
        );
    } else {
        LOG(ERROR) << "unsupported matrix layout: " << (int32_t)to_layout;
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}

template<int32_t TPB, int32_t VPT, matrix_layout_t FROM_LAYOUT>
__global__
void minmax_dequantize_fp16_kernel(
    const int32_t* input,    //  col32 layout [batch, quant_dim(channel)] or [M, N]
    const half* optional_bias, // fp16, [quant_dim]
    const half* scale_per_batch,
    const half* scale_per_channel,
    const int64_t batch,
    const int64_t quant_dim,
    const float down_scale_batch, // batch_scale_val = batch_scale[i] * down_scale_batch for precision
    const float down_scale_channel, // channel_scale_val = channel_scale[i] * down_scale_channel for precision
    half* output
)
{
    const int64_t batch_id = blockIdx.y;
    const int64_t batch_offset = batch_id * quant_dim;
    const int64_t tile_id  = blockIdx.x;
    const int64_t tile_offset = tile_id * TPB * VPT;

    if (tile_offset + threadIdx.x * VPT < quant_dim) {
        MatrixLayoutHelper<FROM_LAYOUT> idx_hlp;
        idx_hlp.Init(batch, quant_dim);

        int32_t local_in[VPT]; half local_w[VPT];
        copy<sizeof(int32_t) * VPT>(&input[idx_hlp.GetOffset(batch_id, tile_offset + threadIdx.x * VPT)], local_in);
        copy<sizeof(half) * VPT>(&scale_per_channel[tile_offset + threadIdx.x * VPT], local_w);

        float batch_scale = __half2float(scale_per_batch[batch_id]) * down_scale_batch;
        half local_out[VPT];
        # pragma unroll
        for(int32_t i = 0; i < VPT; i++){
            local_out[i] = __float2half(local_in[i] * __half2float(local_w[i]) * down_scale_channel * batch_scale);
        }

        if (optional_bias) {
            copy<sizeof(half) * VPT>(&optional_bias[tile_offset + threadIdx.x * VPT], local_w);
            for(int32_t i = 0; i < VPT; i++){
                local_out[i] = local_out[i] + local_w[i];
            }
        }

        copy<sizeof(half) * VPT>(local_out, &output[batch_offset + tile_offset + threadIdx.x * VPT]);
    }
}

ppl::common::RetCode minmax_dequantize_fp16(
    cudaStream_t stream,
    const void* input,    // int32，[batch, quant_dim(channel)] or [M, N]
    const void* optional_bias, // fp16, [quant_dim]
    const void* scale_per_batch,   // fp16, [batch]
    const void* scale_per_channel, // fp16, [quant_dim]
    const int64_t batch,
    const int64_t quant_dim,
    const float down_scale_batch, // batch_scale_val = batch_scale[i] * down_scale_batch for precision
    const float down_scale_channel, // channel_scale_val = channel_scale[i] * down_scale_channel for precision
    const matrix_layout_t from_layout,
    void* output // fp16, [batch, quant_dim]
)
{
    constexpr int32_t TPB = 256;
    constexpr int32_t VPT = 4;
    const dim3 block_grid {
        (unsigned int)ceilf(float(quant_dim) / (TPB * VPT)),
        (unsigned int)batch,
        1
    };

    if (quant_dim % VPT != 0) {
        LOG(ERROR) << "channel must be aligend with 4, but channel = " << quant_dim;
        return ppl::common::RC_INVALID_VALUE;
    }

    if (from_layout == MATRIX_LAYOUT_ROW_MAJOR) {
        minmax_dequantize_fp16_kernel<TPB, VPT, MATRIX_LAYOUT_ROW_MAJOR>
        <<<block_grid, TPB, 0, stream>>> (
            (const int32_t*)input, (const half*)optional_bias, (const half*)scale_per_batch, (const half*)scale_per_channel,
            batch, quant_dim, down_scale_batch, down_scale_channel, (half*)output
        );
    } else if (from_layout == MATRIX_LAYOUT_COL_MAJOR) {
        minmax_dequantize_fp16_kernel<TPB, VPT, MATRIX_LAYOUT_COL_MAJOR>
        <<<block_grid, TPB, 0, stream>>> (
            (const int32_t*)input, (const half*)optional_bias, (const half*)scale_per_batch, (const half*)scale_per_channel,
            batch, quant_dim, down_scale_batch, down_scale_channel, (half*)output
        );
    } else if (from_layout == MATRIX_LAYOUT_COL32) {
        minmax_dequantize_fp16_kernel<TPB, VPT, MATRIX_LAYOUT_COL32>
        <<<block_grid, TPB, 0, stream>>> (
            (const int32_t*)input, (const half*)optional_bias, (const half*)scale_per_batch, (const half*)scale_per_channel,
            batch, quant_dim, down_scale_batch, down_scale_channel, (half*)output
        );
    } else if (from_layout == MATRIX_LAYOUT_COL4_4R2_8C) {
        minmax_dequantize_fp16_kernel<TPB, VPT, MATRIX_LAYOUT_COL4_4R2_8C>
        <<<block_grid, TPB, 0, stream>>> (
            (const int32_t*)input, (const half*)optional_bias, (const half*)scale_per_batch, (const half*)scale_per_channel,
            batch, quant_dim, down_scale_batch, down_scale_channel, (half*)output
        );
    } else if (from_layout == MATRIX_LAYOUT_COL32_2R_4R4) {
        minmax_dequantize_fp16_kernel<TPB, VPT, MATRIX_LAYOUT_COL32_2R_4R4>
        <<<block_grid, TPB, 0, stream>>> (
            (const int32_t*)input, (const half*)optional_bias, (const half*)scale_per_batch, (const half*)scale_per_channel,
            batch, quant_dim, down_scale_batch, down_scale_channel, (half*)output
        );
    } else {
        LOG(ERROR) << "unsupported matrix layout: " << (int32_t)from_layout;
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}}

