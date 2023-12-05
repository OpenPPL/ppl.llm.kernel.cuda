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

struct minmax_dequantize_split3_fp16_kernel_param {
    int32_t* input;    //  col32 layout [batch, quant_dim(channel)] or [M, N]
    half* optional_bias; // fp16, [quant_dim]
    half* scale_per_batch;
    half* scale_per_channel;
    int64_t batch;
    int64_t quant_dim;
    int64_t split_dim_0;
    int64_t split_dim_1;
    int64_t split_dim_2;
    float down_scale_batch; // batch_scale_val = batch_scale[i] * down_scale_batch for precision
    float down_scale_channel; // channel_scale_val = channel_scale[i] * down_scale_channel for precision
    half* output_0;
    half* output_1;
    half* output_2;
};

template<int32_t TPB, int32_t VPT, matrix_layout_t FROM_LAYOUT>
__global__
void minmax_dequantize_split3_fp16_kernel(
    minmax_dequantize_split3_fp16_kernel_param p
)
{
    const int64_t batch_id = blockIdx.y;
    const int64_t tile_id  = blockIdx.x;
    const int64_t tile_offset = tile_id * TPB * VPT;

    if (tile_offset + threadIdx.x * VPT < p.quant_dim) {
        MatrixLayoutHelper<FROM_LAYOUT> idx_hlp;
        idx_hlp.Init(p.batch, p.quant_dim);

        int32_t local_in[VPT]; half local_w[VPT];
        copy<sizeof(int32_t) * VPT>(&p.input[idx_hlp.GetOffset(batch_id, tile_offset + threadIdx.x * VPT)], local_in);
        copy<sizeof(half) * VPT>(&p.scale_per_channel[tile_offset + threadIdx.x * VPT], local_w);

        float batch_scale = __half2float(p.scale_per_batch[batch_id]) * p.down_scale_batch;
        half local_out[VPT];
        # pragma unroll
        for(int32_t i = 0; i < VPT; i++){
            local_out[i] = __float2half(local_in[i] * __half2float(local_w[i]) * p.down_scale_channel * batch_scale);
        }

        if (p.optional_bias) {
            copy<sizeof(half) * VPT>(&p.optional_bias[tile_offset + threadIdx.x * VPT], local_w);
            for(int32_t i = 0; i < VPT; i++){
                local_out[i] = local_out[i] + local_w[i];
            }
        }

        if (tile_offset + threadIdx.x * VPT < p.split_dim_0) {
            copy<sizeof(half) * VPT>(local_out, &p.output_0[batch_id * p.split_dim_0 + tile_offset + threadIdx.x * VPT]);
        } else if (tile_offset + threadIdx.x * VPT < p.split_dim_0 + p.split_dim_1) {
            copy<sizeof(half) * VPT>(local_out, &p.output_1[batch_id * p.split_dim_1 + tile_offset + threadIdx.x * VPT - p.split_dim_0]);
        } else if (tile_offset + threadIdx.x * VPT < p.split_dim_0 + p.split_dim_1 + p.split_dim_2) {
            copy<sizeof(half) * VPT>(local_out, &p.output_2[batch_id * p.split_dim_2 + tile_offset + threadIdx.x * VPT - p.split_dim_0 - p.split_dim_1]);
        }
    }
}

ppl::common::RetCode minmax_dequantize_split3_fp16(
    cudaStream_t stream,
    const void* input,    // int32，[batch, quant_dim(channel)] or [M, N]
    const void* optional_bias, // fp16, [quant_dim]
    const void* scale_per_batch,   // fp16, [batch]
    const void* scale_per_channel, // fp16, [quant_dim]
    const int64_t batch,
    const int64_t quant_dim,
    const int64_t split_dim_0,
    const int64_t split_dim_1,
    const int64_t split_dim_2,
    const float down_scale_batch, // batch_scale_val = batch_scale[i] * down_scale_batch for precision
    const float down_scale_channel, // channel_scale_val = channel_scale[i] * down_scale_channel for precision
    const matrix_layout_t from_layout,
    void* output_0, // fp16, [batch, split_dim_0]
    void* output_1, // fp16, [batch, split_dim_1]
    void* output_2 // fp16, [batch, split_dim_2]
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

    if (split_dim_0 % VPT != 0) {
        LOG(ERROR) << "split_dim_0 must be aligend with 4, but channel = " << split_dim_0;
        return ppl::common::RC_INVALID_VALUE;
    }
    if (split_dim_1 % VPT != 0) {
        LOG(ERROR) << "split_dim_1 must be aligend with 4, but channel = " << split_dim_1;
        return ppl::common::RC_INVALID_VALUE;
    }
    if (split_dim_2 % VPT != 0) {
        LOG(ERROR) << "split_dim_2 must be aligend with 4, but channel = " << split_dim_2;
        return ppl::common::RC_INVALID_VALUE;
    }

    minmax_dequantize_split3_fp16_kernel_param p = {
        (int32_t*)input,    // int32，[batch, quant_dim(channel)] or [M, N]
        (half*)optional_bias, // fp16, [quant_dim]
        (half*)scale_per_batch,   // fp16, [batch]
        (half*)scale_per_channel, // fp16, [quant_dim]
        batch,
        quant_dim,
        split_dim_0,
        split_dim_1,
        split_dim_2,
        down_scale_batch, // batch_scale_val = batch_scale[i] * down_scale_batch for precision
        down_scale_channel, // channel_scale_val = channel_scale[i] * down_scale_channel for precision
        (half*)output_0, // fp16, [batch, split_dim_0]
        (half*)output_1, // fp16, [batch, split_dim_1]
        (half*)output_2, // fp16, [batch, split_dim_2]
    };

    if (from_layout == MATRIX_LAYOUT_ROW_MAJOR) {
        minmax_dequantize_split3_fp16_kernel<TPB, VPT, MATRIX_LAYOUT_ROW_MAJOR>
        <<<block_grid, TPB, 0, stream>>>(p);
    } else if (from_layout == MATRIX_LAYOUT_COL_MAJOR) {
        minmax_dequantize_split3_fp16_kernel<TPB, VPT, MATRIX_LAYOUT_COL_MAJOR>
        <<<block_grid, TPB, 0, stream>>>(p);
    } else if (from_layout == MATRIX_LAYOUT_COL32) {
        minmax_dequantize_split3_fp16_kernel<TPB, VPT, MATRIX_LAYOUT_COL32>
        <<<block_grid, TPB, 0, stream>>>(p);
    } else if (from_layout == MATRIX_LAYOUT_COL4_4R2_8C) {
        minmax_dequantize_split3_fp16_kernel<TPB, VPT, MATRIX_LAYOUT_COL4_4R2_8C>
        <<<block_grid, TPB, 0, stream>>>(p);
    } else if (from_layout == MATRIX_LAYOUT_COL32_2R_4R4) {
        minmax_dequantize_split3_fp16_kernel<TPB, VPT, MATRIX_LAYOUT_COL32_2R_4R4>
        <<<block_grid, TPB, 0, stream>>>(p);
    } else {
        LOG(ERROR) << "unsupported matrix layout: " << (int32_t)from_layout;
        return ppl::common::RC_UNSUPPORTED;
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}}

