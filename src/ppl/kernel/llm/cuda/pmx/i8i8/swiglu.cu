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

struct minmax_requantize_swiglu_fp16_kernel_param {
    int32_t* input;    //  [batch, 2 * quant_dim(channel)] or [M, N]
    half* input_scale_per_batch;
    half* input_scale_per_channel;
    int64_t batch;
    int64_t quant_dim;
    float beta;
    float up_scale;
    float down_scale_batch; // batch_scale_val = batch_scale[i] * down_scale_batch for precision
    float down_scale_channel; // channel_scale_val = channel_scale[i] * down_scale_channel for precision
    half* workspace; // [batch, quant_dim(channel)] or [M, N]
    int8_t* output;
    half* scale;
};

template<int32_t TPB, matrix_layout_t LAYOUT>
__global__
void minmax_requantize_swiglu_fp16_kernel(
    minmax_requantize_swiglu_fp16_kernel_param p
)
{
    constexpr int32_t VPT       = 16;
    constexpr int32_t V8PT      = VPT / sizeof(int8_t);
    constexpr int32_t V16PT     = VPT / sizeof(half);
    constexpr int32_t V32PT     = VPT / sizeof(int32_t);
    constexpr int32_t WPT       = 32; // warp per thread block
    const int64_t batch_id      = blockIdx.x;

    __shared__ float red_smem[WPT];

    static_assert(TPB >= WPT);
    if(threadIdx.x < WPT) {
        red_smem[threadIdx.x] = 0;
    }
    __syncthreads();

    MatrixLayoutHelper<LAYOUT> input_idx_hlp, output_idx_hlp;
    input_idx_hlp.Init(p.batch, 2 * p.quant_dim);
    output_idx_hlp.Init(p.batch, p.quant_dim);

    const float batch_scale = __half2float(p.input_scale_per_batch[batch_id]) * p.down_scale_batch;

    float local_max = 0.0f;
    // TODO
    if (LAYOUT == MATRIX_LAYOUT_COL4_4R2_8C) {

    } else {
        for (int64_t i = threadIdx.x * V16PT; i < p.quant_dim; i += TPB * V16PT) {
            int32_t local_in[V16PT];
            half local_channel_scale[V16PT];
            int32_t local_gate[V16PT];
            half local_gate_channel_scale[V16PT];
            half local_dq_in[V16PT];

            const auto input_offset = input_idx_hlp.GetOffset(batch_id, i);
            copy<VPT>(&p.input[input_offset], local_in);
            copy<VPT>(&p.input[input_offset + V32PT], &local_in[V32PT]);
            copy<VPT>(&p.input_scale_per_channel[i], local_channel_scale);
            const auto gate_offset = input_idx_hlp.GetOffset(batch_id, i + p.quant_dim);
            copy<VPT>(&p.input[gate_offset], local_gate);
            copy<VPT>(&p.input[gate_offset + V32PT], &local_gate[V32PT]);
            copy<VPT>(&p.input_scale_per_channel[i + p.quant_dim], local_gate_channel_scale);

            #pragma unroll
            for (int32_t vidx = 0; vidx < V16PT; vidx += 1) {
                auto dq_input = local_in[vidx] * __half2float(local_channel_scale[vidx]) * p.down_scale_channel * batch_scale;
                auto dq_gate = local_gate[vidx] * __half2float(local_gate_channel_scale[vidx]) * p.down_scale_channel * batch_scale;

                // swiglu
                auto res = (dq_input / (1.f + __expf(-dq_input * p.beta))) * dq_gate;

                local_dq_in[vidx] = __float2half(res);
                local_max = fmax(fabs(res), local_max);
            }

            const auto output_offset = output_idx_hlp.GetOffset(batch_id, i);
            copy<VPT>(local_dq_in, &p.workspace[output_offset]);
        }

        local_max = block_reduce_max<WPT>(local_max, red_smem);
        float scale = min_max_range_to_scale(local_max, INT8_QLEVEL);
        if(threadIdx.x == 0){
            p.scale[batch_id] = __float2half(scale * p.up_scale);
        }

        for (int64_t i = threadIdx.x * V8PT; i < p.quant_dim; i += TPB * V8PT) {
            half local_dq_in[V8PT];
            int8_t local_out[V8PT];

            const auto output_offset = output_idx_hlp.GetOffset(batch_id, i);
            copy<VPT>(&p.workspace[output_offset], local_dq_in);
            copy<VPT>(&p.workspace[output_offset + V16PT], &local_dq_in[V16PT]);

            #pragma unroll
            for (int32_t vidx = 0; vidx < V8PT; vidx += 1) {
                local_out[vidx] = (int8_t)__float2int_rn(__half2float(local_dq_in[vidx]) / scale);
            }

            copy<VPT>(local_out, &p.output[output_offset]);
        }
    }
}

#define KERNEL_BLOCK() do {\
    if (layout == MATRIX_LAYOUT_ROW_MAJOR) {\
        minmax_requantize_swiglu_fp16_kernel<TPB, MATRIX_LAYOUT_ROW_MAJOR>\
        <<<batch, TPB, 0, stream>>>(p);\
    } else if (layout == MATRIX_LAYOUT_COL_MAJOR) {\
        minmax_requantize_swiglu_fp16_kernel<TPB, MATRIX_LAYOUT_COL_MAJOR>\
        <<<batch, TPB, 0, stream>>>(p);\
    } else if (layout == MATRIX_LAYOUT_COL32) {\
        minmax_requantize_swiglu_fp16_kernel<TPB, MATRIX_LAYOUT_COL32>\
        <<<batch, TPB, 0, stream>>>(p);\
    } else if (layout == MATRIX_LAYOUT_COL32_2R_4R4) {\
        minmax_requantize_swiglu_fp16_kernel<TPB, MATRIX_LAYOUT_COL32_2R_4R4>\
        <<<batch, TPB, 0, stream>>>(p);\
    } else {\
        LOG(ERROR) << "unsupported matrix layout: " << (int32_t)layout;\
        return ppl::common::RC_UNSUPPORTED;\
    }\
} while (0)

ppl::common::RetCode minmax_requantize_swiglu_fp16(
    cudaStream_t stream,
    const void* input,    //  col32 layout [batch, quant_dim(channel)] or [M, N]
    const void* input_scale_per_batch,
    const void* input_scale_per_channel,
    const int64_t batch,
    const int64_t quant_dim,
    const float beta,
    const matrix_layout_t layout,
    const float up_scale,
    const float down_scale_batch, // batch_scale_val = batch_scale[i] * down_scale_batch for precision
    const float down_scale_channel, // channel_scale_val = channel_scale[i] * down_scale_channel for precision
    void* workspace, // [batch, quant_dim(channel)] or [M, N]
    void* output, // [batch, quant_dim(channel)] or [M, N]
    void* scale // [batch] or [M]
) {
    minmax_requantize_swiglu_fp16_kernel_param p = {
        (int32_t*) input,    //  col32 layout [batch, quant_dim(channel)] or [M, N]
        (half*) input_scale_per_batch,
        (half*) input_scale_per_channel,
        batch,
        quant_dim,
        beta,
        up_scale,
        down_scale_batch, // batch_scale_val = batch_scale[i] * down_scale_batch for precision
        down_scale_channel, // channel_scale_val = channel_scale[i] * down_scale_channel for precision
        (half*) workspace,
        (int8_t*) output,
        (half*) scale
    };

    const int TPB = 256;
    
    KERNEL_BLOCK();

    return ppl::common::RC_SUCCESS;
}

#undef KERNEL_BLOCK

}}}}}}
