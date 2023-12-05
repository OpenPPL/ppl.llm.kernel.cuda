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

struct minmax_requantize_silu_fp16_kernel_param {
    int32_t* input;    //  [batch, quant_dim(channel)] or [M, N]
    half* input_scale_per_batch;
    half* input_scale_per_channel;
    int32_t* optional_gate;
    half* gate_scale_per_batch;
    half* gate_scale_per_channel;
    int64_t batch;
    int64_t quant_dim;
    float up_scale;
    float down_scale_batch; // batch_scale_val = batch_scale[i] * down_scale_batch for precision
    float down_scale_channel; // channel_scale_val = channel_scale[i] * down_scale_channel for precision
    half* workspace; // [batch, quant_dim(channel)] or [M, N]
    int8_t* output;
    half* scale;
};

template<int32_t TPB, bool GATED, matrix_layout_t LAYOUT>
__global__
void minmax_requantize_silu_fp16_kernel(
    minmax_requantize_silu_fp16_kernel_param p
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

    MatrixLayoutHelper<LAYOUT> idx_hlp;
    idx_hlp.Init(p.batch, p.quant_dim);

    const float batch_scale = __half2float(p.input_scale_per_batch[batch_id]) * p.down_scale_batch;
    const float gate_scale = GATED ? __half2float(p.gate_scale_per_batch[batch_id]) * p.down_scale_batch : 0;

    float local_max = 0.0f;
    // TODO
    if (LAYOUT == MATRIX_LAYOUT_COL4_4R2_8C) {

    } else {
        for (int64_t i = threadIdx.x * V16PT; i < p.quant_dim; i += TPB * V16PT) {
            int32_t local_in[V16PT];
            half local_channel_scale[V16PT];
            int32_t local_optional_gate[V16PT];
            half local_gate_channel_scale[V16PT];
            half local_dq_in[V16PT];

            const auto offset = idx_hlp.GetOffset(batch_id, i);
            copy<VPT>(&p.input[offset], local_in);
            copy<VPT>(&p.input[offset + V32PT], &local_in[V32PT]);
            copy<VPT>(&p.input_scale_per_channel[i], local_channel_scale);
            if (GATED) {
                copy<VPT>(&p.optional_gate[offset], local_optional_gate);
                copy<VPT>(&p.optional_gate[offset + V32PT], &local_optional_gate[V32PT]);
                copy<VPT>(&p.gate_scale_per_channel[i], local_gate_channel_scale);
            }

            #pragma unroll
            for (int32_t vidx = 0; vidx < V16PT; vidx += 1) {
                auto dq_input = local_in[vidx] * __half2float(local_channel_scale[vidx]) * p.down_scale_channel * batch_scale;

                // silu
                dq_input = dq_input / (1.f + __expf(-dq_input));
                // gate
                if (GATED) {
                    auto dq_gate = local_optional_gate[vidx] * __half2float(local_gate_channel_scale[vidx]) * p.down_scale_channel * gate_scale;
                    dq_input *= dq_gate;
                }
                local_dq_in[vidx] = __float2half(dq_input);
                local_max = fmax(fabs(dq_input), local_max);
            }

            copy<VPT>(local_dq_in, &p.workspace[offset]);
        }

        local_max = block_reduce_max<WPT>(local_max, red_smem);
        float scale = min_max_range_to_scale(local_max, INT8_QLEVEL);
        if(threadIdx.x == 0){
            p.scale[batch_id] = __float2half(scale * p.up_scale);
        }

        for (int64_t i = threadIdx.x * V8PT; i < p.quant_dim; i += TPB * V8PT) {
            half local_dq_in[V8PT];
            int8_t local_out[V8PT];

            const auto offset = idx_hlp.GetOffset(batch_id, i);
            copy<VPT>(&p.workspace[offset], local_dq_in);
            copy<VPT>(&p.workspace[offset + V16PT], &local_dq_in[V16PT]);

            #pragma unroll
            for (int32_t vidx = 0; vidx < V8PT; vidx += 1) {
                local_out[vidx] = (int8_t)__float2int_rn(__half2float(local_dq_in[vidx]) / scale);
            }

            copy<VPT>(local_out, &p.output[offset]);
        }
    }
}

#define KERNEL_BLOCK(GATED) do {\
    if (layout == MATRIX_LAYOUT_ROW_MAJOR) {\
        minmax_requantize_silu_fp16_kernel<TPB, GATED, MATRIX_LAYOUT_ROW_MAJOR>\
        <<<batch, TPB, 0, stream>>>(p);\
    } else if (layout == MATRIX_LAYOUT_COL_MAJOR) {\
        minmax_requantize_silu_fp16_kernel<TPB, GATED, MATRIX_LAYOUT_COL_MAJOR>\
        <<<batch, TPB, 0, stream>>>(p);\
    } else if (layout == MATRIX_LAYOUT_COL32) {\
        minmax_requantize_silu_fp16_kernel<TPB, GATED, MATRIX_LAYOUT_COL32>\
        <<<batch, TPB, 0, stream>>>(p);\
    } else if (layout == MATRIX_LAYOUT_COL32_2R_4R4) {\
        minmax_requantize_silu_fp16_kernel<TPB, GATED, MATRIX_LAYOUT_COL32_2R_4R4>\
        <<<batch, TPB, 0, stream>>>(p);\
    } else {\
        LOG(ERROR) << "unsupported matrix layout: " << (int32_t)layout;\
        return ppl::common::RC_UNSUPPORTED;\
    }\
} while (0)

ppl::common::RetCode minmax_requantize_silu_fp16(
    cudaStream_t stream,
    const void* input,    //  col32 layout [batch, quant_dim(channel)] or [M, N]
    const void* input_scale_per_batch,
    const void* input_scale_per_channel,
    const void* optional_gate,
    const void* gate_scale_per_batch,
    const void* gate_scale_per_channel,
    const int64_t batch,
    const int64_t quant_dim,
    const matrix_layout_t layout,
    const float up_scale,
    const float down_scale_batch, // batch_scale_val = batch_scale[i] * down_scale_batch for precision
    const float down_scale_channel, // channel_scale_val = channel_scale[i] * down_scale_channel for precision
    void* workspace, // [batch, quant_dim(channel)] or [M, N]
    void* output, // [batch, quant_dim(channel)] or [M, N]
    void* scale // [batch] or [M]
) {
    minmax_requantize_silu_fp16_kernel_param p = {
        (int32_t*) input,    //  col32 layout [batch, quant_dim(channel)] or [M, N]
        (half*) input_scale_per_batch,
        (half*) input_scale_per_channel,
        (int32_t*) optional_gate,
        (half*) gate_scale_per_batch,
        (half*) gate_scale_per_channel,
        batch,
        quant_dim,
        up_scale,
        down_scale_batch, // batch_scale_val = batch_scale[i] * down_scale_batch for precision
        down_scale_channel, // channel_scale_val = channel_scale[i] * down_scale_channel for precision
        (half*) workspace,
        (int8_t*) output,
        (half*) scale
    };

    bool gated = optional_gate != nullptr;
    const int TPB = 256;

    if (gated) {
        KERNEL_BLOCK(true);
    } else {
        KERNEL_BLOCK(false);
    }

    return ppl::common::RC_SUCCESS;
}

#undef KERNEL_BLOCK

}}}}}}
