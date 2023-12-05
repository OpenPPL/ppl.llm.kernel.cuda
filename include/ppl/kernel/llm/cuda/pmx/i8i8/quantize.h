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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_I8I8_QUANTIZE_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_I8I8_QUANTIZE_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"
#include "ppl/kernel/llm/cuda/common/matrix_layout.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i8i8 {

constexpr float token_up_scale = 4096.0f;
constexpr float hidden_up_scale = 8192.0f;

constexpr float token_down_scale = 1.0f / 4096.0f;
constexpr float hidden_down_scale = 1.0f / 8192.0f;

ppl::common::RetCode minmax_quantize_fp16(
    cudaStream_t stream,
    const void* input, // fp16, [batch, quant_dim]
    const int64_t batch,
    const int64_t quant_dim,
    const float up_scale, // scale[i] = scale_val * up_scale for precision
    const matrix_layout_t to_layout,
    void* quantized, // int8, [batch, quant_dim]
    void* scale // fp16, [batch]
);

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
);

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
);

ppl::common::RetCode minmax_dequantize_silu_fp16(
    cudaStream_t stream,
    const void* input,    //  col32 layout [batch, quant_dim(channel)] or [M, N]
    const void* input_scale_per_batch,
    const void* input_scale_per_channel,
    const void* optional_gate,
    const void* gate_scale_per_batch,
    const void* gate_scale_per_channel,
    const int64_t batch,
    const int64_t quant_dim,
    const matrix_layout_t from_layout,
    float down_scale_batch, // batch_scale_val = batch_scale[i] * down_scale_batch for precision
    float down_scale_channel, // channel_scale_val = channel_scale[i] * down_scale_channel for precision
    void* output
);

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
);

ppl::common::RetCode skip_rmsnorm_minmax_quantize_fp16(
  const cudaStream_t stream,
  const void* input,           // [batch, hidden dim]
  const void* weight,           // [hidden dim]
  const void* skip,             // [batch, hidden dim]
  const float eps,
  const int64_t batch,
  const int64_t normalize_shape,
  const float up_scale, // scale[i] = scale * up_scale for precision
  const matrix_layout_t to_layout,
  void* skip_out,               // [batch, hidden dim]
  void* output_scale,           // [batch]
  void* output                  // [batch, hidden dim]
);

}}}}}}


#endif
