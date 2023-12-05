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
    return ppl::common::RC_UNSUPPORTED;
}

}}}}}}

