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
    return ppl::common::RC_UNSUPPORTED;
}

#undef KERNEL_BLOCK

}}}}}}
