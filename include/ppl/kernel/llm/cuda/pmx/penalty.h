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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_PENALTY_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_PENALTY_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode apply_penalty(
    cudaStream_t stream,
    const float* logits_in,
    const float* temperatures,
    const float* repetition_penalties,
    const float* presence_penalties,
    const float* frequency_penalties,
    const int64_t* batch_slots,
    const int64_t* token_inputs,
    const int64_t* seqstarts,
    const int64_t* start_pos,
    int32_t batch,
    int32_t vocab_size,
    uint16_t* penalty_count_map,
    float* logits_out
);

}}}}}

#endif