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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_SILU_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_SILU_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

template <typename T>
ppl::common::RetCode silu(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const void* optional_gate,
    void* output);

}}}}}

#endif
