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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_I4F16_GEMM_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_I4F16_GEMM_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

void* create_gemm_handle();
void destory_gemm_handle(void*);

ppl::common::RetCode gemm(
    const cudaStream_t stream,
    const void* handle,
    const void* input, // fp16
    const void* weight, // int4x8
    const void* weight_scale, // fp16
    const void* bias, // fp16
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t workspace_size,
    void* workspace, // suggest 128MB
    void* output); // fp16

}}}}}}

#endif
