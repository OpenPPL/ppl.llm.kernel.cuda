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

#include "ppl/kernel/llm/cuda/pmx/i4f16/gemm.h"

#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

void* create_gemm_handle() {
    return nullptr;
}

void destory_gemm_handle(void* handle) {
}

ppl::common::RetCode gemm(
    const cudaStream_t stream,
    const void* handle,
    const void* input,
    const void* weight,
    const void* weight_scale,
    const void* bias,
    const int64_t M,
    const int64_t N,
    const int64_t K,
    const int64_t workspace_size,
    void* workspace,
    void* output)
{
    return ppl::common::RC_UNSUPPORTED;
}

}}}}}}
