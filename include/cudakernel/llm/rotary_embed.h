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

#ifndef PPLCUDA_KERNEL_INCLUDE_ROTARYEMB_H_
#define PPLCUDA_KERNEL_INCLUDE_ROTARYEMB_H_

#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDARotaryEmbQKForwardImp(
    cudaStream_t stream,
    const void* input_q,
    const void* input_k,
    const float theta,
    const void* cu_start_pos,
    const int64_t start_pos_val,
    const int32_t type, // 0 for llama, 1 for palm
    const int32_t bypass_k,
    ppl::common::TensorShape* input_shape,
    void* output_q,
    void* output_k);

ppl::common::RetCode PPLCUDARotaryEmbQKDynamicBatchForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_shape,
    const void* input_q, // (S,H,D)
    const void* input_k,
    const void* seqstart_q,
    const float theta,
    ppl::common::TensorShape* start_pos_shape,
    const void* start_pos,
    const int32_t type, // 0 for llama, 1 for palm
    const int32_t bypass_k,
    void* output_q,
    void* output_k,
    const int64_t max_seqlen);

#endif // #define PPLCUDA_KERNEL_INCLUDE_PRELU_H_
