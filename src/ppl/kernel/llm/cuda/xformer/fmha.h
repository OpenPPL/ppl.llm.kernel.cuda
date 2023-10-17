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

#ifndef __PPL_KERNEL_LLM_CUDA_XFORMER_FMHA_H__
#define __PPL_KERNEL_LLM_CUDA_XFORMER_FMHA_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"
#include "utils/kernel_forward.h"

#include <cuda_runtime.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace xformer {

ppl::common::RetCode fmha(
    const cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    const ppl::common::datatype_t datatype,
    const void* query,
    const void* key,
    const void* value,
    const void* optional_attn_mask,
    const void* optional_seqstart_q,
    const void* optional_seqstart_k,
    const int64_t batch,
    const int64_t query_stride_b,
    const int64_t query_stride_s,
    const int64_t query_stride_h,
    const int64_t key_stride_b,
    const int64_t key_stride_s,
    const int64_t key_stride_h,
    const int64_t value_stride_b,
    const int64_t value_stride_s,
    const int64_t value_stride_h,
    const int64_t mask_stride_b,
    const int64_t mask_stride_s,
    const int64_t mask_stride_h,
    const int64_t output_stride_s,
    const int64_t max_seqlen,
    const int64_t max_kvlen,
    const int64_t num_heads,
    const int64_t num_kv_heads,
    const int64_t head_dim,
    const int64_t custom_mask_type,
    const double attn_scale,
    void* output);

}}}}}

#endif // __PPL_KERNEL_LLM_CUDA_XFORMER_FMHA_H__
