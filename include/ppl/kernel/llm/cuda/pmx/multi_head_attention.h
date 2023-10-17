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

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_MULTI_HEAD_ATTENTION_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_MULTI_HEAD_ATTENTION_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode multi_head_attention(
    const cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    const ppl::common::TensorShape* query_shape,
    const void* query, // (B, Sq, Hq ,D)
    const ppl::common::TensorShape* key_shape,
    const void* key, // (B, Skv, Hkv ,D)
    const ppl::common::TensorShape* value_shape,
    const void* value, // (B, Skv, Hkv ,D)
    const ppl::common::TensorShape* attn_mask_shape,
    const void* attn_mask, // (Sq, Skv), (B, Hq, Sq, Skv)
    const bool is_causal,
    const int64_t num_heads,
    const int64_t num_kv_heads,
    const int64_t head_dim,
    const ppl::common::TensorShape* output_shape,
    void* output); // (B, Sq, Hq ,D)

}}}}}

#endif
