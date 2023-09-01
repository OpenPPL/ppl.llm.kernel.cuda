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

#ifndef PPLCUDA_KERNEL_INCLUDE_KEY_VALUE_CACHE_H_
#define PPLCUDA_KERNEL_INCLUDE_KEY_VALUE_CACHE_H_

#include "ppl/common/tensor_shape.h"
#include "ppl/common/retcode.h"
#include <cuda_runtime.h>

ppl::common::RetCode PPLCUDAKeyValueCacheForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* key_shape,
    const void* key,
    const void* value,
    const int64_t start_pos,
    const int64_t num_layer,
    const int64_t layer_idx,
    ppl::common::TensorShape* kvcache_shape,
    void* kvcache,
    void* kvscale,
    void* key_out,
    void* value_out);

ppl::common::RetCode PPLCUDAKeyValueCacheDBForwardImp(
    cudaStream_t stream,
    ppl::common::TensorShape* input_kv_shape,
    const void* current_key, // (seqlen,h,d)
    const void* current_value,
    const void* seqstarts,//(b+1)
    const void* kvstarts,
    const void* cachestarts,//(b)
    ppl::common::TensorShape* start_pos_shape,
    const void* start_pos,
    void* cache, //(max_token,L,2,H,D)
    void* scale, //(max_token,L,2,H,D/g)
    const int64_t layer_idx,
    const int64_t num_layer,
    ppl::common::TensorShape* output_kv_shape,
    void* key,   //[sum(start_pos) + seqlen, h, d]
    void* value);

#endif // #define PPLCUDA_KERNEL_INCLUDE_LAYERNORM_H_
