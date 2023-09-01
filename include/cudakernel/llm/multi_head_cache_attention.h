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

#ifndef __PPLCUDA_MULTI_HEAD_CACHE_ATTENTION_H_
#define __PPLCUDA_MULTI_HEAD_CACHE_ATTENTION_H_

#include <stdexcept>
#include <cuda_runtime.h>
#include "ppl/common/types.h"
#include "ppl/common/retcode.h"

ppl::common::RetCode PPLCUDAMultiHeadCacheAttentionForwardImp(
            const cudaStream_t stream,
            const cudaDeviceProp& device_prop,
            ppl::common::TensorShape* query_shape, //(S,H,D)
            void* query,
            ppl::common::TensorShape* key_shape,
            void* key,
            void* value,
            ppl::common::TensorShape* seqstart_q_shape,
            const void* seqstart_q,
            const void* seqstart_k,
            const void* start_pos,
            void* cache,
            void* scale,
            const void* cachestarts,
            const int64_t decoding_batches,
            const int64_t max_seqlen,
            const int64_t max_kvlen,
            const int64_t layer_idx,
            const int64_t num_layer,
            ppl::common::TensorShape* output_shape,
            void* output);


#endif // __PPLCUDA_IMPLICITMATMUL_MATMUL_INTERNAL_CUBLAS_H_
