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

#ifndef __PPLCUDA_MULTI_HEAD_ATTENTION_H_
#define __PPLCUDA_MULTI_HEAD_ATTENTION_H_

#include <stdexcept>
#include <cuda_runtime.h>
#include "ppl/common/types.h"
#include "ppl/common/retcode.h"


ppl::common::RetCode PPLCUDAMultiHeadAttentionForwardImp(
            const cudaStream_t stream,
            const cudaDeviceProp& device_prop,
            const ppl::common::TensorShape* query_shape,
            void* query,
            const ppl::common::TensorShape* key_shape,
            void* key,
            void* value,
            const ppl::common::TensorShape* mask_shape,
            const void* mask,
            const ppl::common::TensorShape* output_shape,
            void* output);

ppl::common::RetCode PPLCUDAMultiHeadAttentionDBForwardImp(
            const cudaStream_t stream,
            const cudaDeviceProp& device_prop,
            ppl::common::TensorShape* query_shape, //(S,H,D)
            void* query,
            ppl::common::TensorShape* key_shape,
            void* key,
            void* value,
            ppl::common::TensorShape* seqstart_q_shape,
            void* seqstart_q,
            void* seqstart_k,
            int64_t decoding_batches,
            int64_t max_seqlen,
            int64_t max_kvlen,
            ppl::common::TensorShape* output_shape,
            void* output);


#endif // __PPLCUDA_IMPLICITMATMUL_MATMUL_INTERNAL_CUBLAS_H_
