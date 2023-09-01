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

#ifndef __PPLCUDA_PARALLEL_EMBEDDING_H_
#define __PPLCUDA_PARALLEL_EMBEDDING_H_

#include <stdexcept>
#include <cuda_runtime.h>
#include "ppl/common/tensor_shape.h"
#include "ppl/common/types.h"
#include "ppl/common/retcode.h"
#include "ppl/common/cuda/nccl_utils.h"


ppl::common::RetCode PPLCUDAParallelEmbeddingForwardImp(
            const cudaStream_t stream,
            ppl::common::NcclParam* nccl_param,
            const ppl::common::TensorShape* indices_shape,
            const void* indices,
            const ppl::common::TensorShape* weight_shape,
            const void* weight,
            const ppl::common::TensorShape* output_shape,
            void* output,
            const float max_norm,
            const float norm_type,
            void* gather_buffer);


#endif // __PPLCUDA_IMPLICITMATMUL_MATMUL_INTERNAL_CUBLAS_H_
