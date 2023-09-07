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

#include "ppl/kernel/llm/cuda/pmx/column_parallel_linear.h"
#include "ppl/common/log.h"

#include "cudakernel/common/common.cuh"
#include "cudakernel/memory/transpose.h"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

template<int VPT, int TPB>
__global__ 
void embedding_kernel(
    const int64_t* indices,
    const half* weight,
    half* output,
    const int64_t embedding_dim)
{
    int64_t index = indices[blockIdx.x];
    int64_t weight_idx = index * embedding_dim + threadIdx.x * VPT;
    int64_t output_idx = blockIdx.x * embedding_dim + threadIdx.x * VPT;
    copy<sizeof(half) * VPT>(&weight[weight_idx], &output[output_idx]);
}

ppl::common::RetCode parallel_embedding(
    const cudaStream_t stream,
    const ppl::common::TensorShape* indices_shape,
    const void* indices,
    const ppl::common::TensorShape* weight_shape,
    const void* weight,
    const int64_t num_embeddings,
    const int64_t embedding_dim,
    const float max_norm,
    const float norm_type,
    const int64_t padding_idx,
    ppl::common::NcclParam* nccl_param,
    void* gather_buffer,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    if (max_norm != 0.0f) {
        LOG(ERROR) << "currently do not support max_norm.";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (padding_idx != -1) {
        LOG(ERROR) << "currently do not support padding_idx.";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t num_indices = indices_shape->CalcElementsIncludingPadding();
    constexpr int32_t VPT = 16 / sizeof(half);

    void* kernel_output = output;
    const int64_t Ew = embedding_dim / nccl_param->size;

    if (nccl_param->size > 1) {
        kernel_output = (char*)gather_buffer
            + nccl_param->rank * num_indices * Ew * ppl::common::GetSizeOfDataType(output_shape->GetDataType());
    }

    switch (Ew) {
    case 4096:
        embedding_kernel<VPT, 4096 / VPT><<<num_indices, 4096 / VPT, 0, stream>>>(
            (int64_t*)indices,
            (half*)weight,
            (half*)kernel_output,
            Ew);
        break;
    case 2560:
        embedding_kernel<VPT, 2560 / VPT><<<num_indices, 2560 / VPT, 0, stream>>>(
            (int64_t*)indices,
            (half*)weight,
            (half*)kernel_output,
            Ew);
        break;
    case 1024:
        embedding_kernel<VPT, 1024 / VPT><<<num_indices, 1024 / VPT, 0, stream>>>(
            (int64_t*)indices,
            (half*)weight,
            (half*)kernel_output,
            Ew);
        break;
    default:
        LOG(ERROR) << "currently do not support embedding_dim " << embedding_dim;
        return ppl::common::RC_UNSUPPORTED;
    }

    if (nccl_param->size > 1) {
        auto status = ppl::common::NcclAllGather<half>(
          (half*)kernel_output,
          (half*)gather_buffer,
          num_indices * Ew,
          nccl_param,
          stream);
        if (ppl::common::RC_SUCCESS != status)
            return status;

        status = PPLCUDATranspose01ForwardImp(
            stream,
            gather_buffer,
            output_shape->GetDataType(),
            nccl_param->size,
            num_indices,
            Ew,
            output);

        if (ppl::common::RC_SUCCESS != status)
            return status;
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}
