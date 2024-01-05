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

#include "ppl/kernel/llm/cuda/pmx/swiglu.h"
#include "ppl/common/log.h"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

__global__ void swiglu_kernel_fp16(
    const half *input,
    const int64_t batch,
    const int64_t num_elem,
    const float beta,
    half *output)
{
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= batch * num_elem)
        return;

    const int64_t b = index / num_elem;
    const int64_t i = index % num_elem;

    auto val = __half2float(input[(b * 2) * num_elem + i]);
    auto gate_val = input[(b * 2 + 1) * num_elem + i];
    output[index] = __float2half(val / (1.f + __expf(-val * beta))) * gate_val;
}


__global__ void swiglu_kernel_packed_fp16(
    const half2 *input,
    const int64_t batch,
    const int64_t num_elem,
    const float beta,
    half2 *output)
{
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= batch * num_elem)
        return;

    const int64_t b = index / num_elem;
    const int64_t i = index % num_elem;

    auto val = __half22float2(input[(b * 2) * num_elem + i]);
    auto gate_val = input[(b * 2 + 1) * num_elem + i];
    output[index] = {
        __float2half(val.x / (1.f + __expf(-val.x * beta))) * gate_val.x,
        __float2half(val.y / (1.f + __expf(-val.y * beta))) * gate_val.y,
    };
}

ppl::common::RetCode swiglu(
    cudaStream_t stream,
    const void* input,
    const float beta,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    if (output_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "swiglu only support fp16, but got ["<< output_shape->GetDataType() << "]";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t TPB = 256;
    const int64_t num_elem = output_shape->GetDim(output_shape->GetDimCount() - 1);
    const int64_t batch = output_shape->CalcElementsIncludingPadding() / num_elem;

    if (num_elem & 1) {
        const int64_t BPG = ((batch * num_elem) + TPB - 1) / TPB;
        swiglu_kernel_fp16<<<BPG, TPB, 0, stream>>>(
                (const half*)input, batch, num_elem, beta, (half*)output);
    } else {
        const int64_t BPG = (((batch * num_elem) >> 1) + TPB - 1) / TPB;
        swiglu_kernel_packed_fp16<<<BPG, TPB, 0, stream>>>(
                (const half2*)input, batch, num_elem >> 1, beta, (half2*)output);
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}
