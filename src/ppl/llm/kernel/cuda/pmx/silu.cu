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

#include "ppl/kernel/llm/cuda/pmx/silu.h"
#include "ppl/common/log.h"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

template<bool GATED>
__global__ void silu_kernel_fp16(
    const half *input,
    const half *gate,
    const int64_t num_elem,
    half *output)
{
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= num_elem)
        return;

    auto val = __half2float(input[index]);
    if (GATED) {
        auto gate_val = gate[index];
        output[index] = __float2half(val / (1.f + __expf(-val))) * gate_val;
    } else {
        output[index] = __float2half(val / (1.f + __expf(-val)));
    }
}


template<bool GATED>
__global__ void silu_kernel_packed_fp16(
    const half2 *input,
    const half2 *gate,
    const int64_t num_elem,
    half2 *output)
{
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= num_elem)
        return;

    auto val = __half22float2(input[index]);
    if (GATED) {
        auto gate_val = gate[index];
        output[index] = {
            __float2half(val.x / (1.f + __expf(-val.x))) * gate_val.x,
            __float2half(val.y / (1.f + __expf(-val.y))) * gate_val.y,
        };
    } else {
        output[index] = {
            __float2half(val.x / (1.f + __expf(-val.x))),
            __float2half(val.y / (1.f + __expf(-val.y)))
        };
    }
}

ppl::common::RetCode silu(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const void* optional_gate,
    void* output)
{
    if (input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "currently only support fp16.";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t TPB = 256;
    const int64_t num_elem = input_shape->CalcElementsIncludingPadding();

    if (num_elem & 1) {
        const int64_t BPG = (num_elem + TPB - 1) / TPB;
        if (optional_gate) {
            silu_kernel_fp16<true><<<BPG, TPB, 0, stream>>>(
                (const half*)input, (const half*)optional_gate, num_elem, (half*)output);
        } else {
            silu_kernel_fp16<false><<<BPG, TPB, 0, stream>>>(
                (const half*)input, (const half*)optional_gate, num_elem, (half*)output);
        }
    } else {
        const int64_t BPG = ((num_elem >> 1) + TPB - 1) / TPB;
        if (optional_gate) {
            silu_kernel_packed_fp16<true><<<BPG, TPB, 0, stream>>>(
                (const half2*)input, (const half2*)optional_gate, num_elem >> 1, (half2*)output);
        } else {
            silu_kernel_packed_fp16<false><<<BPG, TPB, 0, stream>>>(
                (const half2*)input, (const half2*)optional_gate, num_elem >> 1, (half2*)output);
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}}}
