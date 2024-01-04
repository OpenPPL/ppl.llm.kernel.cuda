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

#include "ppl/kernel/llm/cuda/pmx/gelu.h"
#include "ppl/common/log.h"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

template<bool GATED, bool APPROXIMATE>
__global__ void gelu_kernel_fp16(
    const half *input, 
    const half *gate, 
    const int64_t count, 
    half *output
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count)  return;
    auto val = __half2float(input[index]);

    float out_val = 0.f;
    if (APPROXIMATE) {
        out_val = val * 0.5f * (1.f + tanh(0.7978845608028654f * val * (1.0f + 0.044715f * val * val)));
    } else {
        out_val = val * 0.5f * (1.f + erff(val * 0.707106781f));
    }
    if (GATED) {
        auto gate_val = gate[index];
        output[index] = __float2half(out_val) * gate_val;
    } else {
        output[index] = __float2half(out_val);
    }
}

template<bool GATED, bool APPROXIMATE>
__global__ void gelu_kernel_packed_fp16(const half2 *input, const half2 *gate, const int64_t count, half2 *output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= count) return;
    auto h_val = input[index];
    auto f_val = __half22float2(input[index]);

    half2 t_val;
    if (APPROXIMATE) {
        t_val.x = __float2half(tanh(0.7978845608028654f * f_val.x * (1.0f + 0.044715f * f_val.x * f_val.x)));
        t_val.y = __float2half(tanh(0.7978845608028654f * f_val.y * (1.0f + 0.044715f * f_val.y * f_val.y)));
    } else {
        t_val.x = __float2half(erff(f_val.x * 0.707106781f));
        t_val.y = __float2half(erff(f_val.y * 0.707106781f));
    }
    
    half2 one_constant = {__float2half(1.f),  __float2half(1.f)};
    half2 half_constant = {__float2half(0.5f),  __float2half(0.5f)};
    t_val = __hmul2(half_constant, __hmul2(h_val, __hadd2(one_constant, t_val)));

    if (GATED) {
        half2 gate_val = gate[index];
        output[index] = {
            t_val.x * gate_val.x,
            t_val.y * gate_val.y,
        };
    } else {
        output[index] = {
            t_val.x,
            t_val.y,
        };
    }
}

ppl::common::RetCode gelu(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const void* optional_gate,
    const bool approximate,
    void* output)
{
    if (input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "gelu only support fp16, but got ["<< input_shape->GetDataType() << "]";
        return ppl::common::RC_UNSUPPORTED;
    }

    const int64_t TPB = 256;    // thread_per_block
    const int64_t num_elem = input_shape->CalcElementsIncludingPadding();

    if (num_elem & 1) {
        const int64_t BPG = (num_elem + TPB - 1) / TPB; // block_per_grid
        if (optional_gate && approximate) {
            gelu_kernel_fp16<true, true><<<BPG, TPB, 0, stream>>>(
                (const half*)input, (const half*)optional_gate, num_elem, (half*)output);
        } else if(optional_gate && !approximate) {
            gelu_kernel_fp16<true, false><<<BPG, TPB, 0, stream>>>(
                (const half*)input, (const half*)optional_gate, num_elem, (half*)output);
        } else if (!optional_gate && approximate) {
            gelu_kernel_fp16<false, true><<<BPG, TPB, 0, stream>>>(
                (const half*)input, (const half*)optional_gate, num_elem, (half*)output);
        } else {
            gelu_kernel_fp16<false, false><<<BPG, TPB, 0, stream>>>(
                (const half*)input, (const half*)optional_gate, num_elem, (half*)output);
        }

    } else {
        const int64_t BPG = ((num_elem >> 1) + TPB - 1) / TPB;
        if (optional_gate && approximate) {
            gelu_kernel_packed_fp16<true, true><<<BPG, TPB, 0, stream>>>(
                (const half2*)input, (const half2*)optional_gate, num_elem, (half2*)output);
        } else if(optional_gate && !approximate) {
            gelu_kernel_packed_fp16<true, false><<<BPG, TPB, 0, stream>>>(
                (const half2*)input, (const half2*)optional_gate, num_elem, (half2*)output);
        } else if (!optional_gate && approximate) {
            gelu_kernel_packed_fp16<false, true><<<BPG, TPB, 0, stream>>>(
                (const half2*)input, (const half2*)optional_gate, num_elem, (half2*)output);
        } else {
            gelu_kernel_packed_fp16<false, false><<<BPG, TPB, 0, stream>>>(
                (const half2*)input, (const half2*)optional_gate, num_elem, (half2*)output);
        }
    }

    return ppl::common::RC_SUCCESS;
}


}}}}}
