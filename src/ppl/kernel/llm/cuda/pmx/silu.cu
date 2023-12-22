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

#include "type.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

template<typename T, bool GATED>
__global__ void silu_kernel(
    const T *input,
    const T *gate,
    const int64_t num_elem,
    T *output)
{
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= num_elem)
        return;

    auto val = tofp32<T>(input[index]);
    if (GATED) {
        auto gate_val = gate[index];
        output[index] = fromfp32<T>(val / (1.f + __expf(-val))) * gate_val;
    } else {
        output[index] = fromfp32<T>(val / (1.f + __expf(-val)));
    }
}


template<typename Tx2, bool GATED>
__global__ void silu_kernel_packed(
    const Tx2 *input,
    const Tx2 *gate,
    const int64_t num_elem,
    Tx2 *output)
{
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= num_elem)
        return;

    using T = typename FromType2<Tx2>::type;

    auto val = tofp32x2<Tx2>(input[index]);
    if (GATED) {
        auto gate_val = gate[index];
        output[index] = {
            fromfp32<T>(val.x / (1.f + __expf(-val.x))) * gate_val.x,
            fromfp32<T>(val.y / (1.f + __expf(-val.y))) * gate_val.y,
        };
    } else {
        output[index] = {
            fromfp32<T>(val.x / (1.f + __expf(-val.x))),
            fromfp32<T>(val.y / (1.f + __expf(-val.y)))
        };
    }
}


template<typename T>
ppl::common::RetCode silu_impl(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const void* optional_gate,
    void* output)
{
    const int64_t TPB = 256;
    const int64_t num_elem = input_shape->CalcElementsIncludingPadding();

    if (num_elem & 1) {
        const int64_t BPG = (num_elem + TPB - 1) / TPB;
        if (optional_gate) {
            silu_kernel<T, true><<<BPG, TPB, 0, stream>>>(
                (const T*)input, (const T*)optional_gate, num_elem, (T*)output);
        } else {
            silu_kernel<T, false><<<BPG, TPB, 0, stream>>>(
                (const T*)input, (const T*)optional_gate, num_elem, (T*)output);
        }
    } else {
        using Tx2 = typename ToType2<T>::type;
        const int64_t BPG = ((num_elem >> 1) + TPB - 1) / TPB;
        if (optional_gate) {
            silu_kernel_packed<Tx2, true><<<BPG, TPB, 0, stream>>>(
                (const Tx2*)input, (const Tx2*)optional_gate, num_elem >> 1, (Tx2*)output);
        } else {
            silu_kernel_packed<Tx2, false><<<BPG, TPB, 0, stream>>>(
                (const Tx2*)input, (const Tx2*)optional_gate, num_elem >> 1, (Tx2*)output);
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode silu(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const void* optional_gate,
    void* output)
{
    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        return silu_impl<fp16_t>(stream, input_shape, input, optional_gate, output);
    }
    if (input_shape->GetDataType() == ppl::common::DATATYPE_BFLOAT16) {
        return silu_impl<bf16_t>(stream, input_shape, input, optional_gate, output);
    }

    LOG(ERROR) << "currently only support fp16 & bf16";
    return ppl::common::RC_UNSUPPORTED;
}

}}}}}
