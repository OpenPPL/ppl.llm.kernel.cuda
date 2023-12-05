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

#pragma once

#include "utils.h"

#include "ppl/kernel/llm/cuda/common/general_include.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

#define THREAD_SIZE 128

template <int COUNT>
__global__ __launch_bounds__(THREAD_SIZE) void reduce(const __half* __restrict__ buffer, __half* C, int size) {
    const int idx = blockIdx.x * THREAD_SIZE + threadIdx.x;

    if (idx >= size) {
        return;
    }

    int4* result = reinterpret_cast<int4*>(C);
    const int4* spread_result = reinterpret_cast<const int4*>(buffer);
    int4 result_tmp[COUNT] = {0};

#pragma unroll
    for (int i = 0; i < COUNT; i++) {
        result_tmp[i] = spread_result[idx + i * size];
    }

#pragma unroll
    for (int i = 1; i < COUNT; i++) {
        asm volatile("add.f16x2 %0, %0, %1;\n" : "+r"(result_tmp[0].x) : "r"(result_tmp[i].x));
        asm volatile("add.f16x2 %0, %0, %1;\n" : "+r"(result_tmp[0].y) : "r"(result_tmp[i].y));
        asm volatile("add.f16x2 %0, %0, %1;\n" : "+r"(result_tmp[0].w) : "r"(result_tmp[i].w));
        asm volatile("add.f16x2 %0, %0, %1;\n" : "+r"(result_tmp[0].z) : "r"(result_tmp[i].z));
    }

    result[idx] = result_tmp[0];
}

ppl::common::RetCode reduce(const __half* buffer, __half* d, int m, int n, int count, cudaStream_t stream) {
    int size = m * n / 8;
    int grid = (size + THREAD_SIZE - 1) / THREAD_SIZE;
    switch (count) {
        case 1:
            reduce<1><<<grid, THREAD_SIZE, 0, stream>>>(buffer, d, size);
            break;
        case 2:
            reduce<2><<<grid, THREAD_SIZE, 0, stream>>>(buffer, d, size);
            break;
        case 3:
            reduce<3><<<grid, THREAD_SIZE, 0, stream>>>(buffer, d, size);
            break;
        case 4:
            reduce<4><<<grid, THREAD_SIZE, 0, stream>>>(buffer, d, size);
            break;
        case 5:
            reduce<5><<<grid, THREAD_SIZE, 0, stream>>>(buffer, d, size);
            break;
        case 6:
            reduce<6><<<grid, THREAD_SIZE, 0, stream>>>(buffer, d, size);
            break;
        case 7:
            reduce<7><<<grid, THREAD_SIZE, 0, stream>>>(buffer, d, size);
            break;
        case 8:
            reduce<8><<<grid, THREAD_SIZE, 0, stream>>>(buffer, d, size);
            break;
        case 9:
            reduce<9><<<grid, THREAD_SIZE, 0, stream>>>(buffer, d, size);
            break;
        default:
            LOG(ERROR) << "unsupported reduce count: " << count;
            return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16