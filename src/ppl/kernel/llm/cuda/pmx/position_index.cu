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

#include "ppl/common/log.h"

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

template<int32_t TPB>
__global__
void dynamic_batching_position_index_kernel(
    const int64_t* start_pos,
    const int64_t* seqstarts,
    int64_t* output)
{
    const int64_t batch_idx = blockIdx.y;
    const int64_t seq_idx = blockIdx.x * TPB + threadIdx.x;

    if (seq_idx < seqstarts[batch_idx + 1] - seqstarts[batch_idx]) {
        const int64_t token_idx = seqstarts[batch_idx] + seq_idx;
        const int64_t pos_idx = seq_idx + start_pos[batch_idx];
        output[token_idx] = pos_idx;
    }
}

ppl::common::RetCode dynamic_batching_position_index(
    const cudaStream_t stream,
    const void* start_pos,
    const void* seqstarts,
    const int64_t batch,
    const int64_t max_seqlen,
    void* output)
{
    constexpr int32_t TPB = 256;

    const dim3 grid_size = {
        (unsigned int)((max_seqlen + TPB - 1) / TPB * TPB),
        (unsigned int)batch,
        1,
    };

    dynamic_batching_position_index_kernel<TPB><<<grid_size, TPB, 0, stream>>>(
        (int64_t*)start_pos,
        (int64_t*)seqstarts,
        (int64_t*)output);

    return ppl::common::RC_SUCCESS;
}

}}}}}
