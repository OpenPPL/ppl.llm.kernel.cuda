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

#include "ppl/kernel/llm/cuda/pmx/penalty.h"
#include "ppl/common/log.h"

#include "cudakernel/common/common.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <cub/cub.cuh>


namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

__device__ 
double atomicAdd(uint16_t* address, uint16_t val) {
    uint16_t old = *address, assumed;
    do {
        assumed = old;
        old = atomicCAS(address, assumed, assumed + val);
    } while (assumed != old);
}

__global__
void apply_penalty_kernal(
    const float* logits_in,             // [batch, vocab_size]
    const float* temperatures,          // [batch]
    const float* repetition_penalties,  // [batch]
    const float* presence_penalties,    // [batch]
    const float* frequency_penalties,   // [batch]
    const int64_t* batch_slots,         // [batch]
    const int64_t* token_inputs,        // [total_tokens]
    const int64_t* seqstarts,           // [batch+1]
    const int64_t* start_pos,           // [batch]
    int32_t vocab_size,
    uint16_t* penalty_count_map,        // [max_batch, vocab_size]
    float* logits_out                   // [batch, vocab_size]
) {
    int32_t batch_idx = blockIdx.x;
    int64_t batch_slot = batch_slots[batch_idx];
    int64_t seqlen = seqstarts[batch_idx + 1] - seqstarts[batch_idx];
    uint16_t* penalty_work_space = penalty_count_map + batch_slot * vocab_size;
    const float* local_logits_in = logits_in + batch_idx * vocab_size;
    float* local_logits_out = logits_out + batch_idx * vocab_size;

    if (start_pos[batch_idx] == 0) {   // prefill stage
        for (int index = threadIdx.x; index < vocab_size; index += blockDim.x) {
            penalty_work_space[index] = 0;
        }
        __syncthreads();
        for (int step = threadIdx.x; step < seqlen; step += blockDim.x) {
            int64_t token = token_inputs[seqstarts[batch_idx] + step];
            atomicAdd(&penalty_work_space[token], 1);
        }
    } else {
        if (threadIdx.x == 0) {
            int64_t token = token_inputs[seqstarts[batch_idx]];
            penalty_work_space[token] += 1;
        }
    }
    __syncthreads();

    float inv_temperature = 1.0f / (temperatures[batch_idx] + 1e-6);
    for (int index = threadIdx.x; index < vocab_size; index += blockDim.x) {
        float logit = local_logits_in[index];
        logit *= inv_temperature;
        uint16_t past_cnt = penalty_work_space[index];
        if (past_cnt > 0) {
            if (repetition_penalties != nullptr) {
                float repetition_penalty = repetition_penalties[batch_idx];
                logit = logit < 0.0f ? logit * repetition_penalty : logit / repetition_penalty;
            }
            if (presence_penalties != nullptr) {
                float presence_penalty = presence_penalties[batch_idx];
                logit -= presence_penalty;
            }
            if (frequency_penalties != nullptr) {
                float frequency_penalty = frequency_penalties[batch_idx];
                logit -= frequency_penalty * past_cnt;
            }
        }
        local_logits_out[index] = logit;
    }
}

ppl::common::RetCode apply_penalty(
    cudaStream_t stream,
    const float* logits_in,
    const float* temperatures,
    const float* repetition_penalties,
    const float* presence_penalties,
    const float* frequency_penalties,
    const int64_t* batch_slots,
    const int64_t* token_inputs,
    const int64_t* seqstarts,
    const int64_t* start_pos,
    int32_t batch,
    int32_t vocab_size,
    uint16_t* penalty_count_map,
    float* logits_out
) {
    const int32_t block_size = 256;
    const int32_t grid_size = batch;
    apply_penalty_kernal<<<grid_size, block_size, 0, stream>>>(
        logits_in,
        temperatures,
        repetition_penalties,
        presence_penalties,
        frequency_penalties,
        batch_slots,
        token_inputs,
        seqstarts,
        start_pos,
        vocab_size,
        penalty_count_map,
        logits_out
    );
    return ppl::common::RC_SUCCESS;
}

}}}}}