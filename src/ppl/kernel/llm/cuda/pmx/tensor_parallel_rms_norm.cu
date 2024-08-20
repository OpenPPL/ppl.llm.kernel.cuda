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

#include "ppl/kernel/llm/cuda/pmx/tensor_parallel_rms_norm.h"
#include "cudakernel/common/common.cuh"
#include "ppl/common/log.h"

#include "type.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

struct tensor_parallel_rms_norm_fp16_kernel_param {
    fp16_t* input;              // [batch, hidden dim]
    fp16_t* weight;             // [hidden dim]
    fp32_t eps;
    fp32_t scale;
    int64_t batch;
    int64_t normalize_shape;    // last dimension
    fp32_t* pow_sum;
    fp16_t* output;             // quant(rmsnorm(input + skip))
};

template <int32_t VPT, int32_t TPB, bool UNROLL>
__global__
void tensor_parallel_rms_norm_fp16_pow_sum(
    tensor_parallel_rms_norm_fp16_kernel_param p
) {
    int64_t batch_id = blockIdx.x;
    int64_t norm_size = UNROLL ? VPT * TPB : p.normalize_shape;
    int64_t batch_offset = batch_id * norm_size;

    fp32_t accumulator = 0.0f;

    for (int32_t dim_offset = 0; dim_offset < norm_size; dim_offset += VPT * TPB) {
        int64_t dim_id = dim_offset + threadIdx.x * VPT;
        if (UNROLL || dim_id < norm_size) {
            fp16_t local_in[VPT];
            copy<sizeof(fp16_t) * VPT>(&p.input[batch_offset + dim_offset + threadIdx.x * VPT], local_in);
            #pragma unroll
            for (int32_t it = 0; it < VPT; it++) {
                fp32_t fp32_in = __half2float(local_in[it]);
                accumulator += fp32_in * fp32_in;
            }
        }
    }

    using BlockReduce = cub::BlockReduce<fp32_t, TPB>;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    __syncthreads();
    const fp32_t blk_sum = BlockReduce(tempStorage).Reduce(accumulator, cub::Sum());
    if (threadIdx.x == 0) {
        p.pow_sum[batch_id] = blk_sum;
    }
}


template <int32_t VPT, int32_t TPB, bool UNROLL>
__global__
void tensor_parallel_rms_norm_fp16_norm(
    tensor_parallel_rms_norm_fp16_kernel_param p
) {
    int64_t batch_id = blockIdx.x;
    int64_t norm_size = UNROLL ? VPT * TPB : p.normalize_shape;
    int64_t batch_offset = batch_id * norm_size;

    fp32_t r_normalize_shape = 1.0f / ((fp32_t)(norm_size) * p.scale);
    fp32_t r_reduced = rsqrtf(p.pow_sum[batch_id] * r_normalize_shape + p.eps);

    for (int32_t dim_offset = 0; dim_offset < norm_size; dim_offset += VPT * TPB) {
        int64_t dim_id = dim_offset + threadIdx.x * VPT;
        if (UNROLL || dim_id < norm_size) {
            fp16_t local_in[VPT];
            fp16_t local_weight[VPT];
            fp16_t local_out[VPT];

            copy<sizeof(fp16_t) * VPT>(&p.input[batch_offset + dim_id], local_in);
            copy<sizeof(fp16_t) * VPT>(&p.weight[dim_id], local_weight);

            #pragma unroll
            for (int32_t it = 0; it < VPT; it++) {
                local_out[it] = __float2half(__half2float(local_in[it]) * r_reduced) * local_weight[it];
            }
            copy<sizeof(fp16_t) * VPT>(local_out, &p.output[batch_offset + dim_id]);
        }
    }
}


ppl::common::RetCode tensor_parallel_rms_norm_fp16(
    const cudaStream_t stream,
    const void* input,              // [batch, hidden dim]
    const void* weight,             // [hidden dim]
    const fp32_t eps,
    const fp32_t scale,
    const int64_t batch,
    const int64_t normalize_shape,  // last dimension
    const ppl::common::NcclParam* nccl_param,
    void* pow_sum,                  // [batch,]
    void* output                    // [batch, hidden dim]
) {
    constexpr int32_t VPT = 16 / sizeof(fp16_t);
    const int32_t grid_size = batch;
    int32_t block_size = normalize_shape / VPT;

    tensor_parallel_rms_norm_fp16_kernel_param p = {
        (fp16_t*) input,
        (fp16_t*) weight,
        eps,
        scale,
        batch,
        normalize_shape,
        (fp32_t*) pow_sum,
        (fp16_t*) output
    };

    auto tp_pow_sum_kernel = tensor_parallel_rms_norm_fp16_pow_sum<VPT, 512, false>;
    auto tp_norm_kernel    = tensor_parallel_rms_norm_fp16_norm<VPT, 512, false>;

    switch (normalize_shape)
    {
    case 768:
        tp_pow_sum_kernel = tensor_parallel_rms_norm_fp16_pow_sum<VPT, 768 / VPT, true>;
        tp_norm_kernel    = tensor_parallel_rms_norm_fp16_norm<VPT, 768 / VPT, true>;
        break;
    case 1024:
        tp_pow_sum_kernel = tensor_parallel_rms_norm_fp16_pow_sum<VPT, 1024 / VPT, true>;
        tp_norm_kernel    = tensor_parallel_rms_norm_fp16_norm<VPT, 1024 / VPT, true>;
        break;
    case 4096:
        tp_pow_sum_kernel = tensor_parallel_rms_norm_fp16_pow_sum<VPT, 4096 / VPT, true>;
        tp_norm_kernel    = tensor_parallel_rms_norm_fp16_norm<VPT, 4096 / VPT, true>;
        break;
    case 5120:
        tp_pow_sum_kernel = tensor_parallel_rms_norm_fp16_pow_sum<VPT, 5120 / VPT, true>;
        tp_norm_kernel    = tensor_parallel_rms_norm_fp16_norm<VPT, 5120 / VPT, true>;
        break;
    case 8192:
        tp_pow_sum_kernel = tensor_parallel_rms_norm_fp16_pow_sum<VPT, 8192 / VPT, true>;
        tp_norm_kernel    = tensor_parallel_rms_norm_fp16_norm<VPT, 8192 / VPT, true>;
        break;
    default:
        block_size = 512;
        break;
    };

    tp_pow_sum_kernel<<<grid_size, block_size, 0, stream>>>(p);

    auto status = ppl::common::NcclAllReduceSum<fp32_t>(
        (fp32_t*)pow_sum,
        (fp32_t*)pow_sum,
        batch,
        nccl_param,
        stream);
    if (status != ppl::common::RC_SUCCESS)
        return status;

    tp_norm_kernel<<<grid_size, block_size, 0, stream>>>(p);

    return ppl::common::RC_SUCCESS;
}

}}}}}
