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

#include "ppl/kernel/llm/cuda/pmx/rms_norm.h"
#include "cudakernel/common/common.cuh"
#include "ppl/common/log.h"

#include "type.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

struct rms_norm_fp16_kernel_param {
    fp16_t* input;              // [batch, hidden dim]
    fp16_t* weight;             // [hidden dim]
    fp16_t* skip;               // [batch, hidden dim]
    fp32_t eps;
    int64_t batch;
    int64_t normalize_shape;    // last dimension
    fp16_t* skip_out;           // input + skip
    fp16_t* output;             // quant(rmsnorm(input + skip))
};


template <int32_t VPT, int32_t TPB>
__global__
void rms_norm_fp16_kernel(
    rms_norm_fp16_kernel_param p
) {
    int64_t batch_id = blockIdx.x;
    int64_t batch_offset = batch_id * p.normalize_shape;
    int64_t dim_id = threadIdx.x * VPT;

    fp32_t accumulator       = 0.0f;
    fp32_t r_normalize_shape = 1.0f / (fp32_t)(p.normalize_shape);

    fp16_t in_local[VPT];
    copy<sizeof(fp16_t) * VPT>(&p.input[batch_offset + dim_id], in_local);

    // skip relevent
    if (p.skip) {
        fp16_t local_skip[VPT];
        copy<sizeof(fp16_t) * VPT>(&p.skip[batch_offset + dim_id], local_skip);
        #pragma unroll
        for (int32_t it = 0; it < VPT; it++) {
            in_local[it] = in_local[it] + local_skip[it];
        }
    }
    if (p.skip_out) {
        copy<sizeof(fp16_t) * VPT>(in_local, &p.skip_out[batch_offset + dim_id]);
    }

    fp32_t local_z[VPT];
    #pragma unroll
    for (int32_t it = 0; it < VPT; it++){
        local_z[it] = __half2float(in_local[it]);
        accumulator += local_z[it] * local_z[it];
    }

    using BlockReduce = cub::BlockReduce<fp32_t, TPB>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ fp32_t r_reduced;

    __syncthreads();
    const fp32_t reduced = BlockReduce(tempStorage).Reduce(accumulator, cub::Sum()) * r_normalize_shape;
    if (threadIdx.x == 0) {
        r_reduced = rsqrtf(reduced + p.eps);
    }
    __syncthreads();

    fp16_t local_weight[VPT];
    fp16_t local_out[VPT];
    copy<sizeof(fp16_t) * VPT>(&p.weight[dim_id], local_weight);
    #pragma unroll
    for (int32_t it = 0; it < VPT; it++){
        local_out[it] = __float2half(local_z[it] * r_reduced) * local_weight[it];
    }
    copy<sizeof(fp16_t) * VPT>(local_out, &p.output[batch_offset + dim_id]);
};


template <int32_t VPT, int32_t TPB>
__global__
void rms_norm_fp16_kernel_default(
    rms_norm_fp16_kernel_param p
) {
    int64_t batch_id        = blockIdx.x;
    int64_t batch_offset    = batch_id * p.normalize_shape;

    extern __shared__ fp16_t z_shared[];

    fp32_t accumulator       = 0.0f;
    fp32_t r_normalize_shape = 1.0f / (fp32_t)(p.normalize_shape);

    for (int64_t dim_id = threadIdx.x * VPT; dim_id < p.normalize_shape; dim_id += TPB * VPT) {
        fp16_t local_in[VPT];
        copy<sizeof(fp16_t) * VPT>(&p.input[batch_offset + dim_id], local_in);

        if (p.skip) {
            fp16_t local_skip[VPT];
            copy<sizeof(fp16_t) * VPT>(&p.skip[batch_offset + dim_id], local_skip);
            #pragma unroll
            for (int32_t it = 0; it < VPT; it++)
                local_in[it] += local_skip[it];
            copy<sizeof(fp16_t) * VPT>(local_in, &z_shared[dim_id]);
        }
        else {
            copy<sizeof(fp16_t) * VPT>(&p.input[batch_offset + dim_id], &z_shared[dim_id]);
        }
        if (p.skip_out) {
            copy<sizeof(fp16_t) * VPT>(local_in, &p.skip_out[batch_offset + dim_id]);
        }

        #pragma unroll
        for (int32_t it = 0; it < VPT; it++) {
            fp32_t fp32_in = __half2float(local_in[it]);
            accumulator += fp32_in * fp32_in;
        }
    }

    using BlockReduce = cub::BlockReduce<fp32_t, TPB>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ fp32_t r_reduced;

    __syncthreads();
    const fp32_t reduced = BlockReduce(tempStorage).Reduce(accumulator, cub::Sum()) * r_normalize_shape;
    if (threadIdx.x == 0) {
        r_reduced = rsqrtf(reduced + p.eps);
    }
    __syncthreads();

    for (int64_t dim_id = threadIdx.x * VPT; dim_id < p.normalize_shape; dim_id += TPB * VPT) {
        fp16_t local_z[VPT];
        fp16_t local_weight[VPT];
        fp16_t local_out[VPT];
        copy<sizeof(fp16_t) * VPT>(&z_shared[dim_id], local_z);
        copy<sizeof(fp16_t) * VPT>(&p.weight[dim_id], local_weight);

        #pragma unroll
        for (int32_t it = 0; it < VPT; it++) {
            local_out[it] = __float2half(__half2float(local_z[it]) * r_reduced) * local_weight[it];
        }
        copy<sizeof(fp16_t) * VPT>(local_out, &p.output[batch_offset + dim_id]);
    }
}


ppl::common::RetCode rms_norm_fp16(
    const cudaStream_t stream,
    const void* input,              // [batch, hidden dim]
    const void* weight,             // [hidden dim]
    const void* skip,               // [batch, hidden dim]
    const float eps,
    const int64_t batch,
    const int64_t normalize_shape,  // last dimension
    void* skip_out,                 // [batch, hidden dim]
    void* output                    // [batch, hidden dim]
) {
    constexpr int32_t VPT = 16 / sizeof(fp16_t);
    const int32_t grid_size = batch;

    rms_norm_fp16_kernel_param p = {
        (fp16_t*) input,
        (fp16_t*) weight,
        (fp16_t*) skip,
        eps,
        batch,
        normalize_shape,
        (fp16_t*) skip_out,
        (fp16_t*) output
    };

    switch (normalize_shape)
    {
    case 768:
        rms_norm_fp16_kernel<VPT, 768 / VPT>
        <<<grid_size, 768 / VPT, 0, stream>>>(p);
        break;
    case 1024:
        rms_norm_fp16_kernel<VPT, 1024 / VPT>
        <<<grid_size, 1024 / VPT, 0, stream>>>(p);
        break;
    case 4096:
        rms_norm_fp16_kernel<VPT, 4096 / VPT>
        <<<grid_size, 4096 / VPT, 0, stream>>>(p);
        break;
    case 5120:
        rms_norm_fp16_kernel<VPT, 5120 / VPT>
        <<<grid_size, 5120 / VPT, 0, stream>>>(p);
        break;
    case 8192:
        rms_norm_fp16_kernel<VPT, 8192 / VPT>
        <<<grid_size, 8192 / VPT, 0, stream>>>(p);
        break;
    default:
        auto shm_size = p.normalize_shape * sizeof(fp16_t);
        if (shm_size > 48 * 1024) {
            auto cuda_err = cudaFuncSetAttribute(
                rms_norm_fp16_kernel_default<VPT, 512>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
            if (cuda_err == cudaErrorInvalidValue) {
                LOG(ERROR) << "(normalize_shape * sizeof(fp16_t))(" << shm_size << ") > max share memory size";
                return ppl::common::RC_UNSUPPORTED;
            }
        }
        rms_norm_fp16_kernel_default<VPT, 512>
        <<<grid_size, 512, shm_size, stream>>>(p);
        break;
    };
    return ppl::common::RC_SUCCESS;
}

}}}}}
