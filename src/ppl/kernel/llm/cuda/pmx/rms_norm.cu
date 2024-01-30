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

#include <cuda_fp16.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

template <int32_t VPT, int32_t TPB>
__global__
void rms_norm_kernel_fp16(
    const half *input,
    const half *weight,
    const float eps,
    const int64_t normalize_shape,
    half *output,
    half *skip_out
)
{
    const int64_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half inLocal[VPT]; half weightLocal[VPT];

    copy<sizeof(half) * VPT>(&input[idx], inLocal);

    float accumulator = 0.0f; // accumulator
    float r_normalize_shape = 1.0f / (float)(normalize_shape);

    #pragma unroll
    for (int32_t it = 0; it < VPT; it++)
        accumulator = accumulator + (__half2float(inLocal[it]) * __half2float(inLocal[it]));
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);

    const float reduced = BlockAllReduce<SumOp, float, TPB>(accumulator) * r_normalize_shape;

    __shared__ float r_reduced;

    if (threadIdx.x == 0)
        r_reduced = rsqrt(reduced + eps);
    __syncthreads();

    half outLocal[VPT];

    #pragma unroll
    for (int32_t it = 0; it < VPT; it++)
        outLocal[it] = __float2half(__half2float(inLocal[it]) * r_reduced) * weightLocal[it];

    copy<sizeof(half) * VPT>(outLocal, &output[idx]);
    copy<sizeof(half) * VPT>(inLocal, &skip_out[idx]);
};


template <int32_t TPB>
__global__
void rms_norm_kernel_fp16_default(
    const half *input,
    const half *weight,
    const float eps,
    const int64_t normalize_shape,
    half *output,
    half *skip_out
)
{
    auto cur_x = input + normalize_shape * blockIdx.x;
    auto cur_o1 = output + normalize_shape * blockIdx.x;
    auto cur_o2 = skip_out + normalize_shape * blockIdx.x;

    float accumulator = 0.0f; // accumulator
    float r_normalize_shape = 1.0f / (float)(normalize_shape);

    for (int32_t idx = threadIdx.x; idx < normalize_shape; idx += TPB) {
        cur_o2[idx] = cur_x[idx];
        accumulator = accumulator + (__half2float(cur_x[idx]) * __half2float(cur_x[idx]));
    }

    const float reduced = BlockAllReduce<SumOp, float, TPB>(accumulator) * r_normalize_shape;

    __shared__ float r_reduced;

    if (threadIdx.x == 0)
        r_reduced = rsqrt(reduced + eps);
    __syncthreads();

    for (int32_t idx = threadIdx.x; idx < normalize_shape; idx += TPB) {
        cur_o1[idx] = __float2half(__half2float(cur_x[idx]) * r_reduced) * weight[idx];
    }
};


/**
 * RMSNorm Cuda impl template(with skip_in connection).
 *
 * @param VPT: Value processed per thread.
 * @param TPB: Thread per block.

 * @param input data pointer of input.
 * @param weight parameter of this RmsNorm.
 * @param skip_in skip_in connection of this RmsNorm.
 * @param eps
 * @param normalize_shape num of elements within last dimension of input.
 * @param output data pointer of output.
 * input and output should share a same size.
 */
 template <int32_t VPT, int32_t TPB>
__global__
void skip_rms_norm_kernel_fp16(
    const half *input,
    const half *weight,
    const half *skip_in,
    const float eps,
    const int64_t normalize_shape,
    half *output,
    half *skip_out
){
    const int64_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
    half inLocal[VPT]; half weightLocal[VPT];
    float inLocal_fp32[VPT];

    copy<sizeof(half) * VPT>(&input[idx], inLocal);
    copy<sizeof(half) * VPT>(&skip_in[idx], weightLocal);
    float accumulator = 0.0f; // accumulator
    float r_normalize_shape = 1.0f / (float)(normalize_shape);

    // step 1. compute input + skip_in
    #pragma unroll
    for (int32_t it = 0; it < VPT; it++) 
        inLocal[it] = inLocal[it] + weightLocal[it];

    #pragma unroll
    for (int32_t it = 0; it < VPT; it++) 
        inLocal_fp32[it] = __half2float(inLocal[it]);

    copy<sizeof(half) * VPT>(inLocal, &skip_out[idx]);

    #pragma unroll
    for (int32_t it = 0; it < VPT; it++)
        accumulator = accumulator + (inLocal_fp32[it] * inLocal_fp32[it]);
    copy<sizeof(half) * VPT>(&weight[threadIdx.x * VPT], weightLocal);

    const float reduced = BlockAllReduce<SumOp, float, TPB>(accumulator) * r_normalize_shape;

    __shared__ float r_reduced;

    if (threadIdx.x == 0)
        r_reduced = rsqrt(reduced + eps);
    __syncthreads();

    half outLocal[VPT];
    #pragma unroll
    for (int32_t it = 0; it < VPT; it++) 
        outLocal[it] = __float2half(inLocal_fp32[it] * r_reduced) * weightLocal[it];
    
    copy<sizeof(half) * VPT>(outLocal, &output[idx]);
};



 template <int32_t TPB>
__global__
void skip_rms_norm_kernel_fp16_default(
    const half *input,
    const half *weight,
    const half *skip_in,
    const float eps,
    const int64_t normalize_shape,
    half *output,
    half *skip_out
){
    auto cur_x = input + normalize_shape * blockIdx.x;
    auto cur_skip = skip_in + normalize_shape * blockIdx.x;
    auto cur_o1 = output + normalize_shape * blockIdx.x;
    auto cur_o2 = skip_out + normalize_shape * blockIdx.x;

    float accumulator = 0.0f; // accumulator
    float r_normalize_shape = 1.0f / (float)(normalize_shape);

    // step 1. compute input + skip_in

    for(int32_t idx = threadIdx.x; idx < normalize_shape; idx += TPB) {
        half temp = cur_x[idx] + cur_skip[idx];
        cur_o2[idx] = temp;
        accumulator = accumulator + (__half2float(temp) * __half2float(temp));
    }

    const float reduced = BlockAllReduce<SumOp, float, TPB>(accumulator) * r_normalize_shape;

    __shared__ float r_reduced;

    if (threadIdx.x == 0)
        r_reduced = rsqrt(reduced + eps);
    __syncthreads();

    for(int32_t idx = threadIdx.x; idx < normalize_shape; idx += TPB) {
        float temp = __half2float(cur_x[idx] + cur_skip[idx]);
        cur_o1[idx] = __float2half(temp * r_reduced) * weight[idx];
    }
};

ppl::common::RetCode rms_norm(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const void* weight,
    const void* skip_in,
    const int32_t axis,
    const float eps,
    const bool skip_term,
    void* output,
    void* skip_out)
{
    if (input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "RmsNorm only support fp16, but got ["<< input_shape->GetDataType() << "]";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (!skip_term) {
        LOG(ERROR) << "RmsNorm only support skip_term";
        return ppl::common::RC_UNSUPPORTED;
    }
    constexpr int32_t VPT = 16 / sizeof(half);

    const int32_t real_axis = axis < 0 ? input_shape->GetDimCount() + axis : axis;
    const int64_t normalize_shape = input_shape->CalcElementsFromDimensionExcludingPadding(real_axis);
    const int64_t grid_size = input_shape->CalcElementsToDimensionExcludingPadding(real_axis);
    
    if (skip_in == nullptr) {
      switch (normalize_shape)
      {
      case 768:
        rms_norm_kernel_fp16<VPT, 768 / VPT>
        <<<grid_size, 768 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          eps, normalize_shape,
          (half*)(output),
          (half*)(skip_out));
        break;
      case 1024:
        rms_norm_kernel_fp16<VPT, 1024 / VPT>
        <<<grid_size, 1024 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          eps, normalize_shape, 
          (half*)(output),
          (half*)(skip_out));
        break;
      case 4096:
        rms_norm_kernel_fp16<VPT, 4096 / VPT>
        <<<grid_size, 4096 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          eps, normalize_shape, 
          (half*)(output),
          (half*)(skip_out));
        break;
      case 5120:
        rms_norm_kernel_fp16<VPT, 5120 / VPT>
        <<<grid_size, 5120 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          eps, normalize_shape, 
          (half*)(output),
          (half*)(skip_out));
        break;
      case 8192:
        rms_norm_kernel_fp16<VPT, 8192 / VPT>
        <<<grid_size, 8192 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          eps, normalize_shape, 
          (half*)(output),
          (half*)(skip_out));
        break;
      default:
        rms_norm_kernel_fp16_default<512>
        <<<grid_size, 512, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          eps, normalize_shape, 
          (half*)(output),
          (half*)(skip_out));
      };
    } else {
      switch (normalize_shape)
      {
      case 768:
        skip_rms_norm_kernel_fp16<VPT, 768 / VPT>
        <<<grid_size, 768 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          (half*)(skip_in), 
          eps, normalize_shape, 
          (half*)(output),
          (half*)(skip_out));
        break;
      case 1024:
      skip_rms_norm_kernel_fp16<VPT, 1024 / VPT>
        <<<grid_size, 1024 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          (half*)(skip_in), 
          eps, normalize_shape, 
          (half*)(output),
          (half*)(skip_out));
        break;
      case 4096:
      skip_rms_norm_kernel_fp16<VPT, 4096 / VPT>
        <<<grid_size, 4096 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          (half*)(skip_in), 
          eps, normalize_shape, 
          (half*)(output),
          (half*)(skip_out));
        break;
      case 5120:
      skip_rms_norm_kernel_fp16<VPT, 5120 / VPT>
        <<<grid_size, 5120 / VPT, 0, stream>>>(
          (half*)(input), 
          (half*)(weight), 
          (half*)(skip_in), 
          eps, normalize_shape, 
          (half*)(output),
          (half*)(skip_out));
        break;
      case 8192:
        skip_rms_norm_kernel_fp16<VPT, 8192 / VPT>
          <<<grid_size, 8192 / VPT, 0, stream>>>(
            (half*)(input), 
            (half*)(weight), 
            (half*)(skip_in), 
            eps, normalize_shape, 
            (half*)(output),
            (half*)(skip_out));
        break;
      default:
        skip_rms_norm_kernel_fp16_default<512>
          <<<grid_size, 512, 0, stream>>>(
            (half*)(input), 
            (half*)(weight), 
            (half*)(skip_in), 
            eps, normalize_shape, 
            (half*)(output),
            (half*)(skip_out));
      };
    }
    
    return ppl::common::RC_SUCCESS;
}

}}}}}
