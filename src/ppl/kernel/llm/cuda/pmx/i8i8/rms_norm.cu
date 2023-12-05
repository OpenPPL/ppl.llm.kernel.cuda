
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

#include "ppl/kernel/llm/cuda/pmx/i8i8/quantize.h"
#include "ppl/common/log.h"
#include "cudakernel/common/common.cuh"

#include "../quant_utils.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i8i8 {

struct skip_rmsnorm_minmax_quantize_fp16_kernel_param {
    fp16_t* input;            // [batch, hidden dim]
    fp16_t* weight;           // [hidden dim]
    fp16_t* skip;             // [batch, hidden dim]
    fp32_t eps;
    int64_t batch;
    int64_t normalize_shape; // last dimension
    fp32_t up_scale;
    fp16_t* skip_out;               // input + skip
    fp16_t* output_scale;           // quant(rmsnorm(input + skip))
    int8_t* output;                 // quant(rmsnorm(input + skip))
};

 template <int32_t VPT, int32_t TPB, matrix_layout_t TO_LAYOUT>
__global__
void skip_rmsnorm_minmax_quantize_fp16_kernel(
  skip_rmsnorm_minmax_quantize_fp16_kernel_param p
) {
    int64_t batch_id = blockIdx.x;
    int64_t batch_offset = batch_id * p.normalize_shape;
    int64_t dim_id = threadIdx.x * VPT;
    fp16_t in_local[VPT]; fp16_t weight_local[VPT];

    MatrixLayoutHelper<TO_LAYOUT> idx_hlp;
    idx_hlp.Init(p.batch, p.normalize_shape);

    copy<sizeof(fp16_t) * VPT>(&p.input[batch_offset + dim_id], in_local);

    fp32_t accumulator = 0.0f; // accumulator
    fp32_t local_max   = p.eps;  // for int8 quant
    fp32_t r_normalize_shape = 1.0f / (float)(p.normalize_shape);

    // step 1. compute input + skip
    if (p.skip) {
        copy<sizeof(fp16_t) * VPT>(&p.skip[batch_offset + dim_id], weight_local);
#pragma unroll
        for (int32_t it = 0; it < VPT; it++) {
            in_local[it] = in_local[it] + weight_local[it];
        }
    }
    copy<sizeof(fp16_t) * VPT>(in_local, &p.skip_out[batch_offset + dim_id]);
    copy<sizeof(fp16_t) * VPT>(&p.weight[dim_id], weight_local);

#pragma unroll
    for (int32_t it = 0; it < VPT; it++){
        auto _x = __half2float(in_local[it]);
        auto _w = __half2float(weight_local[it]);

        accumulator = accumulator + _x * _x;
        local_max   = fmax(local_max, fabs(_x * _w));
    }

    using BlockReduce = cub::BlockReduce<fp32_t, TPB>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ fp32_t r_reduced;
    __shared__ fp32_t scale;

    const fp32_t global_max = BlockReduce(tempStorage).Reduce(local_max, cub::Max());
    __syncthreads();
    const fp32_t reduced = BlockReduce(tempStorage).Reduce(accumulator, cub::Sum()) * r_normalize_shape;
    if (threadIdx.x == 0) {
        r_reduced = rsqrtf(reduced + p.eps);
        scale = min_max_range_to_scale(global_max * r_reduced, INT8_QLEVEL);
        p.output_scale[batch_id] = __float2half(scale * p.up_scale);
    }
    __syncthreads();

    int8_t out_local[VPT];
    #pragma unroll
    for (int32_t it = 0; it < VPT; it++){
        fp32_t fp32_value = __half2float(in_local[it]) * __half2float(weight_local[it]) * r_reduced;
        out_local[it] = quant_scalar<fp32_t, fp32_t, false>(fp32_value, scale, INT8_QMIN, INT8_QMAX);
    }
    copy<sizeof(int8_t) * VPT>(out_local, &p.output[idx_hlp.GetOffset(batch_id, dim_id)]);
};


template <int32_t TPB, matrix_layout_t TO_LAYOUT>
__global__
void skip_rmsnorm_minmax_quantize_fp16_kernel_default(
  skip_rmsnorm_minmax_quantize_fp16_kernel_param p
) {
    constexpr int32_t VPT       = 16;
    constexpr int32_t V8PT      = VPT / sizeof(int8_t);
    constexpr int32_t V16PT     = VPT / sizeof(fp16_t);

    int64_t batch_id        = blockIdx.x;
    int64_t batch_offset    = batch_id * p.normalize_shape;

    MatrixLayoutHelper<TO_LAYOUT> idx_hlp;
    idx_hlp.Init(p.batch, p.normalize_shape);

    extern __shared__ fp16_t z_shared[];

    fp32_t accumulator       = 0.0f; // accumulator
    fp32_t local_max         = p.eps; // for int8 quant
    fp32_t r_normalize_shape = 1.0f / (fp32_t)(p.normalize_shape);

    for (int64_t i = threadIdx.x * V16PT; i < p.normalize_shape; i += TPB * V16PT) {
        fp16_t local_in[V16PT];
        fp16_t local_weight[V16PT];
        copy<VPT>(&p.input[batch_offset + i], local_in);
        copy<VPT>(&p.weight[i], local_weight);

        if (p.skip) {
            fp16_t local_skip[V16PT];
            copy<VPT>(&p.skip[batch_offset + i], local_skip);
            #pragma unroll
            for (int32_t v_idx = 0; v_idx < V16PT; v_idx += 1)
                local_in[v_idx] += local_skip[v_idx];
            copy<VPT>(local_in, &z_shared[i]);
        }
        else copy<VPT>(&p.input[batch_offset + i], &z_shared[i]);
        copy<VPT>(local_in, &p.skip_out[batch_offset + i]);

        #pragma unroll
        for (int32_t v_idx = 0; v_idx < V16PT; v_idx += 1) {
            fp32_t fp32_in     = __half2float(local_in[v_idx]);
            fp32_t fp32_weight = __half2float(local_weight[v_idx]);
            accumulator += fp32_in * fp32_in;
            local_max   = fmax(local_max, fabs(fp32_in * fp32_weight));
        }
    }

    using BlockReduce = cub::BlockReduce<fp32_t, TPB>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ fp32_t r_reduced;
    __shared__ fp32_t scale;

    const fp32_t global_max = BlockReduce(tempStorage).Reduce(local_max, cub::Max());
    __syncthreads();
    const fp32_t reduced = BlockReduce(tempStorage).Reduce(accumulator, cub::Sum()) * r_normalize_shape;
    if (threadIdx.x == 0) {
        r_reduced = rsqrtf(reduced + p.eps);
        scale = min_max_range_to_scale(global_max * r_reduced, INT8_QLEVEL);
        p.output_scale[batch_id] = __float2half(scale * p.up_scale);
    }
    __syncthreads();

    for (int64_t i = threadIdx.x * V8PT; i < p.normalize_shape; i += TPB * V8PT) {
        fp16_t local_z[V8PT];
        fp16_t local_weight[V8PT];
        int8_t local_out[V8PT];
        copy<VPT>(&z_shared[i], local_z);
        copy<VPT>(&z_shared[i + V16PT], &local_z[V16PT]);
        copy<VPT>(&p.weight[i], local_weight);
        copy<VPT>(&p.weight[i + V16PT], &local_weight[V16PT]);

        #pragma unroll
        for (int32_t v_idx = 0; v_idx < V8PT; v_idx += 1) {
            fp32_t fp32_value = __half2float(local_z[v_idx]) * __half2float(local_weight[v_idx]) * r_reduced;
            local_out[v_idx] = quant_scalar<fp32_t, fp32_t, false>(fp32_value, scale, INT8_QMIN, INT8_QMAX);
        }
        copy<VPT>(local_out, &p.output[idx_hlp.GetOffset(batch_id, i)]);
    }
}


template<matrix_layout_t TO_LAYOUT>
ppl::common::RetCode skip_rmsnorm_minmax_quantize_fp16(
  const cudaStream_t stream,
  const void* input,           // [batch, hidden dim]
  const void* weight,           // [hidden dim]
  const void* skip,             // [batch, hidden dim]
  const float eps,
  const int64_t batch,
  const int64_t normalize_shape,
  const float up_scale, // scale[i] = scale * up_scale for precision
  void* skip_out,               // [batch, hidden dim]
  void* output_scale,           // [batch]
  void* output                  // [batch, hidden dim]
) {
/*
  Merged Impl of Skip Rmsnorm + PerToken Quant(fp16 input, int8 output)
*/
    constexpr int32_t VPT = 16 / sizeof(fp16_t);
    const int32_t grid_size = batch;

    skip_rmsnorm_minmax_quantize_fp16_kernel_param p = {
        (fp16_t*) input,            // [batch, hidden dim]
        (fp16_t*) weight,           // [hidden dim]
        (fp16_t*) skip,             // [batch, hidden dim]
        eps,
        batch,
        normalize_shape, // last dimension
        up_scale,
        (fp16_t*) skip_out,               // input + skip
        (fp16_t*) output_scale,           // quant(rmsnorm(input + skip))
        (int8_t*) output                 // quant(rmsnorm(input + skip))
    };

    switch (normalize_shape)
    {
    case 768:
        skip_rmsnorm_minmax_quantize_fp16_kernel<VPT, 768 / VPT, TO_LAYOUT>
        <<<grid_size, 768 / VPT, 0, stream>>>(p);
        break;
    case 1024:
        skip_rmsnorm_minmax_quantize_fp16_kernel<VPT, 1024 / VPT, TO_LAYOUT>
        <<<grid_size, 1024 / VPT, 0, stream>>>(p);
        break;
    case 4096:
        skip_rmsnorm_minmax_quantize_fp16_kernel<VPT, 4096 / VPT, TO_LAYOUT>
        <<<grid_size, 4096 / VPT, 0, stream>>>(p);
        break;
    case 5120:
        skip_rmsnorm_minmax_quantize_fp16_kernel<VPT, 5120 / VPT, TO_LAYOUT>
        <<<grid_size, 5120 / VPT, 0, stream>>>(p);
        break;
    case 8192:
        skip_rmsnorm_minmax_quantize_fp16_kernel<VPT, 8192 / VPT, TO_LAYOUT>
        <<<grid_size, 8192 / VPT, 0, stream>>>(p);
        break;
    default:
        auto shm_size = p.normalize_shape * sizeof(fp16_t);
        if (shm_size > 48 * 1024) {
            auto cuda_err = cudaFuncSetAttribute(
                skip_rmsnorm_minmax_quantize_fp16_kernel_default<512, TO_LAYOUT>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
            if (cuda_err == cudaErrorInvalidValue) {
                LOG(ERROR) << "(normalize_shape * sizeof(fp16_t))(" << shm_size << ") > max share memory size";
                return ppl::common::RC_UNSUPPORTED;
            }
        }
        skip_rmsnorm_minmax_quantize_fp16_kernel_default<512, TO_LAYOUT>
        <<<grid_size, 512, shm_size, stream>>>(p);
        break;
    };
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode skip_rmsnorm_minmax_quantize_fp16(
  const cudaStream_t stream,
  const void* input,           // [batch, hidden dim]
  const void* weight,           // [hidden dim]
  const void* skip,             // [batch, hidden dim]
  const float eps,
  const int64_t batch,
  const int64_t normalize_shape,
  const float up_scale, // scale[i] = scale * up_scale for precision
  const matrix_layout_t to_layout,
  void* skip_out,               // [batch, hidden dim]
  void* output_scale,           // [batch]
  void* output                  // [batch, hidden dim]
) {
    if (to_layout == MATRIX_LAYOUT_ROW_MAJOR) {
        return skip_rmsnorm_minmax_quantize_fp16<MATRIX_LAYOUT_ROW_MAJOR>(
            stream, input, weight, skip, eps, batch, normalize_shape,
            up_scale, skip_out, output_scale, output
        );
    }

    if (to_layout == MATRIX_LAYOUT_COL_MAJOR) {
        return skip_rmsnorm_minmax_quantize_fp16<MATRIX_LAYOUT_COL_MAJOR>(
            stream, input, weight, skip, eps, batch, normalize_shape,
            up_scale, skip_out, output_scale, output
        );
    }

    if (to_layout == MATRIX_LAYOUT_COL32) {
        return skip_rmsnorm_minmax_quantize_fp16<MATRIX_LAYOUT_COL32>(
            stream, input, weight, skip, eps, batch, normalize_shape,
            up_scale, skip_out, output_scale, output
        );
    }

    if (to_layout == MATRIX_LAYOUT_COL32_2R_4R4) {
        return skip_rmsnorm_minmax_quantize_fp16<MATRIX_LAYOUT_COL32_2R_4R4>(
            stream, input, weight, skip, eps, batch, normalize_shape,
            up_scale, skip_out, output_scale, output
        );
    }

    LOG(ERROR) << "unsupported matrix layout: " << (int32_t)to_layout;
    return ppl::common::RC_UNSUPPORTED;
}

}}}}}}
