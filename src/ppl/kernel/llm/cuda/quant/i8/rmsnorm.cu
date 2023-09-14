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

# include "ppl/kernel/llm/cuda/quant/common.h"
# include "ppl/kernel/llm/cuda/quant/i8/rmsnorm.h"
# include "ppl/common/log.h"

/**
 * Skip Rmsnorm 与 动态量化 的融合算子
 * 这个算子会执行 Add, Rmsnorm, Dynamic Quant 三个操作
 * Dynamic Quant 是指这个算子会在运行时统计输入的 per token min-max 并以此对输入进行量化
 * 使用公式 int8 value = clip(round(fp16 value / scale), -127, 127)
 *     其中 scale = abs(max(fp16 value)) / 127
 * 这个算子会执行 per token 量化，每一个 token 都将拥有一个属于它的 scale
 *
 * Skip Rmsnorm 会先执行 y1 = x + skip
 *              而后执行 y2 = rmsnorm(y1)
 * 返回 y1, y2 作为输出
 */
 template <int VPT, int TPB>
__global__
void _skip_rmsnorm_with_minmax_quant_fp16i_int8o(
  const fp16_t *x,                // 输入，形如 [batch, hidden dim]
  const fp16_t *weight,           // rmsnorm 的权重，形如 [hidden dim]
  const fp16_t *skip,             // 求和项，这是 skip rmsnorm 中的另一个输入，形如 [batch, hidden dim]
  const float eps,                // 1e-7
  const int32_t normalize_shape,  // x 的最后一维大小
  fp16_t *o1,                     // x + skip
  fp16_t *o2_scale,               // quant(rmsnorm(x + skip))
  int8_t *o2                      // quant(rmsnorm(x + skip))
){
  const int32_t idx = normalize_shape * blockIdx.x + threadIdx.x * VPT;
  fp16_t inLocal[VPT]; fp16_t weightLocal[VPT];

  copy<sizeof(fp16_t) * VPT>(&x[idx], inLocal);
  copy<sizeof(fp16_t) * VPT>(&skip[idx], weightLocal);

  fp32_t accumulator = 0.0f; // accumulator
  fp32_t local_max   = eps;  // for int8 quant
  fp32_t r_normalize_shape = 1 / (float)(normalize_shape);

// step 1. compute x + skip
#pragma unroll
  for (int32_t it = 0; it < VPT; it++)
      inLocal[it] = inLocal[it] + weightLocal[it];
  copy<sizeof(fp16_t) * VPT>(inLocal, &o1[idx]);

#pragma unroll
  for (int32_t it = 0; it < VPT; it++){
    auto _x = __half2float(inLocal[it]);
    auto _w = __half2float(weightLocal[it]);

    accumulator = accumulator + _x * _x;
    local_max   = max(local_max, abs(_x * _w));
  }
  copy<sizeof(fp16_t) * VPT>(&weight[threadIdx.x * VPT], weightLocal);

  using BlockReduce = cub::BlockReduce<fp32_t, TPB>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ fp32_t r_reduced;

  const fp32_t global_max = BlockReduce(tempStorage).Reduce(local_max, cub::Max());
  __syncthreads();
  const fp32_t reduced = BlockReduce(tempStorage).Reduce(accumulator, cub::Sum()) * r_normalize_shape;

  if (threadIdx.x == 0) {
    const fp32_t scale = MIN_MAX_TO_SCALE(global_max);
    o2_scale[blockIdx.x] = __float2half(scale);
    r_reduced = rsqrt(reduced + eps) * RCP(scale);
  }
  __syncthreads();

  int8_t outLocal[VPT];
#pragma unroll
  for (int32_t it = 0; it < VPT; it++){
    fp32_t fp32_value = __half2float(inLocal[it]) * __half2float(weightLocal[it]);
    outLocal[it] = QUANT_FP32_TO_INT8_RCP(fp32_value, r_reduced);
  }
  copy<sizeof(int8_t) * VPT>(outLocal, &o2[idx]);
};


/**
 * Skip Rmsnorm 与 动态量化 的融合算子
 * 这个算子会执行 Add, Rmsnorm, Dynamic Quant 三个操作
 * Dynamic Quant 是指这个算子会在运行时统计输入的 per token min-max 并以此对输入进行量化
 * 使用公式 int8 value = clip(round(fp16 value / scale), -127, 127)
 *     其中 scale = abs(max(fp16 value)) / 127
 * 这个算子会执行 per token 量化，每一个 token 都将拥有一个属于它的 scale
 *
 * Skip Rmsnorm 会先执行 y1 = x + skip
 *              而后执行 y2 = rmsnorm(y1)
 * 返回 y1, y2 作为输出
 */
ppl::common::RetCode skip_rmsnorm_with_minmax_quant_fp16i_int8o(
  const cudaStream_t stream,
  const fp16_t *x,                // 输入，形如 [batch, hidden dim]
  const fp16_t *weight,           // rmsnorm 的权重，形如 [hidden dim]
  const fp16_t *skip,             // 求和项，这是 skip rmsnorm 中的另一个输入，形如 [batch, hidden dim]
  const float eps,                // 1e-7
  const int64_t num_of_batch      // x 第一维的大小
  const int64_t normalize_shape,  // x 的最后一维大小
  fp16_t *o1,                     // x + skip, 形如 [batch, hidden dim]
  fp16_t *o2_scale,               // quant(rmsnorm(x + skip)), 形如 [batch]
  int8_t *o2                      // quant(rmsnorm(x + skip)), 形如 [batch, hidden dim]
) {
/*
  Merged Impl of Skip Rmsnorm + PerToken Quant(fp16 input, int8 output)
*/
  constexpr int32_t VPT = 16 / sizeof(fp16_t);
  const int32_t grid_size = num_of_batch;

  switch (normalize_shape)
  {
  case 768:
    _skip_rmsnorm_with_minmax_quant_fp16i_int8o<VPT, 768 / VPT>
    <<<grid_size, 768 / VPT, 0, stream>>>(
      x,
      weight,
      skip,
      eps, normalize_shape,
      o1,
      o2_scale,
      o2
    );
    break;
  case 1024:
    _skip_rmsnorm_with_minmax_quant_fp16i_int8o<VPT, 1024 / VPT>
    <<<grid_size, 1024 / VPT, 0, stream>>>(
      x,
      weight,
      skip,
      eps, normalize_shape,
      o1,
      o2_scale,
      o2
    );
    break;
  case 4096:
    _skip_rmsnorm_with_minmax_quant_fp16i_int8o<VPT, 4096 / VPT>
    <<<grid_size, 4096 / VPT, 0, stream>>>(
      x,
      weight,
      skip,
      eps, normalize_shape,
      o1,
      o2_scale,
      o2
    );
    break;
  case 8192:
    _skip_rmsnorm_with_minmax_quant_fp16i_int8o<VPT, 8192 / VPT>
    <<<grid_size, 8192 / VPT, 0, stream>>>(
      x,
      weight,
      skip,
      eps, normalize_shape,
      o1,
      o2_scale,
      o2
    );
    break;
  default:
    LOG(ERROR) << "Input data length must be a multiple of 8.";
    return ppl::common::RC_UNSUPPORTED;
    break;
  };
  return ppl::common::RC_SUCCESS;
}