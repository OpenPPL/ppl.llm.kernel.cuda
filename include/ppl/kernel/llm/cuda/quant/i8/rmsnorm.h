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

# ifndef __PPL_KERNEL_LLM_CUDA_QUANT_I8_RMSNORM_H__
# define __PPL_KERNEL_LLM_CUDA_QUANT_I8_RMSNORM_H__

# include "ppl/kernel/llm/cuda/common/general_include.h"
# include "ppl/kernel/llm/cuda/quant/common.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace quant { namespace i8 {
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
template<bool ConvertRowMajorToCol32>
ppl::common::RetCode skip_rmsnorm_with_minmax_quant_fp16i_int8o(
  const cudaStream_t stream,
  const fp16_t *x,                // 输入，形如 [batch, hidden dim]
  const fp16_t *weight,           // rmsnorm 的权重，形如 [hidden dim]
  const fp16_t *skip,             // 求和项，这是 skip rmsnorm 中的另一个输入，形如 [batch, hidden dim]
  const float eps,                // 1e-7
  const int64_t num_of_batch      // x 第一维的大小
  const int64_t normalize_shape,  // x 的最后一维大小
  fp16_t *skip_out,               // x + skip, 形如 [batch, hidden dim]
  fp16_t *out_scale,              // quant(rmsnorm(x + skip)), 形如 [batch]
  int8_t *out                     // quant(rmsnorm(x + skip)), 形如 [batch, hidden dim]
);

}}}}}}

# endif