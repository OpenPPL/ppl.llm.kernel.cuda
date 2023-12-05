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

#include "gemm_config.h"

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

constexpr TileConfig tile_configs[12] = {
    TileConfig::CtaShape16x64x64_WarpShape16x16x64, // 0
    TileConfig::CtaShape16x16x512_WarpShape16x16x128, // 1
    TileConfig::CtaShape16x32x256_WarpShape16x16x128, // 2
    TileConfig::CtaShape16x64x256_WarpShape16x16x128, // 3
    TileConfig::CtaShape16x64x128_WarpShape16x16x128, // 4

    TileConfig::CtaShape32x64x64_WarpShape32x16x64, // 5
    TileConfig::CtaShape32x64x128_WarpShape32x16x128, // 6
    TileConfig::CtaShape32x64x256_WarpShape32x16x128, // 7
    TileConfig::CtaShape32x32x128_WarpShape32x16x128, // 8
    TileConfig::CtaShape32x32x256_WarpShape32x16x128, // 9

    TileConfig::CtaShape64x32x128_WarpShape32x16x128, // 10
    TileConfig::CtaShape64x32x64_WarpShape32x16x64 // 11
};

struct GemmAlgorithm {
    uint64_t data[3];
};

using GemmAlgorithm_t = struct GemmAlgorithm;

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16