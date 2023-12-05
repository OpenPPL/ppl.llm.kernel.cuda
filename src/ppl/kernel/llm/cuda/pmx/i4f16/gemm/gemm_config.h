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

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

enum class TileConfig {
    Undefined,
    ChooseWithHeuristic,

    CtaShape16x64x64_WarpShape16x16x64,
    CtaShape16x16x512_WarpShape16x16x128,
    CtaShape16x32x256_WarpShape16x16x128,
    CtaShape16x64x256_WarpShape16x16x128,
    CtaShape16x64x128_WarpShape16x16x128,

    CtaShape32x64x64_WarpShape32x16x64,
    CtaShape32x64x128_WarpShape32x16x128,
    CtaShape32x64x256_WarpShape32x16x128,
    CtaShape32x32x128_WarpShape32x16x128,
    CtaShape32x32x256_WarpShape32x16x128,

    CtaShape64x32x128_WarpShape32x16x128,
    CtaShape64x32x64_WarpShape32x16x64
};

enum class SplitKStyle { NO_SPLIT_K, SPLIT_K };

struct GemmConfig {
    TileConfig tile_config = TileConfig::ChooseWithHeuristic;
    SplitKStyle split_k_style = SplitKStyle::NO_SPLIT_K;
    int split_k_factor = -1;
    int stages = -1;
};

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16
