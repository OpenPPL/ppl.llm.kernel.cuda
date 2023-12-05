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

template <int M = 1, int N = 1, int K = 1>
struct GemmShape {
    static int constexpr kM = M;
    static int constexpr kN = N;
    static int constexpr kK = K;

    static int constexpr kMN = M * N;
    static int constexpr kMK = M * K;
    static int constexpr kKN = N * K;
    static int constexpr kMNK = M * N * K;

    static int constexpr kCount = kMNK;
};

template <int M = 1, int N = 1, int K = 1>
using BlockShape = GemmShape<M, N, K>;

template <int M = 1, int N = 1, int K = 1>
using WarpShape = GemmShape<M, N, K>;

template <int M = 1, int N = 1, int K = 1>
using MmaShape = GemmShape<M, N, K>;

template <typename BlockShape_, typename WarpShape_, typename MmaShape_, int Stages, bool SplitK, bool Bias,
          int GroupSize>
struct DefaultMma {
    using BlockShape = BlockShape_;
    using WarpShape = WarpShape_;
    using MmaShape = MmaShape_;

    static constexpr int kGroupSize = GroupSize;
    static constexpr int kStages = Stages;

    using WarpCount =
        GemmShape<BlockShape::kM / WarpShape::kM, BlockShape::kN / WarpShape::kN, BlockShape::kK / WarpShape::kK>;

    using BlockShapeS = GemmShape<BlockShape::kM, (BlockShape::kN * 4), BlockShape::kK>;
    using WarpShapeS = GemmShape<WarpShape::kM, (WarpShape::kN * 4), WarpShape::kK>;

    static constexpr int kWarpSize = 32;
    static constexpr int kThreads = WarpCount::kCount * kWarpSize;

    static constexpr int kSlices = WarpCount::kK;
    static constexpr int kSliceK = BlockShape::kK / kSlices;

    static constexpr int kWarpGroupSize = WarpCount::kMN;

    static bool const kSplitK = SplitK;
    static bool const kBias = Bias;
    static constexpr int kIter = (-kStages + 1);

    using IteratorA = DefaultIteratorA<BlockShape, WarpCount, kWarpSize, kStages, kSliceK>;

    using IteratorB = DefaultIteratorB<BlockShape, WarpCount, kWarpSize, kStages, kSliceK>;

    using IteratorS = DefaultIteratorS<BlockShapeS, WarpCount, kWarpSize, kStages, kSliceK, kSlices, kGroupSize>;

    using WarpIterA = WarpIteratorA<BlockShape, WarpShape, MmaShape, IteratorA::kSmemPadCtaK, Stages>;
    using WarpIterB = WarpIteratorB<BlockShape, WarpShape, MmaShape, IteratorB::kSmemPadCtaK, Stages>;
    using WarpIterS =
        WarpIteratorS<BlockShapeS, WarpShapeS, MmaShape, IteratorS::kSmemPadCtaN, Stages, kSliceK, GroupSize>;
};

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16