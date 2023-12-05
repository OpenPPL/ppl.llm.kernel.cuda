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

#include "utils.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

template <typename BlockShape_, typename WarpCount_, int WarpSize, int Stages, int SliceK>
struct DefaultIteratorA {
    static constexpr int kSliceK = SliceK;

    using AccessType = float4;
    using ElementType = half;
    using BlockShape = BlockShape_;
    using WarpCount = WarpCount_;

    static constexpr int kStages = Stages;
    static constexpr int kWarpSize = WarpSize;
    static constexpr int kAccessSize = 128;
    static constexpr int kElementSize = sizeof(ElementType);

    using PShape = LinearShape<kSliceK, BlockShape::kM>;
    static constexpr int kAccessM = 1;
    static constexpr int kAccessK = 8;
    static constexpr int WarpNumThreadsK = const_min(const_max(PShape::kContiguous / kAccessK, 1), kWarpSize);
    static constexpr int WarpNumThreadsM = kWarpSize / WarpNumThreadsK;
    using WarpThreadA = LinearShape<WarpNumThreadsK, WarpNumThreadsM>;
    using WarpAccess = LinearShape<WarpNumThreadsK * kAccessK, WarpNumThreadsM * kAccessM>;
    using WarpAccessI =
        LinearShape<PShape::kContiguous / WarpAccess::kContiguous, PShape::kStrided / WarpAccess::kStrided>;
    using BlockWarpA = LinearShape<const_min(WarpAccessI::kContiguous, WarpCount::kMN),
                                   const_max(WarpCount::kMN / WarpAccessI::kContiguous, 1)>;
    using IterCount = LinearShape<WarpAccessI::kContiguous / BlockWarpA::kContiguous,
                                  const_max(WarpAccessI::kStrided / BlockWarpA::kStrided, 1)>;
    using ElementCount =
        LinearShape<WarpAccess::kContiguous * IterCount::kContiguous, WarpAccess::kStrided * IterCount::kStrided>;

    static constexpr int kSmemPadCtaK = kSliceK + 8;
    static constexpr int kSizePerTile = BlockShape::kM * kSmemPadCtaK;
    static constexpr int kSmemByteSize = kElementSize * kStages * kSizePerTile;

    struct SourceContext {
        mutable ElementType* global_addr;
        mutable int problem_k;
        mutable int problem_m;
        mutable int index_b{0};
        mutable int index_b_k{0};
        mutable int index_b_m{0};
        mutable int src_step_m;

        int offset{0};
        int offset_k{0};
        int offset_m{0};
        int iter_k{0};
        int iter_m{0};
        bool is_valid_k;
        bool is_valid_m;

        static constexpr int src_step_k{WarpAccess::kContiguous};

        SourceContext() = default;
        __device__ SourceContext(const ElementType* src, const int k, const int m, const int src_offset,
                                 const int src_offset_k, const int src_offset_m, const int src_step)
            : global_addr(const_cast<ElementType*>(src))
            , problem_k(k)
            , problem_m(m)
            , index_b(src_offset)
            , index_b_k(src_offset_k)
            , index_b_m(src_offset_m)
            , src_step_m(src_step)
            , offset(src_offset)
            , offset_k(src_offset_k)
            , offset_m(src_offset_m)
            , is_valid_k(offset_k < k)
            , is_valid_m(offset_m < m) {}

        __device__ SourceContext& operator=(const SourceContext& other) {
            if (this != &other) {
                global_addr = other.global_addr;
                problem_k = other.problem_k;
                problem_m = other.problem_m;
                index_b = other.index_b;
                index_b_k = other.index_b_k;
                index_b_m = other.index_b_m;
                src_step_m = other.src_step_m;
                offset = other.offset;
                offset_k = other.offset_k;
                offset_m = other.offset_m;
                is_valid_k = other.is_valid_k;
                is_valid_m = other.is_valid_m;
            }
            return *this;
        }

        __device__ SourceContext& operator++() {
            offset_k += src_step_k;
            is_valid_k = offset_k < problem_k;
            offset += src_step_k;
            ++iter_k;
            if (iter_k < IterCount::kContiguous) {
                return *this;
            }

            iter_k = 0;
            offset_k = index_b_k;
            offset_m += WarpAccess::kStrided;
            offset += src_step_m;
            is_valid_m = offset_m < problem_m;
            is_valid_k = offset_k < problem_k;
            ++iter_m;
            return *this;
        }

        __device__ void nexts() {
            iter_m = 0;
            index_b += BlockShape::kK;

            offset = index_b;
            offset_k = index_b_k;
            offset_m = index_b_m;

            is_valid_m = offset_m < problem_m;
            is_valid_k = offset_k < problem_k;
        }
    };

    struct DestinationContext {
        mutable uint32_t smem_addr;
        mutable int index_b;
        mutable int dst_step_m;

        int offset{0};
        int iter_k{0};
        int iter_m{0};

        static constexpr int dst_step_k{WarpAccess::kContiguous * kElementSize};

        DestinationContext() = default;

        __device__ DestinationContext(const uint32_t addr, const int dst_offset, const int dst_step)
            : smem_addr(addr), index_b(dst_offset), dst_step_m(dst_step), offset(dst_offset) {}

        __device__ DestinationContext& operator=(const DestinationContext& other) {
            if (this != &other) {
                smem_addr = other.smem_addr;
                index_b = other.index_b;
                dst_step_m = other.dst_step_m;
                offset = other.offset;
            }
            return *this;
        }

        __device__ DestinationContext& operator++() {
            offset += dst_step_k;
            ++iter_k;
            if (iter_k < IterCount::kContiguous) {
                return *this;
            }

            iter_k = 0;
            offset += dst_step_m;
            iter_m++;
            return *this;
        }

        __device__ void nexts() {
            iter_m = 0;
            index_b += kElementSize * kSizePerTile;
            if (index_b >= kSmemByteSize) {
                index_b -= kSmemByteSize;
            }
            offset = index_b;
        }
    };

    struct Context {
        SourceContext src_context_;
        DestinationContext dst_context_;

        Context() = default;
        __device__ Context(const ElementType* src, const uint32_t smem_ptr, const int k, const int m,
                           const int src_offset, const int src_offset_k, const int src_offset_m, const int src_step_m,
                           const int dst_offset, const int dst_step_m)
            : src_context_(SourceContext(src, k, m, src_offset, src_offset_k, src_offset_m, src_step_m))
            , dst_context_(DestinationContext(smem_ptr, dst_offset, dst_step_m)) {}

        __device__ Context& operator=(const Context& other) {
            if (this != &other) {
                src_context_ = other.src_context_;
                dst_context_ = other.dst_context_;
            }
            return *this;
        }

        __device__ Context& operator++() {
            ++src_context_;
            ++dst_context_;
            return *this;
        }

        __device__ void nexts() {
            src_context_.nexts();
            dst_context_.nexts();
        }
    };

    const ElementType* src_;
    const uint32_t smem_u32_;
    const int k_;
    const int m_;
    const int cta_k_;
    const int cta_m_;
    const int warp_idx_;
    const int lane_idx_;

    int iter_k_{0};
    int iter_m_{0};

    Context context_;

    DefaultIteratorA() = delete;

    __device__ void init() {
        const int warp_index_k = warp_idx_ % BlockWarpA::kContiguous;
        const int warp_index_m = warp_idx_ / BlockWarpA::kContiguous;

        const int thread_index_k = lane_idx_ % WarpNumThreadsK;
        const int thread_index_m = lane_idx_ / WarpNumThreadsK;

        const int cta_thread_offset_k = ElementCount::kContiguous * warp_index_k + thread_index_k * kAccessK;
        const int cta_thread_offset_m = ElementCount::kStrided * warp_index_m + thread_index_m;

        const int src_offset_k = cta_thread_offset_k + cta_k_;
        const int src_offset_m = cta_thread_offset_m + cta_m_;
        const int src_offset = src_offset_m * k_ + src_offset_k;

        const int dst_offset_k = cta_thread_offset_k;
        const int dst_offset_m = cta_thread_offset_m;
        const int dst_offset = (dst_offset_m * kSmemPadCtaK + dst_offset_k) * kElementSize;

        const int src_step_m = WarpAccess::kStrided * k_ - IterCount::kContiguous * WarpAccess::kContiguous;
        const int dst_step_m =
            (WarpAccess::kStrided * kSmemPadCtaK - IterCount::kContiguous * WarpAccess::kContiguous) *
            kElementSize; // ???

        context_ = Context(src_, smem_u32_, k_, m_, src_offset, src_offset_k, src_offset_m, src_step_m, dst_offset,
                           dst_step_m);
    }

    __device__ DefaultIteratorA(const ElementType* src, const uint32_t dst, const int k, const int m, const int cta_m,
                                const int cta_k, const int warp_idx, const int lane_idx)
        : src_(src)
        , smem_u32_(dst)
        , k_(k)
        , m_(m)
        , cta_k_(cta_k)
        , cta_m_(cta_m)
        , warp_idx_(warp_idx)
        , lane_idx_(lane_idx) {
        init();
    }

    __device__ void advance(const bool mask) {
#pragma unroll
        for (int i = 0; i < IterCount::kCount; ++i) {
            copy(mask);
            ++(*this);
        }
        nexts();
    }

    __device__ void next(const int batch_idx, const int batch_size, const bool mask) {
#pragma unroll
        for (int i = 0; i < batch_size; ++i) {
            if (batch_idx * batch_size + i < IterCount::kCount) {
                copy(mask);
                ++(*this);
            }
        }
    }

    template <int BatchSize>
    __device__ void next(const int batch_idx, const bool mask) {
#pragma unroll
        for (int i = 0; i < BatchSize; ++i) {
            if (batch_idx * BatchSize + i < IterCount::kCount) {
                copy(mask);
                ++(*this);
            }
        }
    }

    __device__ DefaultIteratorA& operator++() {
        if (!context_.src_context_.is_valid_m || !context_.src_context_.is_valid_k) {
            return *this;
        }

        ++context_;
        return *this;
    }

    __device__ void nexts() {
        context_.nexts();
    }

    __device__ void copy(const bool mask) {
        cp_async_a(context_.dst_context_.smem_addr + context_.dst_context_.offset,
                   (const AccessType*)(context_.src_context_.global_addr + context_.src_context_.offset),
                   context_.src_context_.is_valid_k && context_.src_context_.is_valid_m && mask);
    }
};

template <typename BlockShape_, typename WarpShape_, typename MmaShape_, int Stride, int Stages>
struct WarpIteratorA {
    using ElementType = half;
    using BlockShape = BlockShape_;
    using WarpShape = WarpShape_;
    using MmaShape = MmaShape_;

    static constexpr int kTilekM = BlockShape::kM;
    static constexpr int kTilekK = BlockShape::kK;
    static constexpr int kWarpM = WarpShape::kM;
    static constexpr int kWarpK = WarpShape::kK;
    static constexpr int kMmaM = MmaShape::kM;
    static constexpr int kMmaK = MmaShape::kK;
    static constexpr int kStride = Stride;
    static constexpr int kStages = Stages;

    static constexpr int ITER_M = kWarpM / kMmaM;
    static constexpr int ITER_K = kWarpK / kMmaK;
    static constexpr int kSharedMemorySize = sizeof(ElementType) * kTilekM * kStride;

    static_assert(kMmaM == 16 && kMmaK == 16);

    const int warp_idx_;
    const int lane_idx_;

    const uint32_t smem_;
    uint32_t addr_{0};
    int offset_k_{0};
    int offset_m_{0};
    int stage_{0};

    __device__ WarpIteratorA(const uint32_t smem_u32, const int warp_idx, const int lane_idx, const int offset_m)
        : warp_idx_(warp_idx), lane_idx_(lane_idx), smem_(smem_u32), addr_(smem_u32) {
        offset_k_ = lane_idx / 8 / 2 * 8;
        offset_m_ = lane_idx / 8 % 2 * 8 + lane_idx % 8 + warp_idx_ * kWarpM;
    }

    __device__ void load(Array<half, 8>* data, int iter_k) {
        const int kk = iter_k * kMmaK + offset_k_;
        uint32_t* data_u32 = reinterpret_cast<uint32_t*>(data);
#pragma unroll
        for (int iter_m = 0; iter_m < ITER_M;) {
            const int mm = offset_m_ + iter_m * kMmaM;
            const uint32_t src = addr_ + sizeof(half) * (mm * kStride + kk);
            load_cache(data_u32[0], data_u32[1], data_u32[2], data_u32[3], src);
            data_u32 += 4;
            iter_m += 1;
        }
    }

    __device__ void nexts() {
        ++stage_;
        if (stage_ >= kStages) {
            stage_ = 0;
        }
        addr_ = smem_ + stage_ * kSharedMemorySize;
    }
};

template <typename BlockShape_, typename WarpCount_, int WarpSize, int Stages, int SliceK>
struct DefaultIteratorB {
    static constexpr int kSliceK = SliceK;

    using BlockShape = BlockShape_;
    using WarpCount = WarpCount_;
    using AccessType = uint4;
    using ElementType = half;

    static constexpr int kStages = Stages;
    static constexpr int kWarpSize = WarpSize;
    static constexpr int kAccessSize = 128;
    static constexpr int kElementSize = sizeof(ElementType);

    using PShape = LinearShape<kSliceK, BlockShape::kN>;

    static constexpr int kAccessN = 1;
    static constexpr int kAccessK = 8;
    static constexpr int WarpNumThreadsK = const_min(const_max(PShape::kContiguous / kAccessK, 1), kWarpSize);
    static constexpr int WarpNumThreadsN = kWarpSize / WarpNumThreadsK;
    using WarpThreadA = LinearShape<WarpNumThreadsK, WarpNumThreadsN>;
    using WarpAccess = LinearShape<WarpNumThreadsK * kAccessK, WarpNumThreadsN * kAccessN>;
    using WarpAccessI =
        LinearShape<PShape::kContiguous / WarpAccess::kContiguous, PShape::kStrided / WarpAccess::kStrided>;
    using BlockWarpA = LinearShape<const_min(WarpAccessI::kContiguous, WarpCount::kMN),
                                   const_max(WarpCount::kMN / WarpAccessI::kContiguous, 1)>;

    using IterCount = LinearShape<WarpAccessI::kContiguous / BlockWarpA::kContiguous,
                                  const_max(WarpAccessI::kStrided / BlockWarpA::kStrided, 1)>;
    using ElementCount =
        LinearShape<WarpAccess::kContiguous * IterCount::kContiguous, WarpAccess::kStrided * IterCount::kStrided>;

    static constexpr int kSmemPadCtaK = kSliceK + 8;
    static constexpr int kSizePerTile = BlockShape::kN * kSmemPadCtaK;
    static constexpr int kSmemByteSize = kElementSize * kStages * kSizePerTile;

    struct SourceContext {
        mutable ElementType* global_addr;
        mutable int problem_k;
        mutable int problem_n;
        mutable int index_b{0};
        mutable int index_b_k{0};
        mutable int index_b_n{0};
        mutable int src_step_n;

        int offset{0};
        int offset_k{0};
        int offset_n{0};
        int iter_k{0};
        int iter_n{0};
        bool is_valid_k;
        bool is_valid_n;

        static constexpr int src_step_k{WarpAccess::kContiguous};

        SourceContext() = default;
        __device__ SourceContext(const ElementType* src, const int k, const int n, const int src_offset,
                                 const int src_offset_k, const int src_offset_n, const int src_step)
            : global_addr(const_cast<ElementType*>(src))
            , problem_k(k)
            , problem_n(n)
            , index_b(src_offset)
            , index_b_k(src_offset_k)
            , index_b_n(src_offset_n)
            , src_step_n(src_step)
            , offset(src_offset)
            , offset_k(src_offset_k)
            , offset_n(src_offset_n)
            , is_valid_k(offset_k < k)
            , is_valid_n(offset_n < n) {}

        __device__ SourceContext& operator=(const SourceContext& other) {
            if (this != &other) {
                global_addr = other.global_addr;
                problem_k = other.problem_k;
                problem_n = other.problem_n;
                index_b = other.index_b;
                index_b_k = other.index_b_k;
                index_b_n = other.index_b_n;
                src_step_n = other.src_step_n;
                offset = other.offset;
                offset_k = other.offset_k;
                offset_n = other.offset_n;
                is_valid_k = other.is_valid_k;
                is_valid_n = other.is_valid_n;
            }
            return *this;
        }

        __device__ SourceContext& operator++() {
            offset_k += src_step_k;
            is_valid_k = offset_k < problem_k;
            offset += src_step_k;
            ++iter_k;
            if (iter_k < IterCount::kContiguous) {
                return *this;
            }

            iter_k = 0;
            offset_k = index_b_k;
            offset_n += WarpAccess::kStrided;
            offset += src_step_n;
            is_valid_n = offset_n < problem_n;
            is_valid_k = offset_k < problem_k;
            ++iter_n;
            return *this;
        }

        __device__ void nexts() {
            iter_n = 0;
            index_b += BlockShape::kK;

            offset = index_b;
            offset_k = index_b_k;
            offset_n = index_b_n;

            is_valid_n = offset_n < problem_n;
            is_valid_k = offset_k < problem_k;
        }
    };

    struct DestinationContext {
        mutable uint32_t smem_addr;
        mutable int index_b;
        mutable int dst_step_n;

        int offset{0};
        int iter_k{0};
        int iter_n{0};

        static constexpr int dst_step_k{WarpAccess::kContiguous * kElementSize};

        DestinationContext() = default;

        __device__ DestinationContext(const uint32_t addr, const int dst_offset, const int dst_step)
            : smem_addr(addr), index_b(dst_offset), dst_step_n(dst_step), offset(dst_offset) {}

        __device__ DestinationContext& operator=(const DestinationContext& other) {
            if (this != &other) {
                smem_addr = other.smem_addr;
                index_b = other.index_b;
                dst_step_n = other.dst_step_n;
                offset = other.offset;
            }
            return *this;
        }

        __device__ DestinationContext& operator++() {
            offset += dst_step_k;
            ++iter_k;
            if (iter_k < IterCount::kContiguous) {
                return *this;
            }

            iter_k = 0;
            offset += dst_step_n;
            iter_n++;
            return *this;
        }

        __device__ void nexts() {
            iter_n = 0;
            index_b += kElementSize * kSizePerTile;
            if (index_b >= kSmemByteSize) {
                index_b -= kSmemByteSize;
            }
            offset = index_b;
        }
    };

    struct Context {
        SourceContext src_context_;
        DestinationContext dst_context_;

        Context() = default;
        __device__ Context(const ElementType* src, const uint32_t smem_ptr, const int k, const int m,
                           const int src_offset, const int src_offset_k, const int src_offset_n, const int src_step_n,
                           const int dst_offset, const int dst_step_n)
            : src_context_(SourceContext(src, k, m, src_offset, src_offset_k, src_offset_n, src_step_n))
            , dst_context_(DestinationContext(smem_ptr, dst_offset, dst_step_n)) {}

        __device__ Context& operator=(const Context& other) {
            if (this != &other) {
                src_context_ = other.src_context_;
                dst_context_ = other.dst_context_;
            }
            return *this;
        }

        __device__ Context& operator++() {
            ++src_context_;
            ++dst_context_;
            return *this;
        }

        __device__ void nexts() {
            src_context_.nexts();
            dst_context_.nexts();
        }
    };

    const half* src_;
    const uint32_t smem_u32_;
    const int k_;
    const int n_;
    const int cta_k_;
    const int cta_n_;
    const int warp_idx_;
    const int lane_idx_;

    int iter_k_{0};
    int iter_n_{0};

    Context context_;

    DefaultIteratorB() = default;

    __device__ void init() {
        const int warp_index_k = warp_idx_ % BlockWarpA::kContiguous;
        const int warp_index_n = warp_idx_ / BlockWarpA::kContiguous;

        const int thread_index_k = lane_idx_ % WarpNumThreadsK;
        const int thread_index_n = lane_idx_ / WarpNumThreadsK;

        const int cta_thread_offset_k = ElementCount::kContiguous * warp_index_k + thread_index_k * kAccessK;
        const int cta_thread_offset_n = ElementCount::kStrided * warp_index_n + thread_index_n;

        const int src_offset_k = cta_thread_offset_k + cta_k_;
        const int src_offset_n = cta_thread_offset_n + cta_n_;
        const int src_offset = src_offset_n * k_ + src_offset_k;

        const int dst_offset_k = cta_thread_offset_k;
        const int dst_offset_n = cta_thread_offset_n;
        const int dst_offset = (dst_offset_n * kSmemPadCtaK + dst_offset_k) * kElementSize;

        const int src_step_n = WarpAccess::kStrided * k_ - IterCount::kContiguous * WarpAccess::kContiguous;
        const int dst_step_n =
            (WarpAccess::kStrided * kSmemPadCtaK - IterCount::kContiguous * WarpAccess::kContiguous) * kElementSize;

        context_ = Context(src_, smem_u32_, k_, n_, src_offset, src_offset_k, src_offset_n, src_step_n, dst_offset,
                           dst_step_n);
    }

    __device__ DefaultIteratorB(const ElementType* src, const uint32_t dst, const int k, const int n, const int cta_n,
                                const int cta_k, const int warp_idx, const int lane_idx)
        : src_(src)
        , smem_u32_(dst)
        , k_(k)
        , n_(n)
        , cta_k_(cta_k)
        , cta_n_(cta_n)
        , warp_idx_(warp_idx)
        , lane_idx_(lane_idx) {
        init();
    }

    __device__ void advance(bool mask) {
#pragma unroll
        for (int i = 0; i < IterCount::kCount; ++i) {
            copy(mask);
            ++(*this);
        }
        nexts();
    }

    template <int BATCH_SIZE>
    __device__ void next(const int batch_idx, const bool mask) {
#pragma unroll
        for (int i = 0; i < BATCH_SIZE; ++i) {
            if (batch_idx * BATCH_SIZE + i < IterCount::kCount) {
                copy(mask);
                ++(*this);
            }
        }
    }

    __device__ void next(const int batch_idx, const int batch_size, const bool mask) {
#pragma unroll
        for (int i = 0; i < batch_size; ++i) {
            if (batch_idx * batch_size + i < IterCount::kCount) {
                copy(mask);
                ++(*this);
            }
        }
    }

    __device__ DefaultIteratorB& operator++() {
        if (!context_.src_context_.is_valid_n || !context_.src_context_.is_valid_k) {
            return *this;
        }

        ++context_;
        return *this;
    }

    __device__ void nexts() {
        context_.nexts();
    }

    __device__ void copy(bool mask) {
        cp_async_b(context_.dst_context_.smem_addr + context_.dst_context_.offset,
                   (const AccessType*)(context_.src_context_.global_addr + context_.src_context_.offset),
                   context_.src_context_.is_valid_k && context_.src_context_.is_valid_n && mask);
    }
};

template <typename BlockShape_, typename WarpShape_, typename MmaShape_, int Stride, int Stages>
struct WarpIteratorB {
    using ElementType = half;
    using BlockShape = BlockShape_;
    using WarpShape = WarpShape_;
    using MmaShape = MmaShape_;
    static constexpr int kTilekN = BlockShape::kN;
    static constexpr int kTilekK = BlockShape::kK;
    static constexpr int kWarpN = WarpShape::kN;
    static constexpr int kWarpK = WarpShape::kK;
    static constexpr int kMmaN = MmaShape::kN;
    static constexpr int kMmaK = MmaShape::kK;
    static constexpr int kStride = Stride;
    static constexpr int kStages = Stages;

    static constexpr int kNum = kWarpN == 8 ? 2 : 4;
    static constexpr int ITER_N = kWarpN / kMmaN;
    static constexpr int ITER_K = kWarpK / kMmaK;
    static constexpr int kSharedMemorySize = sizeof(ElementType) * kTilekN * kStride;

    static_assert(kMmaN == 8 && kMmaK == 16);

    const int warp_idx_;
    const int lane_idx_;

    const uint32_t smem_;
    uint32_t addr_;

    int offset_k_;
    int offset_n_;

    int stage_{0};

    __device__ WarpIteratorB(const uint32_t smem_u32, const int warp_idx, const int lane_idx, const int offset_k)
        : warp_idx_(warp_idx), lane_idx_(lane_idx), smem_(smem_u32), addr_(smem_u32) {
        offset_k_ = lane_idx / 8 % 2 * 8;
        offset_n_ = lane_idx / 8 / 2 * 8 + lane_idx % 8;

        if constexpr (kNum == 2) {
            offset_n_ -= (lane_idx / 8 / 2 * 8);
        }

        offset_n_ += warp_idx_ * kWarpN;
    }

    __device__ void load(Array<half, 4>* data, int iter_k) {
        const int kk = iter_k * kMmaK + offset_k_;
        uint32_t* data_u32 = reinterpret_cast<uint32_t*>(data);
#pragma unroll
        for (int iter_n = 0; iter_n < ITER_N;) {
            const int nn = offset_n_ + iter_n * kMmaN;
            const uint32_t src = addr_ + sizeof(half) * (nn * kStride + kk);
            if constexpr (kNum == 4) {
                load_cache(data_u32[0], data_u32[1], data_u32[2], data_u32[3], src);
                data_u32 += 4;
                iter_n += 2;
            } else {
                load_cache(data_u32[0], data_u32[1], src);
                data_u32 += 2;
                iter_n += 1;
            }
        }
    }

    __device__ void nexts() {
        ++stage_;
        if (stage_ >= kStages) {
            stage_ = 0;
        }
        addr_ = smem_ + stage_ * kSharedMemorySize;
    }
};

template <typename BlockShape_, typename WarpCount_, int WarpSize, int Stages, int SliceK, int Slices, int GroupSize>
struct DefaultIteratorS {
    static constexpr int kSliceK = SliceK;
    static constexpr int kSlices = Slices;

    using BlockShape = BlockShape_;
    using WarpCount = WarpCount_;
    using AccessType = uint4;
    using ElementType = half;

    static constexpr int kStages = Stages;
    static constexpr int kWarpSize = WarpSize;
    static constexpr int kAccessSize = 128;
    static constexpr int kElementSize = sizeof(ElementType);

    static constexpr int kGroupSize = GroupSize;

    using PShape = LinearShape<BlockShape::kN, kSliceK>;

    static constexpr int kAccessN = 8;
    static constexpr int kAccessK = kGroupSize;
    static constexpr int WarpNumThreadsN = const_min(const_max(PShape::kContiguous / kAccessN, 1), kWarpSize);
    static constexpr int WarpNumThreadsK = kWarpSize / WarpNumThreadsN;

    using WarpThreadA = LinearShape<WarpNumThreadsN, WarpNumThreadsK>;
    using WarpAccess = LinearShape<WarpNumThreadsN * kAccessN, WarpNumThreadsK * kAccessK>;
    using WarpAccessI =
        LinearShape<PShape::kContiguous / WarpAccess::kContiguous, PShape::kStrided / WarpAccess::kStrided>;
    using BlockWarpA = LinearShape<const_min(WarpAccessI::kContiguous, WarpCount::kMN),
                                   const_max(WarpCount::kMN / WarpAccessI::kContiguous, 1)>;
    using IterCount = LinearShape<WarpAccessI::kContiguous / BlockWarpA::kContiguous,
                                  const_max(WarpAccessI::kStrided / BlockWarpA::kStrided, 1)>;
    using ElementCount =
        LinearShape<WarpAccess::kContiguous * IterCount::kContiguous, WarpAccess::kStrided * IterCount::kStrided>;

    static constexpr int kSmemPadCtaN = BlockShape::kN;
    static constexpr int kSizePerStage = std::max(kSliceK / kGroupSize, 1) * kSmemPadCtaN;
    static constexpr int kSmemByteSize = kElementSize * kStages * kSizePerStage;

    struct SourceContext {
        mutable ElementType* global_addr;
        mutable int max_k;
        mutable int problem_k;
        mutable int problem_n;
        mutable int index_b{0};
        mutable int index_b_k{0};
        mutable int index_b_n{0};
        mutable int src_step_k;

        int offset{0};
        int offset_k{0};
        int offset_n{0};
        int iter_k{0};
        int iter_n{0};
        bool is_valid_k;
        bool is_valid_n;

        static constexpr int src_step_n{WarpAccess::kContiguous};

        SourceContext() = default;
        __device__ SourceContext(const ElementType* src, const int m_k, const int k, const int n, const int src_offset,
                                 const int src_offset_k, const int src_offset_n, const int src_step)
            : global_addr(const_cast<ElementType*>(src))
            , max_k(m_k)
            , problem_k(k)
            , problem_n(n)
            , index_b(src_offset)
            , index_b_k(src_offset_k)
            , index_b_n(src_offset_n)
            , src_step_k(src_step)
            , offset(src_offset)
            , offset_k(src_offset_k)
            , offset_n(src_offset_n)
            , is_valid_k(offset_k < m_k)
            , is_valid_n(offset_n < n) {}

        __device__ SourceContext& operator=(const SourceContext& other) {
            if (this != &other) {
                global_addr = other.global_addr;
                max_k = other.max_k;
                problem_k = other.problem_k;
                problem_n = other.problem_n;
                index_b = other.index_b;
                index_b_k = other.index_b_k;
                index_b_n = other.index_b_n;
                src_step_k = other.src_step_n;
                offset = other.offset;
                offset_k = other.offset_k;
                offset_n = other.offset_n;
                is_valid_k = other.is_valid_k;
                is_valid_n = other.is_valid_n;
            }
            return *this;
        }

        __device__ SourceContext& operator++() {
            offset_n += src_step_n;
            offset += src_step_n;
            is_valid_n = offset_n < problem_n;
            ++iter_n;
            if (iter_n < IterCount::kContiguous) {
                return *this;
            }

            iter_n = 0;
            offset_n = index_b_n;
            offset_k += kGroupSize;
            offset += src_step_k;
            is_valid_n = offset_n < problem_n;
            is_valid_k = offset_k < max_k;
            iter_k++;
            return *this;
        }

        __device__ void nexts() {
            iter_n = 0;
            index_b_k += BlockShape::kK;
            index_b = (index_b_k / kGroupSize) * problem_n + index_b_n;

            offset = index_b;
            offset_k = index_b_k;
            offset_n = index_b_n;

            is_valid_n = offset_n < problem_n;
            is_valid_k = offset_k < max_k;
        }
    };

    struct DestinationContext {
        mutable uint32_t smem_addr;
        mutable int index_b;
        mutable int dst_step_k;

        int offset{0};
        int iter_k{0};
        int iter_n{0};

        static constexpr int dst_step_n{WarpAccess::kContiguous * kElementSize};

        DestinationContext() = default;

        __device__ DestinationContext(const uint32_t addr, const int dst_offset, const int dst_step)
            : smem_addr(addr), index_b(dst_offset), dst_step_k(dst_step), offset(dst_offset) {}

        __device__ DestinationContext& operator=(const DestinationContext& other) {
            if (this != &other) {
                smem_addr = other.smem_addr;
                index_b = other.index_b;
                dst_step_k = other.dst_step_k;
                offset = other.offset;
            }
            return *this;
        }

        __device__ DestinationContext& operator++() {
            offset += dst_step_n;
            ++iter_n;
            if (iter_n < IterCount::kContiguous) {
                return *this;
            }

            iter_n = 0;
            offset += dst_step_k;
            iter_k++;
            return *this;
        }

        __device__ void nexts() {
            iter_n = 0;
            index_b += kElementSize * kSizePerStage;
            if (index_b >= kSmemByteSize) {
                index_b -= kSmemByteSize;
            }
            offset = index_b;
        }
    };

    struct Context {
        SourceContext src_context_;
        DestinationContext dst_context_;

        Context() = default;
        __device__ Context(const ElementType* src, const uint32_t smem_ptr, const int max_k, const int k, const int n,
                           const int src_offset, const int src_offset_k, const int src_offset_n, const int src_step_k,
                           const int dst_offset, const int dst_step_k)
            : src_context_(SourceContext(src, max_k, k, n, src_offset, src_offset_k, src_offset_n, src_step_k))
            , dst_context_(DestinationContext(smem_ptr, dst_offset, dst_step_k)) {}

        __device__ Context& operator=(const Context& other) {
            if (this != &other) {
                src_context_ = other.src_context_;
                dst_context_ = other.dst_context_;
            }
            return *this;
        }

        __device__ Context& operator++() {
            ++src_context_;
            ++dst_context_;
            return *this;
        }

        __device__ void nexts() {
            src_context_.nexts();
            dst_context_.nexts();
        }
    };

    const half* src_;
    const uint32_t smem_u32_;
    const int k_;
    const int n_;
    const int cta_k_;
    const int cta_n_;
    const int warp_idx_;
    const int lane_idx_;

    int iter_k_{0};
    int iter_n_{0};

    bool is_out_of_bound_;
    bool is_out_of_bound_n_;

    Context context_;

    template <typename T>
    __inline__ __device__ void cp_async_ca(const uint32_t smem_u32, const T* __restrict__ src, bool mask) {
        constexpr int cp_size = sizeof(T);

        int actual_size = mask ? cp_size : 0;
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_u32), "l"(src), "n"(cp_size), "r"(actual_size));
    }

    DefaultIteratorS() = delete;

    __device__ void init() {
        const int warp_index_n = warp_idx_ % BlockWarpA::kContiguous;

        const int thread_index_n = lane_idx_ % WarpNumThreadsN;
        const int thread_index_k = lane_idx_ / WarpNumThreadsN;

        const int cta_thread_offset_n = ElementCount::kContiguous * warp_index_n + thread_index_n * kAccessN;
        const int cta_thread_offset_k = thread_index_k * kAccessK;

        const int max_k_ = (k_ + kGroupSize - 1) / kGroupSize * kGroupSize;

        is_out_of_bound_ = cta_thread_offset_k >= kSliceK;
        is_out_of_bound_n_ = cta_thread_offset_n >= BlockShape::kN;

        const int src_offset_n = cta_thread_offset_n + cta_n_;
        const int src_offset_k = cta_thread_offset_k + cta_k_;
        const int src_offset = src_offset_k / kGroupSize * n_ + src_offset_n;

        const int dst_offset_k = cta_thread_offset_k;
        const int dst_offset_n = cta_thread_offset_n;
        const int dst_offset = (dst_offset_k / kGroupSize * kSmemPadCtaN + dst_offset_n) * kElementSize;
        const int src_step_k = n_ - IterCount::kContiguous * WarpAccess::kContiguous;
        const int dst_step_k = (BlockShape::kN - IterCount::kContiguous * WarpAccess::kContiguous) * kElementSize;

        context_ = Context(src_, smem_u32_, max_k_, k_, n_, src_offset, src_offset_k, src_offset_n, src_step_k,
                           dst_offset, dst_step_k);
    }

    __device__ DefaultIteratorS(const ElementType* src, uint32_t dst, int n, int k, int cta_n, int cta_k, int warp_idx,
                                int lane_idx)
        : src_(src)
        , smem_u32_(dst)
        , k_(k)
        , n_(n)
        , cta_k_(cta_k)
        , cta_n_(cta_n)
        , warp_idx_(warp_idx)
        , lane_idx_(lane_idx) {
        init();
    }

    __device__ void advance(bool mask) {
        if (is_out_of_bound_) {
            return;
        }

#pragma unroll
        for (int i = 0; i < IterCount::kCount; ++i) {
            copy(mask);
            ++(*this);
        }
        nexts();
    }

    template <int BATCH_SIZE>
    __device__ void next(const int batch_idx, const bool mask) {
        if (is_out_of_bound_) {
            return;
        }

#pragma unroll
        for (int i = 0; i < BATCH_SIZE; ++i) {
            if (batch_idx * BATCH_SIZE + i < IterCount::kCount) {
                copy(mask);
                ++(*this);
            }
        }
    }

    __device__ void next(const int batch_idx, const int batch_size, const bool mask) {
        if (is_out_of_bound_) {
            return;
        }

#pragma unroll
        for (int i = 0; i < batch_size; ++i) {
            if (batch_idx * batch_size + i < IterCount::kCount) {
                copy(mask);
                ++(*this);
            }
        }
    }

    __device__ DefaultIteratorS& operator++() {
        if (!context_.src_context_.is_valid_n || !context_.src_context_.is_valid_k) {
            return *this;
        }

        ++context_;
        return *this;
    }

    __device__ void nexts() {
        context_.nexts();
    }

    __device__ void copy(bool mask) {
        cp_async_ca(context_.dst_context_.smem_addr + context_.dst_context_.offset,
                    (const AccessType*)(context_.src_context_.global_addr + context_.src_context_.offset),
                    (!is_out_of_bound_n_) && (!is_out_of_bound_) && context_.src_context_.is_valid_k &&
                        context_.src_context_.is_valid_n && mask);
    }
};

template <typename BlockShape_, typename WarpShape_, typename MmaShape_, int Stride, int Stages, int SliceK,
          int GroupSize>
struct WarpIteratorS {
    using ElementType = half;
    using BlockShape = BlockShape_;
    using WarpShape = WarpShape_;
    using MmaShape = MmaShape_;

    static constexpr int kWarpN = WarpShape::kN;
    static constexpr int kWarpK = WarpShape::kK;
    static constexpr int kMmaN = MmaShape::kN;
    static constexpr int kMmaK = MmaShape::kK;
    static constexpr int kStride = Stride;
    static constexpr int kStages = Stages;
    static constexpr int kGroupSize = GroupSize;
    static constexpr int kSliceK = SliceK;

    static constexpr int ITER_N = kWarpN / (kMmaN * 4);

    static_assert(kMmaN == 8 && kMmaK == 16);

    const uint32_t smem_;
    const int warp_idx_;
    const int lane_idx_;
    const int offset_n_int4_;
    uint32_t addr_;
    int stage_{0};

    __device__ WarpIteratorS(const uint32_t& smem_u32, const int warp_idx, const int lane_idx, const int offset_k)
        : smem_(smem_u32)
        , addr_(smem_u32)
        , warp_idx_(warp_idx)
        , lane_idx_(lane_idx)
        , offset_n_int4_(warp_idx_ * kWarpN) {}

    __device__ void load_scale(int iter_k, half* scales) {
        const int row = iter_k * kMmaK / kGroupSize;
        uint32_t* dst = reinterpret_cast<uint32_t*>(scales);

#pragma unroll
        for (int iter_n = 0; iter_n < ITER_N; iter_n++) {
            const int col = offset_n_int4_ + iter_n * 32 + lane_idx_ % 4 * 8;
            auto src = addr_ + sizeof(half) * (row * kStride + col);
            load(dst[iter_n * 4 + 0], dst[iter_n * 4 + 1], dst[iter_n * 4 + 2], dst[iter_n * 4 + 3], src);
        }
    }

    __device__ void nexts() {
        ++stage_;
        if (stage_ >= kStages) {
            stage_ = 0;
        }

        addr_ = smem_ + stage_ * sizeof(ElementType) * (max(kSliceK / kGroupSize, 1)) * kStride;
    }
};

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16
