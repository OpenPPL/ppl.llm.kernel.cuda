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

#include "gemm.h"
#include "heuristic.h"
#include "reduce.cuh"
#include "shape.h"

#include "ppl/kernel/llm/cuda/common/general_include.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

struct GemmRunner {
    virtual ~GemmRunner() = default;

    virtual ppl::common::RetCode gemm(void* C, const void* A, const void* B, const void* S, const void* bias, void* workspace,
                            int M, int N, int K, int workspace_size, GemmAlgorithm_t* algo, cudaStream_t st) = 0;

    virtual ppl::common::RetCode get_algo(int M, int N, int K, int workspace_size, GemmAlgorithm_t* algo) = 0;
};

template <int GroupSize, bool Bias>
struct GemmRunnerImpl : public GemmRunner {
    std::shared_ptr<cudaDeviceProp> props_;
    int max_active_ctas_{};
    int device_id_{-1};
    int sm_{};

    static constexpr int kGroupSize = GroupSize;
    static constexpr bool kBias = Bias;

    GemmRunnerImpl() {
        if (!props_) {
            props_ = std::make_shared<cudaDeviceProp>();
            cudaGetDevice(&device_id_);
            cudaGetDeviceProperties(props_.get(), device_id_);
        }
        sm_ = get_sm_version();
    };

    int get_sm_version() {
        return props_->major * 10 + props_->minor;
    }

    template <typename Kernel>
    inline int compute_occupancy_for_kernel() const {
        static constexpr int cSlices = Kernel::Mma::kSlices;
        static constexpr int cSmemSizeA = Kernel::IteratorA::kSmemByteSize * cSlices;
        static constexpr int cSmemSizeB = Kernel::IteratorB::kSmemByteSize * cSlices;
        static constexpr int cSmemSizeS = Kernel::IteratorS::kSmemByteSize * cSlices;
        static constexpr int cSmemSizeC =
            sizeof(float) * Kernel::BlockShape::kM * Kernel::BlockShape::kN * 4; // *4 for int4
        static constexpr int cSmemByteSize = std::max(cSmemSizeA + cSmemSizeB + cSmemSizeS, cSmemSizeC);
        const int max_shared_memory = props_->sharedMemPerMultiprocessor;
        if (max_shared_memory < cSmemByteSize) {
            return 0;
        }

        cudaError_t status =
            cudaFuncSetAttribute(gemm_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, cSmemByteSize);
        if (status == cudaError::cudaErrorInvalidValue) {
            status = cudaGetLastError();
            return 0;
        }

        int max_active_ctas;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_ctas, gemm_kernel<Kernel>, Kernel::Mma::kThreads,
                                                      cSmemByteSize);

        return max_active_ctas;
    }

    template <int BT_M, int BT_N, int BT_K, int WT_M, int WT_N, int WT_K>
    ppl::common::RetCode dispatch_config(const GemmConfig& gemm_config, int* occupancy) const {
        using BlockShape_ = BlockShape<BT_M, BT_N, BT_K>;
        using WarpShape_ = WarpShape<WT_M, WT_N, WT_K>;

        switch (gemm_config.stages) {
            case 2:
                using GemmType2 = Gemm<BlockShape_, WarpShape_, 2, true, kBias, kGroupSize>;
                *occupancy = compute_occupancy_for_kernel<GemmType2>();
                break;
            case 3:
                using GemmType3 = Gemm<BlockShape_, WarpShape_, 3, true, kBias, kGroupSize>;
                *occupancy = compute_occupancy_for_kernel<GemmType3>();
                break;
            case 4:
                using GemmType4 = Gemm<BlockShape_, WarpShape_, 4, true, kBias, kGroupSize>;
                *occupancy = compute_occupancy_for_kernel<GemmType4>();
                break;
            default:
                LOG(ERROR) << "config is invalid for mixed type stages";
                return ppl::common::RC_INVALID_VALUE;
        }
        return ppl::common::RC_SUCCESS;
    }

    ppl::common::RetCode get_occupancy(const GemmConfig& gemm_config, int* occupancy) const {
        switch (gemm_config.tile_config) {
            case TileConfig::CtaShape16x64x64_WarpShape16x16x64:
                return dispatch_config<16, 64, 64, 16, 16, 64>(gemm_config, occupancy);
            case TileConfig::CtaShape16x16x512_WarpShape16x16x128:
                return dispatch_config<16, 16, 512, 16, 16, 128>(gemm_config, occupancy);
            case TileConfig::CtaShape16x32x256_WarpShape16x16x128:
                return dispatch_config<16, 32, 256, 16, 16, 128>(gemm_config, occupancy);
            case TileConfig::CtaShape16x64x256_WarpShape16x16x128:
                return dispatch_config<16, 64, 256, 16, 16, 128>(gemm_config, occupancy);
            case TileConfig::CtaShape16x64x128_WarpShape16x16x128:
                return dispatch_config<16, 64, 128, 16, 16, 128>(gemm_config, occupancy);

            case TileConfig::CtaShape32x64x64_WarpShape32x16x64:
                return dispatch_config<32, 64, 64, 32, 16, 64>(gemm_config, occupancy);
            case TileConfig::CtaShape32x64x128_WarpShape32x16x128:
                return dispatch_config<32, 64, 128, 32, 16, 128>(gemm_config, occupancy);
            case TileConfig::CtaShape32x64x256_WarpShape32x16x128:
                return dispatch_config<32, 64, 256, 32, 16, 128>(gemm_config, occupancy);
            case TileConfig::CtaShape32x32x128_WarpShape32x16x128:
                return dispatch_config<32, 32, 128, 32, 16, 128>(gemm_config, occupancy);
            case TileConfig::CtaShape32x32x256_WarpShape32x16x128:
                return dispatch_config<32, 32, 256, 32, 16, 128>(gemm_config, occupancy);

            case TileConfig::CtaShape64x32x128_WarpShape32x16x128:
                return dispatch_config<64, 32, 128, 32, 16, 128>(gemm_config, occupancy);
            case TileConfig::CtaShape64x32x64_WarpShape32x16x64:
                return dispatch_config<64, 32, 64, 32, 16, 64>(gemm_config, occupancy);

            default:
                LOG(ERROR) << "config is invalid for mixed type shape";
                return ppl::common::RC_INVALID_VALUE;
        }
        return  ppl::common::RC_SUCCESS;
    }

    ppl::common::RetCode get_occupancies(const std::vector<GemmConfig>& candidate_configs, std::vector<int>& occupancies) {
        for (int i = 0; i < (int)candidate_configs.size(); i++) {
            auto rc = get_occupancy(candidate_configs[i], &occupancies[i]);
            if (rc != ppl::common::RC_SUCCESS)
                return rc;
        }
        return  ppl::common::RC_SUCCESS;
    }

    template <int BT_M, int BT_N, int BT_K, int WT_M, int WT_N, int WT_K, int Stages, bool IsSplitK>
    ppl::common::RetCode dispatch_gemm(void* C, const void* A, const void* B, const void* S, const void* bias, void* workspace, int M,
                       int N, int K, GemmConfig gemm_config, cudaStream_t stream) {
        using BlockShape_ = BlockShape<BT_M, BT_N, BT_K>;
        using WarpShape_ = WarpShape<WT_M, WT_N, WT_K>;
        using Gemm = Gemm<BlockShape_, WarpShape_, Stages, true, kBias, kGroupSize>;

        static constexpr int Slices = Gemm::Mma::kSlices;
        static constexpr int SmemSizeA = Gemm::IteratorA::kSmemByteSize * Slices;
        static constexpr int SmemSizeB = Gemm::IteratorB::kSmemByteSize * Slices;
        static constexpr int SmemSizeS = Gemm::IteratorS::kSmemByteSize * Slices;
        static constexpr int SmemSizeC = sizeof(float) * BT_M * BT_N * 4; // *4 for int4
        static constexpr int SmemByteSize = std::max(SmemSizeA + SmemSizeB + SmemSizeS, SmemSizeC);

        cudaFuncSetAttribute(gemm_kernel<Gemm>, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemByteSize);

        const int grid_x = (N + BT_N - 1) / BT_N;
        const int grid_y = (M + BT_M - 1) / BT_M;

        const int slice_batch = (K / gemm_config.split_k_factor + BT_K - 1) / BT_K * BT_K;
        const int grid_z = (K + slice_batch - 1) / slice_batch;

        constexpr int block_size = Gemm::Mma::kThreads;
        dim3 grid_size(grid_x, grid_y, grid_z);

        gemm_kernel<Gemm><<<grid_size, block_size, SmemByteSize, stream>>>(
            C, A, B, S, M, N, K, slice_batch, workspace, bias
        );

        if constexpr (IsSplitK) {
            return reduce(reinterpret_cast<const __half*>(workspace), (half*)C, M, N * 4, grid_z, stream);
        }
        return  ppl::common::RC_SUCCESS;
    }

    template <int BT_M, int BT_N, int BT_K, int WT_M, int WT_N, int WT_K, int Stages>
    ppl::common::RetCode dispatch_gemm(void* C, const void* A, const void* B, const void* S, const void* bias, void* workspace, int M,
                       int N, int K, GemmConfig gemm_config, cudaStream_t st) {
        switch (gemm_config.split_k_factor) {
            case 1:
                return dispatch_gemm<BT_M, BT_N, BT_K, WT_M, WT_N, WT_K, Stages, false>(
                    C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return dispatch_gemm<BT_M, BT_N, BT_K, WT_M, WT_N, WT_K, Stages, true>(
                    C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            default:
                LOG(ERROR) << "config is invalid for mixed type splitk";
                return ppl::common::RC_INVALID_VALUE;
        }
        return ppl::common::RC_SUCCESS;
    }

    template <int BT_M, int BT_N, int BT_K, int WT_M, int WT_N, int WT_K>
    ppl::common::RetCode dispatch_gemm_stages(void* C, const void* A, const void* B, const void* S, const void* bias, void* workspace,
                              int M, int N, int K, GemmConfig gemm_config, cudaStream_t st) {
        switch (gemm_config.stages) {
            case 2:
                return dispatch_gemm<BT_M, BT_N, BT_K, WT_M, WT_N, WT_K, 2>(
                    C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            case 3:
                return dispatch_gemm<BT_M, BT_N, BT_K, WT_M, WT_N, WT_K, 3>(
                    C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            case 4:
                return dispatch_gemm<BT_M, BT_N, BT_K, WT_M, WT_N, WT_K, 4>(
                    C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            default:
                LOG(ERROR) << "config is invalid for mixed type stages";
                return ppl::common::RC_INVALID_VALUE;
        }
        return ppl::common::RC_SUCCESS;
    }

    ppl::common::RetCode dispatch_gemm_shape(void* C, const void* A, const void* B, const void* S, const void* bias, void* workspace,
                             int M, int N, int K, GemmConfig gemm_config, cudaStream_t st) {
        switch (gemm_config.tile_config) {
            case TileConfig::CtaShape16x64x64_WarpShape16x16x64:
                return dispatch_gemm_stages<16, 64, 64, 16, 16, 64>(C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            case TileConfig::CtaShape16x16x512_WarpShape16x16x128:
                return dispatch_gemm_stages<16, 16, 512, 16, 16, 128>(C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            case TileConfig::CtaShape16x32x256_WarpShape16x16x128:
                return dispatch_gemm_stages<16, 32, 256, 16, 16, 128>(C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            case TileConfig::CtaShape16x64x256_WarpShape16x16x128:
                return dispatch_gemm_stages<16, 64, 256, 16, 16, 128>(C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            case TileConfig::CtaShape16x64x128_WarpShape16x16x128:
                return dispatch_gemm_stages<16, 64, 128, 16, 16, 128>(C, A, B, S, bias, workspace, M, N, K, gemm_config, st);

            case TileConfig::CtaShape32x64x64_WarpShape32x16x64:
                return dispatch_gemm_stages<32, 64, 64, 32, 16, 64>(C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            case TileConfig::CtaShape32x64x128_WarpShape32x16x128:
                return dispatch_gemm_stages<32, 64, 128, 32, 16, 128>(C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            case TileConfig::CtaShape32x64x256_WarpShape32x16x128:
                return dispatch_gemm_stages<32, 64, 256, 32, 16, 128>(C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            case TileConfig::CtaShape32x32x128_WarpShape32x16x128:
                return dispatch_gemm_stages<32, 32, 128, 32, 16, 128>(C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            case TileConfig::CtaShape32x32x256_WarpShape32x16x128:
                return dispatch_gemm_stages<32, 32, 256, 32, 16, 128>(C, A, B, S, bias, workspace, M, N, K, gemm_config, st);

            case TileConfig::CtaShape64x32x128_WarpShape32x16x128:
                return dispatch_gemm_stages<64, 32, 128, 32, 16, 128>(C, A, B, S, bias, workspace, M, N, K, gemm_config, st);
            case TileConfig::CtaShape64x32x64_WarpShape32x16x64:
                return dispatch_gemm_stages<64, 32, 64, 32, 16, 64>(C, A, B, S, bias, workspace, M, N, K, gemm_config, st);

            default:
                LOG(ERROR) << "config is invalid for mixed type shape";
                return ppl::common::RC_INVALID_VALUE;
        }
        return ppl::common::RC_SUCCESS;
    }

    GemmConfig get_config_from_algo(GemmAlgorithm_t* algo) {
        GemmConfig config{TileConfig::Undefined, SplitKStyle::NO_SPLIT_K, -1, -1};
        config.tile_config = tile_configs[algo->data[0]];
        config.split_k_factor = algo->data[1];
        config.stages = algo->data[2];
        return config;
    }

    static ppl::common::RetCode can_implement(int M, int N, int K, int workspace_size, GemmAlgorithm_t* algo) {
        if (N % 8 != 0 || K % 64 != 0) {
            return ppl::common::RC_UNSUPPORTED;
        }

        if (algo == nullptr || algo->data[0] >= 11 || algo->data[1] >= 9 || algo->data[2] != 3) {
            return ppl::common::RC_INVALID_VALUE;
        }

        TileShape tile_shape = get_cta_shape_for_config(tile_configs[algo->data[0]]);
        if (!is_config_valid(tile_shape, M, N, K, workspace_size, algo->data[1], kGroupSize)) {
            return ppl::common::RC_INVALID_VALUE;
        }

        return ppl::common::RC_SUCCESS;
    }

    GemmAlgorithm_t get_algo_from_config(const GemmConfig& config) const {
        GemmAlgorithm_t algo;
        switch (config.tile_config) {
            case TileConfig::CtaShape16x64x64_WarpShape16x16x64:
                algo.data[0] = 0;
                break;
            case TileConfig::CtaShape16x16x512_WarpShape16x16x128:
                algo.data[0] = 1;
                break;
            case TileConfig::CtaShape16x32x256_WarpShape16x16x128:
                algo.data[0] = 2;
                break;
            case TileConfig::CtaShape16x64x256_WarpShape16x16x128:
                algo.data[0] = 3;
                break;
            case TileConfig::CtaShape16x64x128_WarpShape16x16x128:
                algo.data[0] = 4;
                break;

            case TileConfig::CtaShape32x64x64_WarpShape32x16x64:
                algo.data[0] = 5;
                break;
            case TileConfig::CtaShape32x64x128_WarpShape32x16x128:
                algo.data[0] = 6;
                break;
            case TileConfig::CtaShape32x64x256_WarpShape32x16x128:
                algo.data[0] = 7;
                break;
            case TileConfig::CtaShape32x32x128_WarpShape32x16x128:
                algo.data[0] = 8;
                break;
            case TileConfig::CtaShape32x32x256_WarpShape32x16x128:
                algo.data[0] = 9;
                break;

            case TileConfig::CtaShape64x32x128_WarpShape32x16x128:
                algo.data[0] = 10;
                break;
            case TileConfig::CtaShape64x32x64_WarpShape32x16x64:
                algo.data[0] = 11;
                break;
            default:
                LOG(ERROR) << "config is invalid algo";
                return algo;
        }
        algo.data[1] = config.split_k_factor;
        algo.data[2] = config.stages;
        return algo;
    }

    ppl::common::RetCode get_algo(int M, int N, int K, int workspace_size, GemmAlgorithm_t* algo) override {
        GemmConfig config;

        if (algo == nullptr) {
            return ppl::common::RC_INVALID_VALUE;
        }

        int multi_processor_count_ = props_->multiProcessorCount;
        const int split_k_limit = 8;

        std::vector<GemmConfig> candidate_configs = get_candidate_configs(sm_, M, false);
        std::vector<int> occupancies(candidate_configs.size());

        get_occupancies(candidate_configs, occupancies);

        static constexpr int num_experts = 1;
        config = estimate_best_config_from_occupancies(
            candidate_configs, occupancies, M, N, K, num_experts, split_k_limit,
            workspace_size, multi_processor_count_, kGroupSize);
        if (config.tile_config == TileConfig::Undefined || config.tile_config == TileConfig::ChooseWithHeuristic) {
            return ppl::common::RC_UNSUPPORTED;
        }
        *algo = get_algo_from_config(config);
        return ppl::common::RC_SUCCESS;
    }

    ppl::common::RetCode gemm(void* C, const void* A, const void* B, const void* S, const void* bias, void* workspace, int M,
                    int N, int K, int workspace_size, GemmAlgorithm_t* algo, cudaStream_t st) override {
        GemmConfig config;
        ppl::common::RetCode status;

        if (algo != nullptr) {
            config = get_config_from_algo(algo);
            if (config.tile_config == TileConfig::Undefined) {
                LOG(ERROR) << "config is invalid for mixed type algo";
                return ppl::common::RC_INVALID_VALUE;
            }

            status = can_implement(M, N, K, workspace_size, algo);
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "config is invalid for mixed type algo";
                return status;
            }
        } else {
            int multi_processor_count_ = props_->multiProcessorCount;
            const int split_k_limit = 8;

            std::vector<GemmConfig> candidate_configs = get_candidate_configs(sm_, M, false);
            std::vector<int> occupancies(candidate_configs.size());

            get_occupancies(candidate_configs, occupancies);

            static constexpr int num_experts = 1;
            config = estimate_best_config_from_occupancies(candidate_configs, occupancies, M, N, K, num_experts,
                                                           split_k_limit, workspace_size, multi_processor_count_,
                                                           kGroupSize);
        }

        if (config.tile_config != TileConfig::Undefined && config.tile_config != TileConfig::ChooseWithHeuristic) {
            return dispatch_gemm_shape(C, A, B, S, bias, workspace, M, N, K, config, st);
        }
        return ppl::common::RC_OTHER_ERROR;
    }
};

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16
