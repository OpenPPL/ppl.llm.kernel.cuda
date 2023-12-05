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

#include "heuristic.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

bool is_config_valid(const TileShape& tile_shape, int M, int N, int K, int workspace_size, const int splitk_factor,
                     const int group_size) {
    const size_t workspace_size_min = splitk_factor * M * N * 4 * sizeof(half);
    if (splitk_factor > 1 && (size_t)workspace_size < workspace_size_min) {
        return false;
    }

    const uint32_t block_k = tile_shape.k;
    const int slice_batch = (K / splitk_factor + block_k - 1) / block_k * block_k;
    const int slice_tail = K % slice_batch;
    if (slice_tail % block_k != 0) {
        return false;
    }

    return true;
}

TileShape get_cta_shape_for_config(TileConfig tile_config) {
    switch (tile_config) {
        case TileConfig::CtaShape16x64x64_WarpShape16x16x64:
            return TileShape{16, 64, 64};
        case TileConfig::CtaShape16x16x512_WarpShape16x16x128:
            return TileShape{16, 16, 512};
        case TileConfig::CtaShape16x32x256_WarpShape16x16x128:
            return TileShape{16, 32, 256};
        case TileConfig::CtaShape16x64x256_WarpShape16x16x128:
            return TileShape{16, 64, 256};
        case TileConfig::CtaShape16x64x128_WarpShape16x16x128:
            return TileShape{16, 64, 128};

        case TileConfig::CtaShape32x64x64_WarpShape32x16x64:
            return TileShape{32, 64, 64};
        case TileConfig::CtaShape32x64x128_WarpShape32x16x128:
            return TileShape{32, 64, 128};
        case TileConfig::CtaShape32x64x256_WarpShape32x16x128:
            return TileShape{32, 64, 256};
        case TileConfig::CtaShape32x32x128_WarpShape32x16x128:
            return TileShape{32, 32, 128};
        case TileConfig::CtaShape32x32x256_WarpShape32x16x128:
            return TileShape{32, 32, 256};

        case TileConfig::CtaShape64x32x128_WarpShape32x16x128:
            return TileShape{64, 32, 128};
        case TileConfig::CtaShape64x32x64_WarpShape32x16x64:
            return TileShape{64, 32, 64};
        default:
            return TileShape{0, 0, 0};
    }
}

std::vector<TileConfig> get_candidate_tiles(int m, const bool simt_configs_only) {
    std::vector<TileConfig> simt_configs{TileConfig::CtaShape16x64x64_WarpShape16x16x64};

    if (m >= 1 && m <= 16) {
        std::vector<TileConfig> quant_B_configs{
            TileConfig::CtaShape16x64x64_WarpShape16x16x64, TileConfig::CtaShape16x16x512_WarpShape16x16x128,
            TileConfig::CtaShape16x32x256_WarpShape16x16x128, TileConfig::CtaShape16x64x256_WarpShape16x16x128,
            TileConfig::CtaShape16x64x128_WarpShape16x16x128};
        return quant_B_configs;
    } else if (m > 16 && m <= 32) {
        std::vector<TileConfig> quant_B_configs{
            TileConfig::CtaShape32x64x64_WarpShape32x16x64, TileConfig::CtaShape32x64x128_WarpShape32x16x128,
            TileConfig::CtaShape32x64x256_WarpShape32x16x128
            // Not use here:
            // TileConfig::CtaShape32x32x128_WarpShape32x16x128,
            // TileConfig::CtaShape32x32x256_WarpShape32x16x128,
        };
        return quant_B_configs;
    } else {
        std::vector<TileConfig> quant_B_configs{
            TileConfig::CtaShape32x64x64_WarpShape32x16x64, TileConfig::CtaShape32x64x128_WarpShape32x16x128,
            TileConfig::CtaShape32x64x256_WarpShape32x16x128,
            // Not use here:
            // TileConfig::CtaShape32x32x128_WarpShape32x16x128,
            // TileConfig::CtaShape32x32x256_WarpShape32x16x128,
            TileConfig::CtaShape64x32x64_WarpShape32x16x64, TileConfig::CtaShape64x32x128_WarpShape32x16x128};
        return quant_B_configs;
    }
}

std::vector<GemmConfig> get_candidate_configs(int sm, int m, const bool simt_configs_only) {
    std::vector<TileConfig> tiles = get_candidate_tiles(m, simt_configs_only);
    std::vector<GemmConfig> candidate_configs;
    const int min_stages = 3;
    const int max_stages = sm >= 80 ? 3 : 2;

    for (const auto& tile_config : tiles) {
        for (int stages = min_stages; stages <= max_stages; ++stages) {
            GemmConfig config{tile_config, SplitKStyle::NO_SPLIT_K, 1, stages};
            candidate_configs.push_back(config);
        }
    }

    return candidate_configs;
}

bool is_valid_split_k_factor(const int64_t m, const int64_t n, const int64_t k, const TileShape tile_shape,
                             const int split_k_factor, const size_t workspace_bytes, const int group_size) {
    if (tile_shape.k >= 256 && split_k_factor > 1)
        return false;

    return is_config_valid(tile_shape, m, n, k, workspace_bytes, split_k_factor, group_size);
}

GemmConfig estimate_best_config_from_occupancies(const std::vector<GemmConfig>& candidate_configs,
                                                 const std::vector<int>& occupancies, const int64_t m, const int64_t n,
                                                 const int64_t k, const int64_t num_experts, const int split_k_limit,
                                                 const size_t workspace_bytes, const int multi_processor_count,
                                                 const int group_size) {
    GemmConfig best_config;
    ScoreComponent best_score{0};
    float best_score_f = std::numeric_limits<float>::max();

    int current_m_tile = 0;

    const int max_split_k = n >= multi_processor_count * 256 ? 1 : split_k_limit;
    for (int ii = 0; ii < (int)candidate_configs.size(); ++ii) {
        GemmConfig candidate_config = candidate_configs[ii];
        TileShape tile_shape = get_cta_shape_for_config(candidate_config.tile_config);
        if (tile_shape.m == 0)
            continue;

        int occupancy = occupancies[ii];

        if (occupancy == 0) {
            continue;
        }

        if (best_config.tile_config != TileConfig::ChooseWithHeuristic && m < current_m_tile &&
            current_m_tile < tile_shape.m) {
            continue;
        }

        const int ctas_in_m_dim = (m + tile_shape.m - 1) / tile_shape.m;
        const int ctas_in_n_dim = (n + tile_shape.n - 1) / tile_shape.n;

        if (multi_processor_count <= 20 && n >= multi_processor_count * 32) {
            if (is_config_valid(tile_shape, m, n, k, workspace_bytes, 1, group_size)) {
                const int ctas_per_wave = occupancy * multi_processor_count;
                const int ctas_for_problem = ctas_in_m_dim * ctas_in_n_dim;

                const int num_waves_total = (ctas_for_problem + ctas_per_wave - 1) / ctas_per_wave;
                const float num_waves_fractional = ctas_for_problem / float(ctas_per_wave);
                const float current_score = float(num_waves_total) - num_waves_fractional;
                const int ctas_tails = ctas_for_problem - (num_waves_total - 1) * ctas_per_wave;
                // float block_tail = 0;
                if (ctas_tails % multi_processor_count != 0) {
                    float block_tail = float(1.0) - ctas_tails % multi_processor_count / float(multi_processor_count);
                }

                float extra_cost = 0;
                if (current_score > 0.6) {
                    extra_cost = 0.6;
                }
                float assume_score = float(int(num_waves_fractional)) * 0.6 + (current_score) + extra_cost;

                if (best_config.tile_config == TileConfig::ChooseWithHeuristic || (!(best_score_f < assume_score))) {
                    best_score_f = assume_score;
                    best_config =
                        GemmConfig{candidate_config.tile_config, SplitKStyle::NO_SPLIT_K, 1, candidate_config.stages};
                }
            }
        } else {
            for (int split_k_factor = 1; split_k_factor <= max_split_k; ++split_k_factor) {
                if (is_valid_split_k_factor(m, n, k, tile_shape, split_k_factor, workspace_bytes, group_size)) {
                    const int ctas_per_wave = occupancy * multi_processor_count;
                    const int ctas_for_problem = ctas_in_m_dim * ctas_in_n_dim * split_k_factor;

                    const int num_waves_total = (ctas_for_problem + ctas_per_wave - 1) / ctas_per_wave;
                    const float num_waves_fractional = ctas_for_problem / float(ctas_per_wave);
                    const float current_score = float(num_waves_total) - num_waves_fractional;

                    const int ctas_tails = ctas_for_problem - (num_waves_total - 1) * ctas_per_wave;
                    float block_tail_low =
                        float(1.0) - ctas_tails % multi_processor_count / float(multi_processor_count);
                    const bool is_cta_almost_full = (block_tail_low < float(0.5) || current_score < 0.5) &&
                        ((current_score + block_tail_low) <= 1.0);
                    const bool is_not_cta_spill = (ctas_for_problem / multi_processor_count) < float(1.0);
                    const bool is_k_not_big = (k / n <= 4);

                    const bool is_no_split_sure = ((n <= 256 && k <= 4096) || (k <= 2048));

                    bool is_prefer_no_split = (m < 16) || ((float(k) / float(n)) < float(1.5));
                    const bool is_no_split = (is_cta_almost_full && is_not_cta_spill && is_k_not_big &&
                                              (split_k_factor == 1) && is_prefer_no_split) ||
                        is_no_split_sure;

                    float block_tail = ctas_tails % multi_processor_count / float(multi_processor_count);

                    ScoreComponent current_score_component;
                    current_score_component.shape = tile_shape;
                    current_score_component.wave_num = num_waves_total;
                    current_score_component.waves_tail = current_score;
                    current_score_component.block_tail = (float(1.0) - block_tail);
                    current_score_component.is_splitk = split_k_factor > 1 ? 1 : 0;
                    current_score_component.is_no_splitk = is_no_split;
                    current_score_component.wave_fractional = num_waves_fractional;

                    current_score_component.is_prefer_big_k = (k / n > 8) ? true : false;
                    current_score_component.m = m;

                    if (best_config.tile_config == TileConfig::ChooseWithHeuristic || (!(best_score < current_score_component))) {
                        SplitKStyle split_style = split_k_factor > 1 ? SplitKStyle::SPLIT_K : SplitKStyle::NO_SPLIT_K;

                        best_score = current_score_component;
                        best_config = GemmConfig{candidate_config.tile_config, split_style, split_k_factor,
                                                 candidate_config.stages};
                    }
                }
            }
        }
    }

    return best_config;
}

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16