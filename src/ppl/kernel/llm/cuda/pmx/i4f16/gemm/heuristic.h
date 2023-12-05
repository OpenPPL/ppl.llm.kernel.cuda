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

#include <vector>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

struct TileShape {
    int m;
    int n;
    int k;
};

struct ScoreComponent {
    TileShape shape;
    int wave_num;
    float waves_tail;
    float wave_fractional;
    float block_tail;
    bool is_splitk;
    bool is_no_splitk;
    int splitk_penalty;
    bool is_prefer_big_k;
    int m;

    bool operator<(const ScoreComponent& rhs) {
        float offset = m > 16 ? float(0.0) : float(0.3);
        float score_lhs = waves_tail + block_tail + offset * wave_num;
        float score_rhs = rhs.waves_tail + rhs.block_tail + offset * rhs.wave_num;

        if (is_no_splitk || rhs.is_no_splitk) {
            if (!is_splitk && (rhs.is_splitk)) {
                return true;
            } else if (is_splitk && !rhs.is_splitk) {
                return false;
            } else {
                return ((score_lhs + 0.2) < score_rhs) || ((score_lhs <= score_rhs && shape.k > rhs.shape.k));
            }
        }

        if (is_splitk && (!rhs.is_splitk)) {
            return true;
        } else if (!is_splitk && rhs.is_splitk) {
            return false;
        }

        if (shape.m == rhs.shape.m && shape.n == rhs.shape.n) {
            const float threshold = rhs.m <= 32 ? float(1.0) : float(0.8);
            const float offset = rhs.m <= 32 ? float(0.4) : float(0.2);
            if ((((wave_fractional <= threshold) && (rhs.wave_fractional <= threshold)) &&
                 (!is_prefer_big_k && !rhs.is_prefer_big_k)) ||
                (rhs.m >= 64)) {
                if (shape.k < rhs.shape.k) {
                    score_rhs += offset;
                } else if (shape.k > rhs.shape.k) {
                    score_lhs += offset;
                }
            } else {
                if (shape.k < rhs.shape.k) {
                    score_lhs += offset;
                } else if (shape.k > rhs.shape.k) {
                    score_rhs += offset;
                }
            }
        }
        return score_lhs <= score_rhs;
    }
};

std::vector<GemmConfig> get_candidate_configs(int sm, int m, const bool simt_configs_only);

bool is_config_valid(const TileShape& shape, int M, int N, int K, int workspace_size, const int splitk_factor,
                     const int group_size);

GemmConfig estimate_best_config_from_occupancies(const std::vector<GemmConfig>& candidate_configs,
                                                 const std::vector<int>& occupancies, const int64_t m, const int64_t n,
                                                 const int64_t k, const int64_t num_experts, const int split_k_limit,
                                                 const size_t workspace_bytes, const int multi_processor_count,
                                                 const int group_size);

TileShape get_cta_shape_for_config(TileConfig tile_config);

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16