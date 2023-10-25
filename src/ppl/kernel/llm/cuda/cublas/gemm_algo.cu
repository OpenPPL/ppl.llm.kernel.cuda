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

#include "ppl/kernel/llm/cuda/cublas/gemm_algo.h"

#include "ppl/common/log.h"

#include <algorithm>
#include <map>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace cublas {

#define CUBLAS_CHECK_RC(X) do { \
        cublasStatus_t __status = (X); \
        if (__status != CUBLAS_STATUS_SUCCESS) { \
            LOG(ERROR) << "cublasLt failed: " << cublasLtGetStatusString(__status); \
            return {ppl::common::RC_DEVICE_RUNTIME_ERROR, cublasLtMatmulAlgo_t{}}; \
        } \
    } while (0)

std::pair<ppl::common::RetCode, cublasLtMatmulAlgo_t> find_best_algo(
    const cudaStream_t     stream,
    const cublasLtHandle_t&lightHandle,
    const std::vector<int>&banned_algo_ids,
    cublasLtMatmulDesc_t   computeDesc,
    const void*            alpha,
    const void*            A,
    cublasLtMatrixLayout_t Adesc,
    const void*            B,
    cublasLtMatrixLayout_t Bdesc,
    const void*            beta,
    const void*            C,
    cublasLtMatrixLayout_t Cdesc,
    void*                  D,
    cublasLtMatrixLayout_t Ddesc,
    const int64_t          workspace_size,
    void*                  workspace)
{
    size_t returnSize;
    int32_t pointer_mode;
    cublasLtMatmulDescGetAttribute(
        computeDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode), &returnSize);

    std::vector<cublasLtMatmulHeuristicResult_t> heuristics(200);
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK_RC(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK_RC(cublasLtMatmulPreferenceInit(preference));
    CUBLAS_CHECK_RC(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
#if (CUBLAS_VERSION) <= 12000
    uint32_t pointer_mode_mask = 0;
    CUBLAS_CHECK_RC(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK, &pointer_mode_mask, sizeof(pointer_mode_mask)));
#endif

    int return_count = 0;
    auto ret = cublasLtMatmulAlgoGetHeuristic(lightHandle,
                                              computeDesc,
                                              Adesc,
                                              Bdesc,
                                              Cdesc,
                                              Ddesc,
                                              preference,
                                              heuristics.size(),
                                              heuristics.data(),
                                              &return_count);
    heuristics.resize(return_count);

    std::map<int, std::vector<float>> algo_results;
    for (const auto& heuristic : heuristics) {
        cublasLtMatmulAlgo_t algo = heuristic.algo;
        int32_t algo_id;
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &returnSize);

        if (std::find(banned_algo_ids.begin(), banned_algo_ids.end(), algo_id) != banned_algo_ids.end())
            continue;

        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);

        for (int i = 0; i < 11; i++) {
            float duration_ms;
            cudaEventRecord(start_event, stream);
            CUBLAS_CHECK_RC(cublasLtMatmul(lightHandle,
                                            computeDesc,
                                            alpha,
                                            A,
                                            Adesc,
                                            B,
                                            Bdesc,
                                            beta,
                                            C,
                                            Cdesc,
                                            D,
                                            Ddesc,
                                            &algo,
                                            workspace,
                                            workspace_size,
                                            stream));
            cudaEventRecord(stop_event, stream);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&duration_ms, start_event, stop_event);

            algo_results[algo_id].push_back(duration_ms);
        }
        std::sort(algo_results[algo_id].begin(), algo_results[algo_id].end());
    }

    cublasLtMatmulHeuristicResult_t result;
    float best_time = INFINITY;
    for (const auto& heuristic : heuristics) {
        cublasLtMatmulAlgo_t algo = heuristic.algo;
        int32_t algo_id;
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &returnSize);
        const auto& results = algo_results[algo_id];

        if (results.size() > 0 && results[5] < best_time) {
            best_time = results[5];
            result = heuristic;
        }
    }

    return {best_time != INFINITY ? ppl::common::RC_SUCCESS : ppl::common::RC_NOT_FOUND, result.algo};
}

}}}}}
