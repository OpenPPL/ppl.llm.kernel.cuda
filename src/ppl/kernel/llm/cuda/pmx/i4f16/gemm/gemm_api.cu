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

#include "gemm_api.h"
#include "gemm_runner.h"
#include "gemv.h"

#include "ppl/common/log.h"

#include <cuda_runtime.h>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

struct GemmAPI::GemmImpl {
    ppl::common::RetCode run(void* C, const void* A, const void* B, const void* S, const void* bias, void* workspace, int m,
                   int n, int k, int group_size, size_t workspace_size, GemmAlgorithm_t* algo, cudaStream_t stream) {
        switch (group_size) {
            case 128: {
                if(m == 1 || m == 2) {
                    if (bias != nullptr) {
                        gemv.run_bias(A, B, S, bias,C, m, n / 4, k, stream);
                    } else {
                        gemv.run(A, B, S, C, m, n / 4, k, stream);
                    }
                } else {
                    if (bias != nullptr) {
                        gemm_bias.gemm(C, A, B, S, bias, workspace, m, n / 4, k, workspace_size, algo, stream);
                    } else {
                        gemm.gemm(C, A, B, S, bias, workspace, m, n / 4, k, workspace_size, algo, stream);
                    }
                }
                break;
            }
            default:
                LOG(ERROR) << "group size not support!";
                return ppl::common::RC_UNSUPPORTED;
                break;
        }
        return ppl::common::RC_SUCCESS;
    }

    GemmRunnerImpl<128, false> gemm;
    GemmRunnerImpl<128, true> gemm_bias;
    Gemv gemv;
};

GemmAPI::GemmAPI() : impl_(std::make_unique<GemmImpl>()) {}

GemmAPI::~GemmAPI() = default;

ppl::common::RetCode GemmAPI::Execute(void* C, const void* A, const void* B, const void* S, const void* bias, void* workspace,
                               int m, int n, int k, int group_size, size_t workspace_size, GemmAlgorithm_t* algo,
                               cudaStream_t stream) const {
    return impl_->run(C, A, B, S, bias, workspace, m, n, k, group_size, workspace_size, algo, stream);
}

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16