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

#include "algorithm.h"

#include "ppl/kernel/llm/cuda/common/general_include.h"

#include <memory>

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {

class GemmAPI {
public:
    GemmAPI();

    ~GemmAPI();

    ppl::common::RetCode Execute(void* C, const void* A, const void* B, const void* S, const void* bias, void* workspace,
                          int m, int n, int k, int group_size, int workspace_size, GemmAlgorithm_t* algo = nullptr,
                          cudaStream_t st = nullptr) const;
private:
    struct GemmImpl;
    std::unique_ptr<GemmImpl> impl_;
};

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16