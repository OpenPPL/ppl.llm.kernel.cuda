#pragma once

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace i4f16 {
    
class Gemv {
public:
    void run(const void* __restrict__ A, const void* __restrict__ B, const void* __restrict__ S, void* __restrict__ C,
             int M, int N, int K, cudaStream_t stream) const;

    void run_bias(const void* __restrict__ A, const void* __restrict__ B, const void* __restrict__ S,
                  const void* __restrict__ BS, void* __restrict__ C, int M, int N, int K, cudaStream_t stream) const;
};

}}}}}} // namespace ppl::kernel::llm::cuda::pmx::i4f16