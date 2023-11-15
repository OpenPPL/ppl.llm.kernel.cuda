#ifndef __PPL_KERNEL_LLM_CUDA_PMX_GELU_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_GELU_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx {

ppl::common::RetCode gelu(
    cudaStream_t stream,
    const ppl::common::TensorShape* input_shape,
    const void* input,
    const void* optional_gate,
    bool approximate,
    void* output);

}}}}}

#endif