#ifdef PPLNN_ENABLE_FP8

#ifndef __PPL_KERNEL_LLM_CUDA_PMX_F8F8_CAST_H__
#define __PPL_KERNEL_LLM_CUDA_PMX_F8F8_CAST_H__

#include "ppl/kernel/llm/cuda/common/general_include.h"
#include "ppl/kernel/llm/cuda/common/matrix_layout.h"

namespace ppl { namespace kernel { namespace llm { namespace cuda { namespace pmx { namespace f8f8 {

ppl::common::RetCode cast_fp16(
    cudaStream_t stream,
    const void* input, // fp16, [batch, quant_dim]
    const int64_t batch,
    const int64_t quant_dim,
    void* casted // fp8, [batch, quant_dim]
);

}}}}}}

#endif
#endif