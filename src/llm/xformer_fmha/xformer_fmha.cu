
#include "xformer_fmha.h"
#include "cudakernel/common/cuda_check.h"
#include <cuda_fp16.h>
#include <cmath>

ppl::common::RetCode PPLCUDAFMHAForwardImp(
    const cudaDeviceProp& device_prop,
    const cudaStream_t stream,
    const ppl::common::TensorShape* query_shape,
    const void* query,
    const ppl::common::TensorShape* key_shape,
    const void* key,
    const ppl::common::TensorShape* value_shape,
    const void* value,
    const ppl::common::TensorShape* attn_mask_shape,
    const void* attn_mask,
    const ppl::common::TensorShape* seqstart_q_shape,
    const void* seqstart_q,
    const ppl::common::TensorShape* seqstart_k_shape,
    const void* seqstart_k,
    const ppl::common::TensorShape* seqlen_k_shape,
    const void* seqlen_k,
    const int64_t max_seqlen,
    const int64_t custom_mask_type,
    const double scale,
    const ppl::common::TensorShape* output_shape,
    void* output)
{
    int64_t max_seqlen_q, max_seqlen_k;
    if (seqstart_q != nullptr) {
        max_seqlen_q = max_seqlen;
        max_seqlen_k = 0; // Will be set inside the kernel
    } else {
        max_seqlen_q = query_shape->GetDim(1);
        max_seqlen_k = key_shape->GetDim(1);
    }

    int64_t B = query_shape->GetDim(0);
    int64_t M = query_shape->GetDim(1);
    int64_t N = key_shape->GetDim(1);
    int64_t num_heads = query_shape->GetDim(2);
    int64_t K = query_shape->GetDim(3);
    int64_t Kv = value_shape->GetDim(3);

    const double dropout_p = 0;
    const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;
    if (use_dropout) {
        // to do softmax's dropout
    }
    const int computeCapability = device_prop.major * 10 + device_prop.minor;
    bool kernel_launched = false;
    const auto maxShmem = device_prop.sharedMemPerBlockOptin;

    // launchKernel lambda func
    auto launchKernel = [&](auto _k, auto kernel_fn) { // _k is struct AttentionKernel in kernel_forward.h
        using Kernel = decltype(_k);
        using scalar_t = typename Kernel::scalar_t;
        (void)_k;

        if (kernel_launched) {
            return;
        }

        // Check if this kernel is compatible
        if (!Kernel::kSupportsDropout && use_dropout) {
            return;
        }
        if (!Kernel::kSupportsBias && (attn_mask != nullptr)) {
            return;
        }
        if (Kernel::kSingleValueIteration &&
            Kernel::kKeysPerBlock < value_shape->GetDim(3)) {
            return;
        }

        // Uses too much shmem
        size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
        if (smem_bytes > maxShmem) {
            return;
        }
        kernel_launched = true;

        typename Kernel::Params p;
        p.query_ptr = (scalar_t*)query;
        p.key_ptr = (scalar_t*)key;
        p.value_ptr = (scalar_t*)value;

        p.logsumexp_ptr = nullptr;
        p.output_accum_ptr = nullptr;

        p.output_ptr = (typename Kernel::output_t*)output;

        if (seqstart_q != nullptr) {
            p.seqstart_q_ptr = (int64_t*)seqstart_q;
            p.seqstart_k_ptr = (int64_t*)seqstart_k;
        }

        p.num_heads = num_heads;
        p.head_dim = query_shape->GetDim(3);
        p.head_dim_value = value_shape->GetDim(3);
        p.num_queries = max_seqlen_q;
        p.num_keys = max_seqlen_k;
        p.num_batches = seqstart_q != nullptr ? seqstart_q_shape->GetDim(0) - 1 : B;
        p.custom_mask_type = custom_mask_type;

        p.seqlen_k_ptr = nullptr;
        if (seqlen_k != nullptr) {
            // CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(seqlen_k.value());
            PPL_CHECK(seqlen_k_shape->GetDataType() == ppl::common::DATATYPE_INT32, "seqlen_k must be int32");
            p.seqlen_k_ptr = (int32_t*)seqlen_k;
        }

        if (scale != 0) {
            p.scale = float(scale);
        } else {
            p.scale = float(1.0 / std::sqrt(float(p.head_dim)));
        }

        p.q_strideB = query_shape->GetDim(1) * query_shape->GetDim(2) * query_shape->GetDim(3);
        p.k_strideB = key_shape->GetDim(1) * key_shape->GetDim(2) * key_shape->GetDim(3);
        p.v_strideB = value_shape->GetDim(1) * value_shape->GetDim(2) * value_shape->GetDim(3);
        p.q_strideM = query_shape->GetDim(2) * query_shape->GetDim(3);
        p.k_strideM = key_shape->GetDim(2) * key_shape->GetDim(3);
        p.v_strideM = value_shape->GetDim(2) * value_shape->GetDim(3);
        p.q_strideH = query_shape->GetDim(3);
        p.k_strideH = key_shape->GetDim(3);
        p.v_strideH = value_shape->GetDim(3);
        p.o_strideM = output_shape->GetDim(2) * output_shape->GetDim(3);

        if (attn_mask != nullptr) {
            //CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
            PPL_CHECK(
                attn_mask_shape->GetDataType() == query_shape->GetDataType(),
                "invalid dtype for bias - should match query's dtype");
            p.attn_bias_ptr = (scalar_t*)attn_mask;
            p.bias_strideB = attn_mask_shape->GetDim(1) * attn_mask_shape->GetDim(2) * attn_mask_shape->GetDim(3);
            p.bias_strideH = attn_mask_shape->GetDim(2) * attn_mask_shape->GetDim(3);
            p.bias_strideM = attn_mask_shape->GetDim(3);

        }

        p.use_dropout = false;

        if (smem_bytes > 0xc000) {
            auto err = cudaFuncSetAttribute(
                kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
            PPL_CHECK(
                err != cudaErrorInvalidValue,
                "This GPU does not have enough shared-memory (kernel requires ");
        }
        Kernel::check_supported(p);
        kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
  };

  DISPATCH_TYPES(query_shape, ([&]() { dispatch_cutlassF<scalar_t>(launchKernel, computeCapability); }));
  PPL_CHECK(kernel_launched, "cutlassF: no kernel found to launch!");
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      LOG(ERROR) << "CUDA Error: " << cudaGetErrorString(err);
      return ppl::common::RC_DEVICE_RUNTIME_ERROR;
  }

  return ppl::common::RC_SUCCESS;
}