#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_mla_<cutlass::half_t, 192, 0, 0>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_mla_hdim192<cutlass::half_t, 0, 0>(params, stream);
}
