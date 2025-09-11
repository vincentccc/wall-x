#include "dual_asym_grouped_gemm.h"
#include "permute.h"
#include "rope.h"

#include <torch/extension.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("asym_dual_gmm", &AsymmetricDualExpertGemm, "Asymmetric Dual Expert Grouped GEMM.");
  m.def("permute", &moe_permute_topK_op, "Token permutation kernel");
  m.def("unpermute", &moe_recover_topK_op, "Token un-permutation kernel");
  m.def("unpermute_bwd", &moe_recover_topK_bwd_op, "Token un-permutation backward kernel");
  m.def("rope", &launch_multimodal_rope_forward, "Multimodal RoPE forward kernel");
  m.def("rope_bwd", &launch_multimodal_rope_backward, "Multimodal RoPE backward kernel");
}
