# humble-int8-head — status: INCOMPLETE, does not build

This branch applies JP-devv/humble-b70-llm's `0001-int8-w8a8-lmhead-backport.patch` to
CySpiegel/vllm-xpu-kernels `main`. That published patch is a fragment. It references code that is
not in the patch, in upstream, or anywhere in humble-b70-llm (single branch, squashed history):

- `csrc/xpu/onednn/int8_gemm_w8a8.h` — `#include`d by onednn_matmul.cpp; must define
  `oneDNN::dnnl_matmul_w8a8_int8(result, A, A_scale, B, B_scale, is_nt, bias)` (s8 x s8 -> f16/bf16
  oneDNN matmul with per-token A scales and per-channel B scales; model it on
  `fp8_gemm_w8a8.h`). The patch only adds the `s8_s8_f16`/`s8_s8_bf16` dtype mappers.
- `csrc/xpu/quantization/int8_quant.cpp` — dropped from CMake in a later commit (per-token quant is
  done with torch ops on the vLLM side instead).

Until `int8_gemm_w8a8.h` is written this branch fails to compile.
