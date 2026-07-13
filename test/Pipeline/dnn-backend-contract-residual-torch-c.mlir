// RUN: not dnn-mlir-opt -verify-dnn-backend-contract %s 2>&1 | FileCheck %s

func.func @residual_torch_conversion(%input: tensor<2x4xf32>) {
  %value = torch_c.from_builtin_tensor %input
      : tensor<2x4xf32> -> !torch.vtensor<[2,4],f32>
  return
}

// CHECK: error: 'torch_c.from_builtin_tensor' op is illegal in the DNN backend contract: residual 'torch_c' operation

