// RUN: not dnn-mlir-opt -verify-dnn-backend-contract %s 2>&1 | FileCheck %s

func.func @residual_torch_signature(%input: !torch.vtensor<[2,4],f32>)
    -> !torch.vtensor<[2,4],f32> {
  return %input : !torch.vtensor<[2,4],f32>
}

// CHECK: error: 'func.func' op is illegal in the DNN backend contract: attribute 'function_type' contains a residual Torch type

