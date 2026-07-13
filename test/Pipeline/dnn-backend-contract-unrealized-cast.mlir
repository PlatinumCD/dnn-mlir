// RUN: not dnn-mlir-opt -verify-dnn-backend-contract %s 2>&1 | FileCheck %s

func.func @unresolved_conversion(%input: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  %result = builtin.unrealized_conversion_cast %input
      : tensor<2x4xf32> to tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// CHECK: error: 'builtin.unrealized_conversion_cast' op is illegal in the DNN backend contract: unresolved conversion cast

