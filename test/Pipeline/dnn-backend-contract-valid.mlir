// RUN: dnn-mlir-opt -verify-dnn-backend-contract %s | FileCheck %s

func.func @mixed_dnn_backend_ir(%input: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2x4xf32>
  %filled = linalg.fill ins(%zero : f32)
      outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>
  %sum = arith.addf %input, %filled : tensor<2x4xf32>
  %result = dnn.relu %sum : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @mixed_dnn_backend_ir
// CHECK: linalg.fill
// CHECK: dnn.relu

