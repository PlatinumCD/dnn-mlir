// RUN: not dnn-mlir-opt -verify-dnn-backend-contract %s 2>&1 | FileCheck %s

func.func @residual_torch_operation() {
  %value = torch.constant.int 1
  return
}

// CHECK: error: 'torch.constant.int' op is illegal in the DNN backend contract: residual 'torch' operation

