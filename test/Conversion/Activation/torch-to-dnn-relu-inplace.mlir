// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_reluinplace(%input: !torch.tensor<[2,4],f32>) -> !torch.tensor<[2,4],f32> {
  %result = torch.aten.relu_ %input
        : !torch.tensor<[2,4],f32> -> !torch.tensor<[2,4],f32>
  return %result : !torch.tensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_reluinplace
// CHECK: torch.aten.relu_
// CHECK-NOT: dnn.
