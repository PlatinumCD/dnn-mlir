// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_linear(
    %input: !torch.vtensor<[2,3],f32>,
    %weight: !torch.vtensor<[4,3],f32>,
    %bias: !torch.vtensor<[4],f32>) -> !torch.vtensor<[2,4],f32> {
  // CHECK-LABEL: func.func @convert_linear
  // CHECK: dnn.linear
  // CHECK-NOT: torch.aten.linear
  %result = torch.aten.linear %input, %weight, %bias
      : !torch.vtensor<[2,3],f32>, !torch.vtensor<[4,3],f32>,
        !torch.vtensor<[4],f32> -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

func.func @convert_linear_without_bias(
    %input: !torch.vtensor<[2,3],f32>,
    %weight: !torch.vtensor<[4,3],f32>) -> !torch.vtensor<[2,4],f32> {
  // CHECK-LABEL: func.func @convert_linear_without_bias
  // CHECK: dnn.linear
  // CHECK-NOT: torch.aten.linear
  %none = torch.constant.none
  %result = torch.aten.linear %input, %weight, %none
      : !torch.vtensor<[2,3],f32>, !torch.vtensor<[4,3],f32>, !torch.none
        -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}
