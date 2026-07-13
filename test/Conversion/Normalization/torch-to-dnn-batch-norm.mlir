// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @batch_norm(
    %input: !torch.vtensor<[1,64,56,56],f32>,
    %weight: !torch.vtensor<[64],f32>,
    %bias: !torch.vtensor<[64],f32>,
    %mean: !torch.vtensor<[64],f32>,
    %variance: !torch.vtensor<[64],f32>)
    -> !torch.vtensor<[1,64,56,56],f32> {
  %false = torch.constant.bool false
  %momentum = torch.constant.float 1.000000e-01
  %epsilon = torch.constant.float 1.000000e-05
  %result = torch.aten.batch_norm %input, %weight, %bias, %mean, %variance,
      %false, %momentum, %epsilon, %false
      : !torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[64],f32>,
        !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>,
        !torch.vtensor<[64],f32>, !torch.bool, !torch.float, !torch.float,
        !torch.bool -> !torch.vtensor<[1,64,56,56],f32>
  return %result : !torch.vtensor<[1,64,56,56],f32>
}

// CHECK-LABEL: func.func @batch_norm
// CHECK: dnn.batch_norm
// CHECK-SAME: kind = "aten.batch_norm"
// CHECK-SAME: parameter_indices = array<i32: 5, 6, 7, 8>
// CHECK-NOT: torch.aten.batch_norm
