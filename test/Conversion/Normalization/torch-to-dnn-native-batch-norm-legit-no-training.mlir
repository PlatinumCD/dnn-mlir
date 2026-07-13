// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @native_batch_norm_legit_no_training(
    %input: !torch.vtensor<[1,64,56,56],f32>,
    %weight: !torch.vtensor<[64],f32>,
    %bias: !torch.vtensor<[64],f32>,
    %mean: !torch.vtensor<[64],f32>,
    %variance: !torch.vtensor<[64],f32>)
    -> (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[0],f32>,
        !torch.vtensor<[0],f32>) {
  %momentum = torch.constant.float 1.000000e-01
  %epsilon = torch.constant.float 1.000000e-05
  %result:3 = torch.operator
      "torch.aten._native_batch_norm_legit_no_training"(
      %input, %weight, %bias, %mean, %variance, %momentum, %epsilon)
      : (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[64],f32>,
         !torch.vtensor<[64],f32>, !torch.vtensor<[64],f32>,
         !torch.vtensor<[64],f32>, !torch.float, !torch.float)
        -> (!torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[0],f32>,
            !torch.vtensor<[0],f32>)
  return %result#0, %result#1, %result#2
      : !torch.vtensor<[1,64,56,56],f32>, !torch.vtensor<[0],f32>,
        !torch.vtensor<[0],f32>
}

// CHECK-LABEL: func.func @native_batch_norm_legit_no_training
// CHECK: dnn.batch_norm
// CHECK-SAME: kind = "aten._native_batch_norm_legit_no_training"
// CHECK-SAME: parameter_indices = array<i32: 5, 6>
// CHECK-NOT: torch.operator
