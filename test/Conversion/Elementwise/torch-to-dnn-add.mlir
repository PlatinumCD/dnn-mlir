// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @add(%lhs: !torch.vtensor<[1,64,56,56],f32>,
               %rhs: !torch.vtensor<[1,64,56,56],f32>)
    -> !torch.vtensor<[1,64,56,56],f32> {
  %alpha = torch.constant.int 1
  %result = torch.aten.add.Tensor %lhs, %rhs, %alpha
      : !torch.vtensor<[1,64,56,56],f32>,
        !torch.vtensor<[1,64,56,56],f32>, !torch.int
        -> !torch.vtensor<[1,64,56,56],f32>
  return %result : !torch.vtensor<[1,64,56,56],f32>
}

func.func @add_inplace(%lhs: !torch.tensor<[1,64,56,56],f32>,
                       %rhs: !torch.tensor<[1,64,56,56],f32>)
    -> !torch.tensor<[1,64,56,56],f32> {
  %alpha = torch.constant.int 1
  %result = torch.aten.add_.Tensor %lhs, %rhs, %alpha
      : !torch.tensor<[1,64,56,56],f32>,
        !torch.tensor<[1,64,56,56],f32>, !torch.int
        -> !torch.tensor<[1,64,56,56],f32>
  return %result : !torch.tensor<[1,64,56,56],f32>
}

// CHECK-LABEL: func.func @add
// CHECK: dnn.add
// CHECK-SAME: parameter_indices = array<i32: 2>
// CHECK-NOT: torch.aten.add.Tensor

// CHECK-LABEL: func.func @add_inplace
// CHECK: torch.aten.add_.Tensor
// CHECK-NOT: dnn.add
