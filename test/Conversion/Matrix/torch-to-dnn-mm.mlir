// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_mm(
    %lhs: !torch.vtensor<[2,3],f32>,
    %rhs: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[2,4],f32> {
  // CHECK-LABEL: func.func @convert_mm
  // CHECK: torch_c.to_builtin_tensor
  // CHECK: torch_c.to_builtin_tensor
  // CHECK: dnn.mm
  // CHECK: torch_c.from_builtin_tensor
  // CHECK-NOT: torch.aten.mm
  %result = torch.aten.mm %lhs, %rhs
      : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,4],f32>
        -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

func.func @convert_pre_reduce_mm(
    %lhs: !torch.tensor<[2,3],f32>,
    %rhs: !torch.tensor<[3,4],f32>) -> !torch.tensor<[2,4],f32> {
  // CHECK-LABEL: func.func @convert_pre_reduce_mm
  // CHECK: torch.copy.to_vtensor
  // CHECK: torch_c.to_builtin_tensor
  // CHECK: dnn.mm
  // CHECK: torch_c.from_builtin_tensor
  // CHECK: torch.copy.to_tensor
  // CHECK-NOT: torch.aten.mm
  %result = torch.aten.mm %lhs, %rhs
      : !torch.tensor<[2,3],f32>, !torch.tensor<[3,4],f32>
        -> !torch.tensor<[2,4],f32>
  return %result : !torch.tensor<[2,4],f32>
}
