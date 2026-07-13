// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @layer_norm(
    %input: !torch.vtensor<[1,197,768],f32>,
    %weight: !torch.vtensor<[768],f32>,
    %bias: !torch.vtensor<[768],f32>)
    -> !torch.vtensor<[1,197,768],f32> {
  %dim = torch.constant.int 768
  %shape = torch.prim.ListConstruct %dim : (!torch.int) -> !torch.list<int>
  %epsilon = torch.constant.float 1.000000e-06
  %false = torch.constant.bool false
  %result = torch.aten.layer_norm %input, %shape, %weight, %bias, %epsilon,
      %false
      : !torch.vtensor<[1,197,768],f32>, !torch.list<int>,
        !torch.vtensor<[768],f32>, !torch.vtensor<[768],f32>, !torch.float,
        !torch.bool -> !torch.vtensor<[1,197,768],f32>
  return %result : !torch.vtensor<[1,197,768],f32>
}

// CHECK-LABEL: func.func @layer_norm
// CHECK: dnn.layer_norm
// CHECK-SAME: parameter_indices = array<i32: 1, 4, 5>
// CHECK-SAME: parameters =
// CHECK-SAME: 768
// CHECK-SAME: false
// CHECK-NOT: torch.aten.layer_norm
