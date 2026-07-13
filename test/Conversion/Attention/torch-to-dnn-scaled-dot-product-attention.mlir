// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @scaled_dot_product_attention(
    %query: !torch.vtensor<[1,12,197,64],f32>,
    %key: !torch.vtensor<[1,12,197,64],f32>,
    %value: !torch.vtensor<[1,12,197,64],f32>)
    -> !torch.vtensor<[1,12,197,64],f32> {
  %none = torch.constant.none
  %dropout = torch.constant.float 0.000000e+00
  %false = torch.constant.bool false
  %result = torch.aten.scaled_dot_product_attention %query, %key, %value,
      %none, %dropout, %false, %none, %false
      : !torch.vtensor<[1,12,197,64],f32>,
        !torch.vtensor<[1,12,197,64],f32>,
        !torch.vtensor<[1,12,197,64],f32>, !torch.none, !torch.float,
        !torch.bool, !torch.none, !torch.bool
        -> !torch.vtensor<[1,12,197,64],f32>
  return %result : !torch.vtensor<[1,12,197,64],f32>
}

// CHECK-LABEL: func.func @scaled_dot_product_attention
// CHECK: dnn.scaled_dot_product_attention
// CHECK-SAME: parameter_indices = array<i32: 3, 4, 5, 6, 7>
// CHECK-SAME: parameters = [unit, 0.000000e+00, false, unit, false]
// CHECK-NOT: torch.aten.scaled_dot_product_attention
