// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @embedding(
    %weight: !torch.vtensor<[30522,768],f32>,
    %indices: !torch.vtensor<[1,128],si64>)
    -> !torch.vtensor<[1,128,768],f32> {
  %padding_idx = torch.constant.int 0
  %false = torch.constant.bool false
  %result = torch.aten.embedding %weight, %indices, %padding_idx, %false,
      %false
      : !torch.vtensor<[30522,768],f32>, !torch.vtensor<[1,128],si64>,
        !torch.int, !torch.bool, !torch.bool
        -> !torch.vtensor<[1,128,768],f32>
  return %result : !torch.vtensor<[1,128,768],f32>
}

// CHECK-LABEL: func.func @embedding
// CHECK: dnn.embedding
// CHECK-SAME: parameter_indices = array<i32: 2, 3, 4>
// CHECK-SAME: parameters = [0, false, false]
// CHECK-NOT: torch.aten.embedding
