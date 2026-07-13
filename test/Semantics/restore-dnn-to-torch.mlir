// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | dnn-mlir-opt -test-restore-dnn-to-torch -canonicalize | FileCheck %s

func.func @mm_relu(%lhs: !torch.vtensor<[2,3],f32>,
                   %rhs: !torch.vtensor<[3,4],f32>)
    -> !torch.vtensor<[2,4],f32> {
  %mm = torch.aten.mm %lhs, %rhs
      : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,4],f32>
        -> !torch.vtensor<[2,4],f32>
  %result = torch.aten.relu %mm
      : !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @mm_relu
// CHECK: torch.aten.mm
// CHECK-NEXT: torch.aten.relu
// CHECK-NOT: dnn.
// CHECK-NOT: torch_c.

