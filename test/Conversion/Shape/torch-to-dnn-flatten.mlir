// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @flatten(%input: !torch.vtensor<[1,512,1,1],f32>)
    -> !torch.vtensor<[1,512],f32> {
  %start = torch.constant.int 1
  %end = torch.constant.int -1
  %result = torch.aten.flatten.using_ints %input, %start, %end
      : !torch.vtensor<[1,512,1,1],f32>, !torch.int, !torch.int
        -> !torch.vtensor<[1,512],f32>
  return %result : !torch.vtensor<[1,512],f32>
}

// CHECK-LABEL: func.func @flatten
// CHECK: dnn.flatten
// CHECK-SAME: parameter_indices = array<i32: 1, 2>
// CHECK-NOT: torch.aten.flatten.using_ints
