// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_sigmoid(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %result = torch.aten.sigmoid %input
        : !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_sigmoid
// CHECK: dnn.sigmoid
// CHECK-NOT: torch.aten.sigmoid

