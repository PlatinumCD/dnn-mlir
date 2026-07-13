// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_selu(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %result = torch.aten.selu %input
        : !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_selu
// CHECK: dnn.selu
// CHECK-NOT: torch.aten.selu

