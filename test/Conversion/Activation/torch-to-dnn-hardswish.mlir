// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_hardswish(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %result = torch.aten.hardswish %input
        : !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_hardswish
// CHECK: dnn.hardswish
// CHECK-NOT: torch.aten.hardswish

