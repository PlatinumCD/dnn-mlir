// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_softshrink(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %alpha = torch.constant.float 0.1
  %result = torch.aten.softshrink %input, %alpha
        : !torch.vtensor<[2,4],f32>, !torch.float -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_softshrink
// CHECK: dnn.softshrink
// CHECK-NOT: torch.aten.softshrink

