// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_glu(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %dim = torch.constant.int 1
  %result = torch.aten.glu %input, %dim
        : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_glu
// CHECK: dnn.glu
// CHECK-NOT: torch.aten.glu

