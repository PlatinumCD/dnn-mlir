// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_prelu(%input: !torch.vtensor<[2,4],f32>, %weight: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %result = torch.aten.prelu %input, %weight
        : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>
          -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_prelu
// CHECK: dnn.prelu
// CHECK-NOT: torch.aten.prelu

