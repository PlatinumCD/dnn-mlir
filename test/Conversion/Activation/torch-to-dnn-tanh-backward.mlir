// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_tanhbackward(
    %input: !torch.vtensor<[2,4],f32>,
    %grad: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %result = torch.aten.tanh_backward %grad, %input
      : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>
        -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_tanhbackward
// CHECK: dnn.tanh_backward
// CHECK-NOT: torch.aten.tanh_backward

