// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_rreluwithnoisebackward(%input: !torch.vtensor<[2,4],f32>, %grad: !torch.vtensor<[2,4],f32>, %noise: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %lower = torch.constant.float 0.125
    %upper = torch.constant.float 0.3333333333333333
    %true = torch.constant.bool true
    %false = torch.constant.bool false
  %result = torch.aten.rrelu_with_noise_backward
        %grad, %input, %noise, %lower, %upper, %true, %false
        : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>,
          !torch.vtensor<[2,4],f32>, !torch.float, !torch.float, !torch.bool,
          !torch.bool -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_rreluwithnoisebackward
// CHECK: dnn.rrelu_with_noise_backward
// CHECK-NOT: torch.aten.rrelu_with_noise_backward

