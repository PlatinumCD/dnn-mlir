// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_rreluwithnoisefunctional(%input: !torch.vtensor<[2,4],f32>, %noise: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %lower = torch.constant.float 0.125
    %upper = torch.constant.float 0.3333333333333333
    %true = torch.constant.bool true
    %none = torch.constant.none
  %result, %noise_result = torch.aten.rrelu_with_noise_functional
        %input, %noise, %lower, %upper, %true, %none
        : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.float,
          !torch.float, !torch.bool, !torch.none
          -> !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_rreluwithnoisefunctional
// CHECK: dnn.rrelu_with_noise_functional
// CHECK-NOT: torch.aten.rrelu_with_noise_functional

