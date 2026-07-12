// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_elubackward(%input: !torch.vtensor<[2,4],f32>, %grad: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %alpha = torch.constant.float 0.1
    %scale = torch.constant.float 1.0
    %input_scale = torch.constant.float 1.0
    %false = torch.constant.bool false
  %result = torch.aten.elu_backward
        %grad, %alpha, %scale, %input_scale, %false, %input
        : !torch.vtensor<[2,4],f32>, !torch.float, !torch.float, !torch.float,
          !torch.bool, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_elubackward
// CHECK: dnn.elu_backward
// CHECK-NOT: torch.aten.elu_backward

