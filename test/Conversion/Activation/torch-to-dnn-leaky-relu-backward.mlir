// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_leakyrelubackward(%input: !torch.vtensor<[2,4],f32>, %grad: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %alpha = torch.constant.float 0.1
    %false = torch.constant.bool false
  %result = torch.aten.leaky_relu_backward
        %grad, %input, %alpha, %false
        : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.float,
          !torch.bool -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_leakyrelubackward
// CHECK: dnn.leaky_relu_backward
// CHECK-NOT: torch.aten.leaky_relu_backward

