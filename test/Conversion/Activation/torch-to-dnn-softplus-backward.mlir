// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_softplusbackward(%input: !torch.vtensor<[2,4],f32>, %grad: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %beta = torch.constant.float 1.0
    %threshold = torch.constant.float 20.0
  %result = torch.aten.softplus_backward
        %grad, %input, %beta, %threshold
        : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.float,
          !torch.float -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_softplusbackward
// CHECK: dnn.softplus_backward
// CHECK-NOT: torch.aten.softplus_backward

