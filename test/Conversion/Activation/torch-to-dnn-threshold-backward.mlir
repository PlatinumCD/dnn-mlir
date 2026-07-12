// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_thresholdbackward(%input: !torch.vtensor<[2,4],f32>, %grad: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %threshold = torch.constant.float 20.0
  %result = torch.aten.threshold_backward %grad, %input, %threshold
        : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.float
          -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_thresholdbackward
// CHECK: dnn.threshold_backward
// CHECK-NOT: torch.aten.threshold_backward

