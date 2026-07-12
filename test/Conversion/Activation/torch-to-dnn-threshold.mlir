// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_threshold(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %alpha = torch.constant.float 0.1
    %threshold = torch.constant.float 20.0
  %result = torch.aten.threshold %input, %threshold, %alpha
        : !torch.vtensor<[2,4],f32>, !torch.float, !torch.float
          -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_threshold
// CHECK: dnn.threshold
// CHECK-NOT: torch.aten.threshold

