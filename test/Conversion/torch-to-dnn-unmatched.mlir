// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @leave_unmatched(
    %input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  // CHECK-LABEL: func.func @leave_unmatched
  // CHECK: torch.aten.add.Tensor
  %alpha = torch.constant.float 1.0
  %result = torch.aten.add.Tensor %input, %input, %alpha
      : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.float
        -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}
