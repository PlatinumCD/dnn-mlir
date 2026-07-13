// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @leave_unmatched(
    %input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  // CHECK-LABEL: func.func @leave_unmatched
  // CHECK: torch.aten.div.Tensor
  %result = torch.aten.div.Tensor %input, %input
      : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>
        -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}
