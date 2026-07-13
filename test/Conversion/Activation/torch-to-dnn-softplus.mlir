// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_softplus(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %beta = torch.constant.float 1.0
    %threshold = torch.constant.float 20.0
  %result = torch.aten.softplus %input, %beta, %threshold
        : !torch.vtensor<[2,4],f32>, !torch.float, !torch.float
          -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_softplus
// CHECK: dnn.softplus
// CHECK-NOT: torch.aten.softplus

