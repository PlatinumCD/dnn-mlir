// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_softmax(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %dim = torch.constant.int 1
    %false = torch.constant.bool false
  %result = torch.aten._softmax %input, %dim, %false
        : !torch.vtensor<[2,4],f32>, !torch.int, !torch.bool
          -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_softmax
// CHECK: dnn.softmax
// CHECK-NOT: torch.aten._softmax

