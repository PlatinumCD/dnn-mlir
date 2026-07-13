// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @mul(%features: !torch.vtensor<[1,96,14,14],f32>,
               %gate: !torch.vtensor<[1,96,1,1],f32>)
    -> !torch.vtensor<[1,96,14,14],f32> {
  %result = torch.aten.mul.Tensor %features, %gate
      : !torch.vtensor<[1,96,14,14],f32>,
        !torch.vtensor<[1,96,1,1],f32>
        -> !torch.vtensor<[1,96,14,14],f32>
  return %result : !torch.vtensor<[1,96,14,14],f32>
}

// CHECK-LABEL: func.func @mul
// CHECK: dnn.mul
// CHECK-NOT: torch.aten.mul.Tensor
