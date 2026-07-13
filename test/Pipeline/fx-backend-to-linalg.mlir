// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=all' %s | FileCheck %s

// Flat func.func input is the form emitted by Torch-MLIR's FX importer. It
// must bypass TorchScript object-graph normalization and enter DNN capture
// directly.

module {
  func.func @forward(
      %input: !torch.vtensor<[2,3],f32>,
      %weight: !torch.vtensor<[4,3],f32>,
      %bias: !torch.vtensor<[4],f32>) -> !torch.vtensor<[2,4],f32> {
    %linear = torch.aten.linear %input, %weight, %bias
        : !torch.vtensor<[2,3],f32>, !torch.vtensor<[4,3],f32>,
          !torch.vtensor<[4],f32> -> !torch.vtensor<[2,4],f32>
    %result = torch.aten.relu %linear
        : !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
    return %result : !torch.vtensor<[2,4],f32>
  }
}

// CHECK-LABEL: func.func @forward
// CHECK: dnn.linear
// CHECK: dnn.relu
// CHECK-NOT: torch.aten
