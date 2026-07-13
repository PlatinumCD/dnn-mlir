// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s --check-prefix=DIRECT
// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=all' %s | FileCheck %s --check-prefix=PIPELINE

// A dynamic Torch scalar cannot be represented as a constant DNN attribute.
// Direct capture must leave the operation untouched, allowing the integrated
// pipeline to lower it through Torch-MLIR instead.
func.func @dynamic_leaky_relu(
    %input: !torch.vtensor<[2,4],f32>,
    %negative_slope: !torch.float) -> !torch.vtensor<[2,4],f32> {
  %result = torch.aten.leaky_relu %input, %negative_slope
      : !torch.vtensor<[2,4],f32>, !torch.float
        -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// DIRECT-LABEL: func.func @dynamic_leaky_relu
// DIRECT: torch.aten.leaky_relu
// DIRECT-NOT: dnn.leaky_relu

func.func @dynamic_add(
    %lhs: !torch.vtensor<[2,4],f32>,
    %rhs: !torch.vtensor<[2,4],f32>,
    %alpha: !torch.float) -> !torch.vtensor<[2,4],f32> {
  %result = torch.aten.add.Tensor %lhs, %rhs, %alpha
      : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.float
        -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// DIRECT-LABEL: func.func @dynamic_add
// DIRECT: torch.aten.add.Tensor
// DIRECT-NOT: dnn.add

// PIPELINE-LABEL: func.func @dynamic_leaky_relu
// PIPELINE-NOT: dnn.
// PIPELINE-NOT: torch.
// PIPELINE: linalg.generic
// PIPELINE-NOT: dnn.
// PIPELINE-NOT: torch.

// PIPELINE-LABEL: func.func @dynamic_add
// PIPELINE-NOT: dnn.
// PIPELINE-NOT: torch.
// PIPELINE: linalg.generic
// PIPELINE-NOT: dnn.
// PIPELINE-NOT: torch.
