// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=dnn.add' %s | FileCheck %s --check-prefix=ADD
// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=dnn.relu' %s | FileCheck %s --check-prefix=RELU
// RUN: not dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='queries=aten.add_.Tensor' %s 2>&1 | FileCheck %s --check-prefix=INVALID

// INVALID: no DNN query is registered for 'aten.add_.Tensor'

// A regular add is already value-semantic and can be captured directly.
func.func @regular_add(
    %lhs: !torch.vtensor<[2,2],f32>,
    %rhs: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> {
  %alpha = torch.constant.int 1
  %result = torch.aten.add.Tensor %lhs, %rhs, %alpha
      : !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>, !torch.int
        -> !torch.vtensor<[2,2],f32>
  return %result : !torch.vtensor<[2,2],f32>
}

// ADD-LABEL: func.func @regular_add
// ADD: %[[REGULAR_RESULT:.*]] = dnn.add
// ADD: return %[[REGULAR_RESULT]]

// The two returns alias the same mutable input before functionalization. Both
// must become the single functional DNN result; returning the original input
// for either value would lose the semantics of aten.add_.Tensor.
func.func @inplace_add_alias(
    %lhs_value: !torch.vtensor<[2,2],f32>,
    %rhs_value: !torch.vtensor<[2,2],f32>)
    -> (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) {
  %lhs = torch.copy.to_tensor %lhs_value : !torch.tensor<[2,2],f32>
  %rhs = torch.copy.to_tensor %rhs_value : !torch.tensor<[2,2],f32>
  %alpha = torch.constant.int 1
  %result = torch.aten.add_.Tensor %lhs, %rhs, %alpha
      : !torch.tensor<[2,2],f32>, !torch.tensor<[2,2],f32>, !torch.int
        -> !torch.tensor<[2,2],f32>
  %result_value = torch.copy.to_vtensor %result : !torch.vtensor<[2,2],f32>
  %alias_value = torch.copy.to_vtensor %lhs : !torch.vtensor<[2,2],f32>
  return %result_value, %alias_value
      : !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>
}

// ADD-LABEL: func.func @inplace_add_alias
// ADD: %[[RESULT:.*]] = dnn.add
// ADD-NOT: dnn.relu_inplace
// ADD-NOT: torch.aten.add_.Tensor
// ADD: return %[[RESULT]], %[[RESULT]]

// This is the same aliasing case for a trailing-underscore activation. It must
// produce dnn.relu, never a mutation-bearing DNN operation.
func.func @inplace_relu_alias(%input_value: !torch.vtensor<[2,2],f32>)
    -> (!torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>) {
  %input = torch.copy.to_tensor %input_value : !torch.tensor<[2,2],f32>
  %result = torch.aten.relu_ %input
      : !torch.tensor<[2,2],f32> -> !torch.tensor<[2,2],f32>
  %result_value = torch.copy.to_vtensor %result : !torch.vtensor<[2,2],f32>
  %alias_value = torch.copy.to_vtensor %input : !torch.vtensor<[2,2],f32>
  return %result_value, %alias_value
      : !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32>
}

// RELU-LABEL: func.func @inplace_relu_alias
// RELU: %[[RESULT:.*]] = dnn.relu
// RELU-NOT: dnn.relu_inplace
// RELU-NOT: torch.aten.relu_
// RELU: return %[[RESULT]], %[[RESULT]]
