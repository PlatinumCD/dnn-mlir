// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s
// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=dnn.matmul' %s | FileCheck %s --check-prefix=PIPE
// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='queries=aten.matmul' %s | FileCheck %s --check-prefix=PIPE

func.func @convert_batched_matrix_matrix_matmul(
    %lhs: !torch.vtensor<[1,128,768],f32>,
    %rhs: !torch.vtensor<[768,50257],f32>)
    -> !torch.vtensor<[1,128,50257],f32> {
  // CHECK-LABEL: func.func @convert_batched_matrix_matrix_matmul
  // CHECK: torch_c.to_builtin_tensor
  // CHECK: torch_c.to_builtin_tensor
  // CHECK: dnn.matmul
  // CHECK: torch_c.from_builtin_tensor
  // CHECK-NOT: torch.aten.matmul
  %result = torch.aten.matmul %lhs, %rhs
      : !torch.vtensor<[1,128,768],f32>, !torch.vtensor<[768,50257],f32>
        -> !torch.vtensor<[1,128,50257],f32>
  return %result : !torch.vtensor<[1,128,50257],f32>
}

func.func @convert_vector_vector_matmul(
    %lhs: !torch.vtensor<[8],f32>,
    %rhs: !torch.vtensor<[8],f32>) -> !torch.vtensor<[],f32> {
  // CHECK-LABEL: func.func @convert_vector_vector_matmul
  // CHECK: dnn.matmul
  // CHECK-NOT: torch.aten.matmul
  %result = torch.aten.matmul %lhs, %rhs
      : !torch.vtensor<[8],f32>, !torch.vtensor<[8],f32>
        -> !torch.vtensor<[],f32>
  return %result : !torch.vtensor<[],f32>
}

// PIPE-LABEL: func.func @convert_batched_matrix_matrix_matmul
// PIPE: dnn.matmul
// PIPE-NOT: linalg.batch_matmul
// PIPE-NOT: torch.aten.matmul
