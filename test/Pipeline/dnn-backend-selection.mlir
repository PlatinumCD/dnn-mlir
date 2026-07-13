// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=dnn.linear' %s | FileCheck %s
// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=dnn.linear queries=aten.mm' %s | FileCheck %s --check-prefix=BOTH
// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=all' %s | FileCheck %s --check-prefix=ALL
// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='queries=all' %s | FileCheck %s --check-prefix=ALL

// Verify capture-only selection and the union of captures with exact queries.

module attributes {torch.debug_module_name = "M"} {
  func.func private @__torch__.M.forward(
      %arg0: !torch.nn.Module<"__torch__.M">,
      %input: !torch.vtensor<[2,3],f32>,
      %weight: !torch.vtensor<[4,3],f32>,
      %bias: !torch.vtensor<[4],f32>,
      %rhs: !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[2,5],f32> {
    %linear = torch.aten.linear %input, %weight, %bias
        : !torch.vtensor<[2,3],f32>, !torch.vtensor<[4,3],f32>,
          !torch.vtensor<[4],f32> -> !torch.vtensor<[2,4],f32>
    %result = torch.aten.mm %linear, %rhs
        : !torch.vtensor<[2,4],f32>, !torch.vtensor<[4,5],f32>
          -> !torch.vtensor<[2,5],f32>
    return %result : !torch.vtensor<[2,5],f32>
  }

  torch.class_type @__torch__.M {
    torch.attr private "training" : !torch.bool
    torch.attr private "_is_full_backward_hook" : !torch.optional<bool>
    torch.method "forward", @__torch__.M.forward
  }

  %false = torch.constant.bool false
  %none = torch.constant.none
  %module = torch.nn_module {
    torch.slot "training", %false : !torch.bool
    torch.slot "_is_full_backward_hook", %none : !torch.none
  } : !torch.nn.Module<"__torch__.M">
}

// CHECK-LABEL: func.func @forward
// CHECK: dnn.linear
// CHECK: linalg.matmul
// CHECK-NOT: dnn.mm
// CHECK-NOT: torch.aten.linear
// CHECK-NOT: torch.aten.mm

// BOTH-LABEL: func.func @forward
// BOTH: dnn.linear
// BOTH: dnn.mm
// BOTH-NOT: linalg.matmul
// BOTH-NOT: torch.aten.linear
// BOTH-NOT: torch.aten.mm

// ALL-LABEL: func.func @forward
// ALL: dnn.linear
// ALL: dnn.mm
// ALL-NOT: linalg.matmul
// ALL-NOT: torch.aten.linear
// ALL-NOT: torch.aten.mm
