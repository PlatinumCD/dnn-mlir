// RUN: dnn-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=dnn.linear' %s | FileCheck %s --check-prefix=SELECTED
// RUN: dnn-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline %s | FileCheck %s --check-prefix=NONE

module attributes {torch.debug_module_name = "M"} {
  func.func private @__torch__.M.forward(
      %arg0: !torch.nn.Module<"__torch__.M">,
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

// SELECTED-LABEL: func.func @forward(%arg0: tensor<2x3xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<4xf32>) -> tensor<2x4xf32>
// SELECTED: dnn.linear
// SELECTED: linalg.generic
// SELECTED-NOT: torch.aten.linear
// SELECTED-NOT: torch.aten.relu
// SELECTED-NOT: torch_c.

// NONE-LABEL: func.func @forward
// NONE: linalg.matmul
// NONE: linalg.generic
// NONE-NOT: dnn.
