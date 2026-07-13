// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='queries=aten.relu' %s | FileCheck %s

module {
  func.func private @__torch__.Activation.forward(
      %arg0: !torch.nn.Module<"__torch__.Activation">,
      %input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
    %result = torch.aten.relu %input
        : !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
    return %result : !torch.vtensor<[2,4],f32>
  }

  torch.class_type @__torch__.Activation {
    torch.attr private "training" : !torch.bool
    torch.attr private "_is_full_backward_hook" : !torch.optional<bool>
    torch.method "forward", @__torch__.Activation.forward
  }

  %false = torch.constant.bool false
  %none = torch.constant.none
  %module = torch.nn_module {
    torch.slot "training", %false : !torch.bool
    torch.slot "_is_full_backward_hook", %none : !torch.none
  } : !torch.nn.Module<"__torch__.Activation">
}

// CHECK-LABEL: func.func @forward
// CHECK: dnn.relu
// CHECK-NOT: torch.aten.relu
