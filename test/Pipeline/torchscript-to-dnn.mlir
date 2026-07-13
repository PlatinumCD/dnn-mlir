// RUN: dnn-mlir-opt -torchscript-to-dnn-pipeline='captures=dnn.mm' %s | FileCheck %s

module attributes {torch.debug_module_name = "M"} {
  func.func private @__torch__.M.forward(
      %arg0: !torch.nn.Module<"__torch__.M">,
      %arg1: !torch.tensor {
        torch.type_bound = !torch.vtensor<[2,3],f32>
      },
      %arg2: !torch.tensor {
        torch.type_bound = !torch.vtensor<[3,4],f32>
      }) -> !torch.tensor {
    %lhs = torch.tensor_static_info_cast %arg1
        : !torch.tensor to !torch.tensor<[2,3],f32>
    %rhs = torch.tensor_static_info_cast %arg2
        : !torch.tensor to !torch.tensor<[3,4],f32>
    %result = torch.aten.mm %lhs, %rhs
        : !torch.tensor<[2,3],f32>, !torch.tensor<[3,4],f32>
          -> !torch.tensor<[2,4],f32>
    %return = torch.tensor_static_info_cast %result
        : !torch.tensor<[2,4],f32> to !torch.tensor
    return %return : !torch.tensor
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
// CHECK: dnn.mm
// CHECK-NOT: torch.aten.mm
// CHECK-NOT: torch.nn_module
