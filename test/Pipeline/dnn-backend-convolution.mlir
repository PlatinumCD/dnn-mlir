// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=dnn.convolution' %s | FileCheck %s

module {
  func.func private @__torch__.Conv.forward(
      %arg0: !torch.nn.Module<"__torch__.Conv">,
      %input: !torch.vtensor<[1,4,8,8],f32>,
      %weight: !torch.vtensor<[4,2,3,3],f32>,
      %bias: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4,8,8],f32> {
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %groups = torch.constant.int 2
    %one = torch.prim.ListConstruct %int1, %int1
        : (!torch.int, !torch.int) -> !torch.list<int>
    %result = torch.aten.conv2d %input, %weight, %bias, %one, %one, %one, %groups
        : !torch.vtensor<[1,4,8,8],f32>, !torch.vtensor<[4,2,3,3],f32>,
          !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>,
          !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,8,8],f32>
    return %result : !torch.vtensor<[1,4,8,8],f32>
  }

  torch.class_type @__torch__.Conv {
    torch.attr private "training" : !torch.bool
    torch.attr private "_is_full_backward_hook" : !torch.optional<bool>
    torch.method "forward", @__torch__.Conv.forward
  }

  %false = torch.constant.bool false
  %none = torch.constant.none
  %module = torch.nn_module {
    torch.slot "training", %false : !torch.bool
    torch.slot "_is_full_backward_hook", %none : !torch.none
  } : !torch.nn.Module<"__torch__.Conv">
}

// CHECK-LABEL: func.func @forward
// CHECK: dnn.convolution
// CHECK-SAME: kind = "aten.conv2d"
// CHECK-NOT: torch.aten.conv2d
