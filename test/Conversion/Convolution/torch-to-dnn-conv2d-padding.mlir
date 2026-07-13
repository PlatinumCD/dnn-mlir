// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_conv2d_padding(%input: !torch.vtensor<[1,2,8,8],f32>, %weight: !torch.vtensor<[4,2,3,3],f32>, %bias: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4,8,8],f32> {
  %int1 = torch.constant.int 1
  %groups = torch.constant.int 1
  %padding = torch.constant.str "same"
  %one = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.aten.conv2d.padding %input, %weight, %bias, %one, %padding, %one, %groups
      : !torch.vtensor<[1,2,8,8],f32>, !torch.vtensor<[4,2,3,3],f32>, !torch.vtensor<[4],f32>,
        !torch.list<int>, !torch.str, !torch.list<int>, !torch.int
        -> !torch.vtensor<[1,4,8,8],f32>
  return %result : !torch.vtensor<[1,4,8,8],f32>
}

// CHECK-LABEL: func.func @convert_aten_conv2d_padding
// CHECK: dnn.convolution
// CHECK-SAME: kind = "aten.conv2d.padding"
// CHECK-NOT: torch.aten.conv2d.padding

