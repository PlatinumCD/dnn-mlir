// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_conv2d_grouped(%input: !torch.vtensor<[1,4,8,8],f32>, %weight: !torch.vtensor<[4,2,3,3],f32>, %bias: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4,8,8],f32> {
  %int1 = torch.constant.int 1
  %int2 = torch.constant.int 2
  %groups = torch.constant.int 2
  %one = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %int0 = torch.constant.int 0
  %zero = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.aten.conv2d %input, %weight, %bias, %one, %zero, %one, %groups
      : !torch.vtensor<[1,4,8,8],f32>, !torch.vtensor<[4,2,3,3],f32>, !torch.vtensor<[4],f32>,
        !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int
        -> !torch.vtensor<[1,4,8,8],f32>
  return %result : !torch.vtensor<[1,4,8,8],f32>
}

// CHECK-LABEL: func.func @convert_aten_conv2d_grouped
// CHECK: dnn.convolution
// CHECK-SAME: kind = "aten.conv2d"
// CHECK-NOT: torch.aten.conv2d

