// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_conv1d(%input: !torch.vtensor<[1,2,8],f32>, %weight: !torch.vtensor<[4,2,3],f32>, %bias: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4,6],f32> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %groups = torch.constant.int 1
  %none = torch.constant.none
  %one = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %zero = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
  %result = torch.aten.conv1d %input, %weight, %bias, %one, %zero, %one, %groups
      : !torch.vtensor<[1,2,8],f32>, !torch.vtensor<[4,2,3],f32>, !torch.vtensor<[4],f32>,
        !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int
        -> !torch.vtensor<[1,4,6],f32>
  return %result : !torch.vtensor<[1,4,6],f32>
}

// CHECK-LABEL: func.func @convert_aten_conv1d
// CHECK: dnn.convolution
// CHECK-SAME: kind = "aten.conv1d"
// CHECK-NOT: torch.aten.conv1d

