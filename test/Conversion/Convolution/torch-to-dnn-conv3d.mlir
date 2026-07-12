// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_conv3d(%input: !torch.vtensor<[1,2,5,5,5],f32>, %weight: !torch.vtensor<[4,2,3,3,3],f32>, %bias: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4,3,3,3],f32> {
  %int1 = torch.constant.int 1
  %groups = torch.constant.int 1
  %one = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %result = torch.aten.conv3d %input, %weight, %bias, %one, %one, %one, %groups
      : !torch.vtensor<[1,2,5,5,5],f32>, !torch.vtensor<[4,2,3,3,3],f32>, !torch.vtensor<[4],f32>,
        !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int
        -> !torch.vtensor<[1,4,3,3,3],f32>
  return %result : !torch.vtensor<[1,4,3,3,3],f32>
}

// CHECK-LABEL: func.func @convert_aten_conv3d
// CHECK: dnn.convolution
// CHECK-SAME: kind = "aten.conv3d"
// CHECK-NOT: torch.aten.conv3d

