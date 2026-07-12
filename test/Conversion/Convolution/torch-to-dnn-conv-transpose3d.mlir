// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_conv_transpose3d(%input: !torch.vtensor<[1,2,5,5,5],f32>, %weight: !torch.vtensor<[2,4,3,3,3],f32>, %bias: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4,7,7,7],f32> {
  %int1 = torch.constant.int 1
  %groups = torch.constant.int 1
  %one = torch.prim.ListConstruct %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %result = torch.aten.conv_transpose3d.input %input, %weight, %bias, %one, %one, %one, %groups, %one
      : !torch.vtensor<[1,2,5,5,5],f32>, !torch.vtensor<[2,4,3,3,3],f32>, !torch.vtensor<[4],f32>,
        !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int, !torch.list<int>
        -> !torch.vtensor<[1,4,7,7,7],f32>
  return %result : !torch.vtensor<[1,4,7,7,7],f32>
}

// CHECK-LABEL: func.func @convert_aten_conv_transpose3d
// CHECK: dnn.convolution
// CHECK-SAME: kind = "aten.conv_transpose3d.input"
// CHECK-NOT: torch.aten.conv_transpose3d.input

