// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_underscore_convolution_deprecated(%input: !torch.vtensor<[1,2,8,8],f32>, %weight: !torch.vtensor<[4,2,3,3],f32>, %bias: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4,6,6],f32> {
  %int1 = torch.constant.int 1
  %false = torch.constant.bool false
  %groups = torch.constant.int 1
  %one = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %int0 = torch.constant.int 0
  %zero = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.aten._convolution.deprecated %input, %weight, %bias, %one, %zero, %one, %false, %zero, %groups, %false, %false, %false
      : !torch.vtensor<[1,2,8,8],f32>, !torch.vtensor<[4,2,3,3],f32>, !torch.vtensor<[4],f32>,
        !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int,
        !torch.bool, !torch.bool, !torch.bool
        -> !torch.vtensor<[1,4,6,6],f32>
  return %result : !torch.vtensor<[1,4,6,6],f32>
}

// CHECK-LABEL: func.func @convert_aten_underscore_convolution_deprecated
// CHECK: dnn.convolution
// CHECK-SAME: kind = "aten._convolution.deprecated"
// CHECK-NOT: torch.aten._convolution.deprecated

