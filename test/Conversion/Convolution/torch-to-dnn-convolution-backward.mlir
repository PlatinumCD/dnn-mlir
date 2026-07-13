// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_convolution_backward(%grad: !torch.vtensor<[1,4,6,6],f32>, %input: !torch.vtensor<[1,2,8,8],f32>, %weight: !torch.vtensor<[4,2,3,3],f32>) -> (!torch.vtensor<[1,2,8,8],f32>, !torch.vtensor<[4,2,3,3],f32>, !torch.vtensor<[4],f32>) {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %false = torch.constant.bool false
  %bias_sizes = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
  %one = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %zero = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %true = torch.constant.bool true
  %mask = torch.prim.ListConstruct %true, %true, %true : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
  %groups = torch.constant.int 1
  %none = torch.constant.none
  %result0, %result1, %result2 = torch.aten.convolution_backward
      %grad, %input, %weight, %bias_sizes, %one, %zero, %one, %false, %zero, %groups, %mask
      : !torch.vtensor<[1,4,6,6],f32>, !torch.vtensor<[1,2,8,8],f32>,
        !torch.vtensor<[4,2,3,3],f32>, !torch.list<int>, !torch.list<int>,
        !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>,
        !torch.int, !torch.list<bool>
        -> !torch.vtensor<[1,2,8,8],f32>, !torch.vtensor<[4,2,3,3],f32>,
           !torch.vtensor<[4],f32>
  return %result0, %result1, %result2 : !torch.vtensor<[1,2,8,8],f32>, !torch.vtensor<[4,2,3,3],f32>, !torch.vtensor<[4],f32>
}

// CHECK-LABEL: func.func @convert_aten_convolution_backward
// CHECK: dnn.convolution
// CHECK-SAME: kind = "aten.convolution_backward"
// CHECK-NOT: torch.aten.convolution_backward
