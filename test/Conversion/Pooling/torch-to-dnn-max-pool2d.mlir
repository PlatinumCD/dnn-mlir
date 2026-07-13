// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @max_pool2d(%input: !torch.vtensor<[1,64,112,112],f32>)
    -> !torch.vtensor<[1,64,56,56],f32> {
  %int3 = torch.constant.int 3
  %kernel = torch.prim.ListConstruct %int3, %int3
      : (!torch.int, !torch.int) -> !torch.list<int>
  %int2 = torch.constant.int 2
  %stride = torch.prim.ListConstruct %int2, %int2
      : (!torch.int, !torch.int) -> !torch.list<int>
  %int1 = torch.constant.int 1
  %padding = torch.prim.ListConstruct %int1, %int1
      : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1
      : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %result = torch.aten.max_pool2d %input, %kernel, %stride, %padding,
      %dilation, %false
      : !torch.vtensor<[1,64,112,112],f32>, !torch.list<int>,
        !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool
        -> !torch.vtensor<[1,64,56,56],f32>
  return %result : !torch.vtensor<[1,64,56,56],f32>
}

// CHECK-LABEL: func.func @max_pool2d
// CHECK: dnn.max_pool2d
// CHECK-SAME: parameter_indices = array<i32: 1, 2, 3, 4, 5>
// CHECK-NOT: torch.aten.max_pool2d
