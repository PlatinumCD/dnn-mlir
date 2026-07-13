// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @adaptive_avg_pool2d(%input: !torch.vtensor<[1,512,7,7],f32>)
    -> !torch.vtensor<[1,512,1,1],f32> {
  %int1 = torch.constant.int 1
  %output_size = torch.prim.ListConstruct %int1, %int1
      : (!torch.int, !torch.int) -> !torch.list<int>
  %result = torch.aten.adaptive_avg_pool2d %input, %output_size
      : !torch.vtensor<[1,512,7,7],f32>, !torch.list<int>
        -> !torch.vtensor<[1,512,1,1],f32>
  return %result : !torch.vtensor<[1,512,1,1],f32>
}

// CHECK-LABEL: func.func @adaptive_avg_pool2d
// CHECK: dnn.adaptive_avg_pool2d
// CHECK-SAME: parameter_indices = array<i32: 1>
// CHECK-NOT: torch.aten.adaptive_avg_pool2d
