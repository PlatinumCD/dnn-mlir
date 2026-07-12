// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_rrelu(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %lower = torch.constant.float 0.125
    %upper = torch.constant.float 0.3333333333333333
    %false = torch.constant.bool false
    %none = torch.constant.none
  %result = torch.aten.rrelu %input, %lower, %upper, %false, %none
        : !torch.vtensor<[2,4],f32>, !torch.float, !torch.float, !torch.bool,
          !torch.none -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_rrelu
// CHECK: dnn.rrelu
// CHECK-NOT: torch.aten.rrelu

