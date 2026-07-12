// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_softmaxint(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %dim = torch.constant.int 1
    %none = torch.constant.none
  %result = torch.aten.softmax.int %input, %dim, %none
        : !torch.vtensor<[2,4],f32>, !torch.int, !torch.none
          -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_softmaxint
// CHECK: dnn.softmax_int
// CHECK-NOT: torch.aten.softmax.int

