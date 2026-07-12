// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_logsoftmaxint(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %dim = torch.constant.int 1
    %none = torch.constant.none
  %result = torch.aten.log_softmax.int %input, %dim, %none
        : !torch.vtensor<[2,4],f32>, !torch.int, !torch.none
          -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_logsoftmaxint
// CHECK: dnn.log_softmax_int
// CHECK-NOT: torch.aten.log_softmax.int

