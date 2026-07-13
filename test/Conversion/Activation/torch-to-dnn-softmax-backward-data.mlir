// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_softmaxbackwarddata(%input: !torch.vtensor<[2,4],f32>, %grad: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %dim = torch.constant.int 1
  %result = torch.aten._softmax_backward_data
        %grad, %input, %dim, %dim
        : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.int,
          !torch.int -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_softmaxbackwarddata
// CHECK: dnn.softmax_backward_data
// CHECK-NOT: torch.aten._softmax_backward_data

