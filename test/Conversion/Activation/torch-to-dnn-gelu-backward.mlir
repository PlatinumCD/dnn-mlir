// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_gelubackward(%input: !torch.vtensor<[2,4],f32>, %grad: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %approximate = torch.constant.str "none"
  %result = torch.aten.gelu_backward %grad, %input, %approximate
        : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.str
          -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_gelubackward
// CHECK: dnn.gelu_backward
// CHECK-NOT: torch.aten.gelu_backward

