// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_elu(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %alpha = torch.constant.float 0.1
    %scale = torch.constant.float 1.0
    %input_scale = torch.constant.float 1.0
  %result = torch.aten.elu %input, %alpha, %scale, %input_scale
        : !torch.vtensor<[2,4],f32>, !torch.float, !torch.float, !torch.float
          -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_elu
// CHECK: dnn.elu
// CHECK-NOT: torch.aten.elu

