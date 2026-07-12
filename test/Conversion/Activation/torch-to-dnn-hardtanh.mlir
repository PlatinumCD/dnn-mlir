// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_hardtanh(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %min = torch.constant.float -1.0
    %max = torch.constant.float 1.0
  %result = torch.aten.hardtanh %input, %min, %max
        : !torch.vtensor<[2,4],f32>, !torch.float, !torch.float
          -> !torch.vtensor<[2,4],f32>
  return %result : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_aten_hardtanh
// CHECK: dnn.hardtanh
// CHECK-NOT: torch.aten.hardtanh

