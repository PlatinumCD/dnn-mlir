// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_aten_conv_tbc(%input: !torch.vtensor<[5,2,8],f32>, %weight: !torch.vtensor<[3,2,3],f32>, %bias: !torch.vtensor<[3],f32>) -> !torch.vtensor<[5,2,8],f32> {
  %pad = torch.constant.int 1
  %result = torch.aten.conv_tbc %input, %weight, %bias, %pad
      : !torch.vtensor<[5,2,8],f32>, !torch.vtensor<[3,2,3],f32>,
        !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[5,2,8],f32>
  return %result : !torch.vtensor<[5,2,8],f32>
}

// CHECK-LABEL: func.func @convert_aten_conv_tbc
// CHECK: dnn.convolution
// CHECK-SAME: kind = "aten.conv_tbc"
// CHECK-NOT: torch.aten.conv_tbc

