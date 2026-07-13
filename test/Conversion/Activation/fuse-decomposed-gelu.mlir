// RUN: dnn-mlir-opt --convert-torch-to-dnn %s | FileCheck %s
// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=dnn.gelu' %s | FileCheck %s --check-prefix=PIPE

func.func @fuse_gpt2_gelu(%input: !torch.vtensor<[1,4,8],f32>)
    -> !torch.vtensor<[1,4,8],f32> {
  %three = torch.constant.int 3
  %one = torch.constant.float 1.000000e+00
  %sqrt_two_over_pi = torch.constant.float 0.79788456080286541
  %alpha = torch.constant.int 1
  %coefficient = torch.constant.float 4.471500e-02
  %half = torch.constant.float 5.000000e-01
  %0 = torch.aten.mul.Scalar %input, %half
      : !torch.vtensor<[1,4,8],f32>, !torch.float
        -> !torch.vtensor<[1,4,8],f32>
  %1 = torch.aten.pow.Tensor_Scalar %input, %three
      : !torch.vtensor<[1,4,8],f32>, !torch.int
        -> !torch.vtensor<[1,4,8],f32>
  %2 = torch.aten.mul.Scalar %1, %coefficient
      : !torch.vtensor<[1,4,8],f32>, !torch.float
        -> !torch.vtensor<[1,4,8],f32>
  %3 = torch.aten.add.Tensor %input, %2, %alpha
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>,
        !torch.int -> !torch.vtensor<[1,4,8],f32>
  %4 = torch.aten.mul.Scalar %3, %sqrt_two_over_pi
      : !torch.vtensor<[1,4,8],f32>, !torch.float
        -> !torch.vtensor<[1,4,8],f32>
  %5 = torch.aten.tanh %4
      : !torch.vtensor<[1,4,8],f32> -> !torch.vtensor<[1,4,8],f32>
  %6 = torch.aten.add.Scalar %5, %one, %alpha
      : !torch.vtensor<[1,4,8],f32>, !torch.float, !torch.int
        -> !torch.vtensor<[1,4,8],f32>
  %7 = torch.aten.mul.Tensor %0, %6
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>
        -> !torch.vtensor<[1,4,8],f32>
  return %7 : !torch.vtensor<[1,4,8],f32>
}

// A different cubic coefficient is ordinary arithmetic, not GPT-2 GELU.
func.func @reject_similar_arithmetic(%input: !torch.vtensor<[1,4,8],f32>)
    -> !torch.vtensor<[1,4,8],f32> {
  %three = torch.constant.int 3
  %one = torch.constant.float 1.000000e+00
  %scale = torch.constant.float 0.79788456080286541
  %alpha = torch.constant.int 1
  %wrong_coefficient = torch.constant.float 4.000000e-02
  %half = torch.constant.float 5.000000e-01
  %0 = torch.aten.mul.Scalar %input, %half
      : !torch.vtensor<[1,4,8],f32>, !torch.float
        -> !torch.vtensor<[1,4,8],f32>
  %1 = torch.aten.pow.Tensor_Scalar %input, %three
      : !torch.vtensor<[1,4,8],f32>, !torch.int
        -> !torch.vtensor<[1,4,8],f32>
  %2 = torch.aten.mul.Scalar %1, %wrong_coefficient
      : !torch.vtensor<[1,4,8],f32>, !torch.float
        -> !torch.vtensor<[1,4,8],f32>
  %3 = torch.aten.add.Tensor %input, %2, %alpha
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>,
        !torch.int -> !torch.vtensor<[1,4,8],f32>
  %4 = torch.aten.mul.Scalar %3, %scale
      : !torch.vtensor<[1,4,8],f32>, !torch.float
        -> !torch.vtensor<[1,4,8],f32>
  %5 = torch.aten.tanh %4
      : !torch.vtensor<[1,4,8],f32> -> !torch.vtensor<[1,4,8],f32>
  %6 = torch.aten.add.Scalar %5, %one, %alpha
      : !torch.vtensor<[1,4,8],f32>, !torch.float, !torch.int
        -> !torch.vtensor<[1,4,8],f32>
  %7 = torch.aten.mul.Tensor %0, %6
      : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,8],f32>
        -> !torch.vtensor<[1,4,8],f32>
  return %7 : !torch.vtensor<[1,4,8],f32>
}

// CHECK-LABEL: func.func @fuse_gpt2_gelu
// CHECK: dnn.gelu
// CHECK-SAME: parameters = ["tanh"]
// CHECK-NOT: torch.aten.pow
// CHECK-LABEL: func.func @reject_similar_arithmetic
// CHECK-NOT: dnn.gelu

// PIPE-LABEL: func.func @fuse_gpt2_gelu
// PIPE: dnn.gelu
// PIPE-LABEL: func.func @reject_similar_arithmetic
// PIPE-NOT: dnn.gelu
