// RUN: dnn-opt %s | FileCheck %s

func.func @parse_and_print(
    %lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>,
    %weight: tensor<4x3xf32>, %bias: tensor<4xf32>)
    -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
  // CHECK: dnn.mm
  %mm = dnn.mm %lhs, %rhs
      : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>

  // CHECK: dnn.linear
  %with_bias = dnn.linear %lhs, %weight, %bias
      : tensor<2x3xf32>, tensor<4x3xf32>, tensor<4xf32>
        -> tensor<2x4xf32>

  // CHECK: dnn.linear
  %without_bias = dnn.linear %lhs, %weight
      : tensor<2x3xf32>, tensor<4x3xf32> -> tensor<2x4xf32>

  return %mm, %with_bias, %without_bias
      : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
}

func.func @parse_and_print_rnn(
    %input: tensor<5x2x4xf32>, %h0: tensor<1x2x3xf32>,
    %wih: tensor<3x4xf32>, %whh: tensor<3x3xf32>,
    %bih: tensor<3xf32>, %bhh: tensor<3xf32>)
    -> (tensor<5x2x3xf32>, tensor<1x2x3xf32>) {
  // CHECK: = dnn.rnn
  // CHECK-SAME: activation = "tanh"
  // CHECK-NOT: "dnn.rnn"
  %result:2 = dnn.rnn %input, %h0, %wih, %whh, %bih, %bhh {
      activation = "tanh",
      kind = "aten.rnn_tanh.input",
      operand_groups = array<i32: 1, 1, 4, 0, 0, 0, 0, 0, 0>,
      parameter_indices = array<i32: 3, 4, 5, 6, 7, 8>,
      parameters = [true, 1, 0.000000e+00, false, false, false]
    } : (tensor<5x2x4xf32>, tensor<1x2x3xf32>, tensor<3x4xf32>,
         tensor<3x3xf32>, tensor<3xf32>, tensor<3xf32>)
        -> (tensor<5x2x3xf32>, tensor<1x2x3xf32>)
  return %result#0, %result#1 : tensor<5x2x3xf32>, tensor<1x2x3xf32>
}

func.func @parse_and_print_activation(%input: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  // CHECK: dnn.relu
  // CHECK-NOT: dnn.activation
  %result = dnn.relu %input : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
