// RUN: dnn-opt -verify-diagnostics %s

func.func @invalid_mm(%lhs: tensor<2x3xf32>, %rhs: tensor<5x4xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{has mismatching contracting dimensions}}
  %result = dnn.mm %lhs, %rhs
      : tensor<2x3xf32>, tensor<5x4xf32> -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

func.func @invalid_rnn_activation(%input: tensor<5x2x4xf32>)
    -> (tensor<5x2x3xf32>, tensor<1x2x3xf32>) {
  // expected-error @+1 {{requires activation to be 'tanh' or 'relu'}}
  %result:2 = "dnn.rnn"(%input) <{
      activation = "sigmoid",
      kind = "aten.rnn_tanh.input",
      operand_groups = array<i32: 1>
    }> : (tensor<5x2x4xf32>)
      -> (tensor<5x2x3xf32>, tensor<1x2x3xf32>)
  return %result#0, %result#1
      : tensor<5x2x3xf32>, tensor<1x2x3xf32>
}
