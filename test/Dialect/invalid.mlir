// RUN: dnn-mlir-opt -verify-diagnostics %s

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

func.func @invalid_mutable_torch_operand(
    %input: !torch.tensor<[2,4],f32>) -> !torch.tensor<[2,4],f32> {
  // expected-error @+1 {{operand #0 must be variadic of ranked tensor}}
  %result = dnn.relu %input
      : (!torch.tensor<[2,4],f32>) -> !torch.tensor<[2,4],f32>
  return %result : !torch.tensor<[2,4],f32>
}

func.func @invalid_matmul_batch_broadcast(
    %lhs: tensor<2x3x4xf32>, %rhs: tensor<5x4x6xf32>)
    -> tensor<5x3x6xf32> {
  // expected-error @+1 {{has non-broadcastable batch dimensions 2 and 5}}
  %result = dnn.matmul %lhs, %rhs
      : tensor<2x3x4xf32>, tensor<5x4x6xf32> -> tensor<5x3x6xf32>
  return %result : tensor<5x3x6xf32>
}

func.func @invalid_matmul_result_shape(
    %lhs: tensor<2x3x4xf32>, %rhs: tensor<1x4x6xf32>)
    -> tensor<3x3x6xf32> {
  // expected-error @+1 {{result shape does not match the broadcasted matmul shape}}
  %result = dnn.matmul %lhs, %rhs
      : tensor<2x3x4xf32>, tensor<1x4x6xf32> -> tensor<3x3x6xf32>
  return %result : tensor<3x3x6xf32>
}

func.func @invalid_missing_parameters(%input: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{requires parameter_indices and parameters to appear together}}
  %result = dnn.relu %input {parameter_indices = array<i32: 1>}
      : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

func.func @invalid_parameter_lengths(%input: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{requires equal parameter_indices and parameters lengths}}
  %result = dnn.relu %input {
      parameter_indices = array<i32: 1, 2>, parameters = [0.1]
    } : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

func.func @invalid_duplicate_parameter_indices(%input: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{requires strictly increasing, unique parameter indices}}
  %result = dnn.relu %input {
      parameter_indices = array<i32: 1, 1>, parameters = [0.1, 0.2]
    } : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

func.func @invalid_descending_parameter_indices(%input: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{requires strictly increasing, unique parameter indices}}
  %result = dnn.relu %input {
      parameter_indices = array<i32: 2, 1>, parameters = [0.1, 0.2]
    } : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

func.func @invalid_parameter_index_range(%input: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  // expected-error @+1 {{has a parameter index outside the original operand range}}
  %result = dnn.relu %input {
      parameter_indices = array<i32: 2>, parameters = [0.1]
    } : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}

func.func @invalid_recurrent_operand_groups(%input: tensor<5x2x4xf32>)
    -> (tensor<5x2x3xf32>, tensor<1x2x3xf32>) {
  // expected-error @+1 {{requires operand_groups to account for every operand}}
  %result:2 = dnn.rnn %input {
      activation = "tanh", kind = "aten.rnn_tanh.input",
      operand_groups = array<i32: 2>
    } : (tensor<5x2x4xf32>)
      -> (tensor<5x2x3xf32>, tensor<1x2x3xf32>)
  return %result#0, %result#1
      : tensor<5x2x3xf32>, tensor<1x2x3xf32>
}

func.func @invalid_dnn_scalar_operand(%input: f32) -> tensor<2x4xf32> {
  // expected-error @+1 {{operand #0 must be variadic of ranked tensor}}
  %result = dnn.relu %input : (f32) -> tensor<2x4xf32>
  return %result : tensor<2x4xf32>
}
