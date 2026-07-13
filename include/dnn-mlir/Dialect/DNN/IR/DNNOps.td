#ifndef DNN_MLIR_DIALECT_DNN_IR_DNNOPS_TD
#define DNN_MLIR_DIALECT_DNN_IR_DNNOPS_TD

include "dnn-mlir/Dialect/DNN/IR/DNNBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DNN_MmOp : DNN_Op<"mm", [Pure]> {
  let summary = "Rank-two matrix multiplication";
  let description = [{
    Computes `lhs * rhs` for rank-two builtin tensors. The contracting
    dimensions and element types must agree.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
  let hasVerifier = 1;
}

def DNN_MatmulOp : DNN_Op<"matmul", [Pure]> {
  let summary = "Rank-polymorphic matrix multiplication";
  let description = [{
    Computes PyTorch-style matrix multiplication for tensor operands of rank
    one or greater. Vector, matrix, batched-matrix, and broadcasted batch
    dimension forms are supported.
  }];
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
  let hasVerifier = 1;
}

def DNN_LinearOp : DNN_Op<"linear", [Pure]> {
  let summary = "Dense linear transformation";
  let description = [{
    Computes `input * transpose(weight) + bias`. The final input dimension is
    the input-feature dimension, and weight has shape `[out_features,
    in_features]`. Bias is optional and has shape `[out_features]`.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Optional<AnyRankedTensor>:$bias
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict `:`
      type($input) `,` type($weight) (`,` type($bias)^)? `->` type($result)
  }];
  let hasVerifier = 1;
}

// Structured neural-network operations share a flexible operand/result form
// because their Torch schemas include optional tensors and scalar/list
// configuration. Tensor values are bridged to builtin tensors; constant
// configuration is stored with its original operand position.
class DNN_StructuredOp<string mnemonic, string opSummary>
    : DNN_Op<mnemonic, [DNN_ParameterMetadata]> {
  let summary = opSummary;
  let arguments = (ins
    Variadic<AnyRankedTensor>:$operands,
    OptionalAttr<StrAttr>:$kind,
    OptionalAttr<DenseI32ArrayAttr>:$parameter_indices,
    OptionalAttr<ArrayAttr>:$parameters
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    $operands attr-dict `:` functional-type($operands, $results)
  }];
}

def DNN_BatchNormOp : DNN_StructuredOp<
    "batch_norm", "Batch normalization">;
def DNN_LayerNormOp : DNN_StructuredOp<
    "layer_norm", "Layer normalization">;
def DNN_ScaledDotProductAttentionOp : DNN_StructuredOp<
    "scaled_dot_product_attention", "Scaled dot-product attention">;
def DNN_MaxPool2dOp : DNN_StructuredOp<
    "max_pool2d", "Two-dimensional max pooling">;
def DNN_AdaptiveAvgPool2dOp : DNN_StructuredOp<
    "adaptive_avg_pool2d", "Two-dimensional adaptive average pooling">;
def DNN_AddOp : DNN_StructuredOp<
    "add", "Elementwise tensor addition">;
def DNN_MulOp : DNN_StructuredOp<
    "mul", "Elementwise tensor multiplication">;
def DNN_EmbeddingOp : DNN_StructuredOp<
    "embedding", "Embedding table lookup">;
def DNN_FlattenOp : DNN_StructuredOp<
    "flatten", "Contiguous-dimension flattening">;

// Each Torch activation has its own DNN operation. They share this structural
// base because Torch schemas carry different scalar, optional, and auxiliary
// tensor operands. Constant scalar parameters are stored in `parameters` and
// their original operand positions in `parameter_indices`.
class DNN_ActivationOp<string mnemonic, string activationSummary>
    : DNN_Op<mnemonic, [DNN_ParameterMetadata]> {
  let summary = activationSummary;
  let description = [{
    Preserves a Torch activation as a distinct high-level DNN operation.
    Tensor operands are bridged to builtin tensors, while constant scalar
    parameters are retained as attributes for later lowering.
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$operands,
    OptionalAttr<DenseI32ArrayAttr>:$parameter_indices,
    OptionalAttr<ArrayAttr>:$parameters
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    $operands attr-dict `:` functional-type($operands, $results)
  }];
}

def DNN_CeluOp : DNN_ActivationOp<"celu", "CELU activation">;
def DNN_EluOp : DNN_ActivationOp<"elu", "ELU activation">;
def DNN_GeluOp : DNN_ActivationOp<"gelu", "GELU activation">;
def DNN_GluOp : DNN_ActivationOp<"glu", "GLU activation">;
def DNN_HardshrinkOp : DNN_ActivationOp<"hardshrink", "Hardshrink activation">;
def DNN_HardsigmoidOp : DNN_ActivationOp<"hardsigmoid", "Hard sigmoid activation">;
def DNN_HardswishOp : DNN_ActivationOp<"hardswish", "Hard swish activation">;
def DNN_HardtanhOp : DNN_ActivationOp<"hardtanh", "Hardtanh activation">;
def DNN_LeakyReluOp : DNN_ActivationOp<"leaky_relu", "Leaky ReLU activation">;
def DNN_LogSigmoidOp : DNN_ActivationOp<"log_sigmoid", "Log-sigmoid activation">;
def DNN_MishOp : DNN_ActivationOp<"mish", "Mish activation">;
def DNN_PreluOp : DNN_ActivationOp<"prelu", "PReLU activation">;
def DNN_ReluOp : DNN_ActivationOp<"relu", "ReLU activation">;
def DNN_Relu6Op : DNN_ActivationOp<"relu6", "ReLU6 activation">;
def DNN_RreluOp : DNN_ActivationOp<"rrelu", "Randomized ReLU activation">;
def DNN_RreluWithNoiseFunctionalOp : DNN_ActivationOp<
    "rrelu_with_noise_functional", "Functional randomized ReLU activation">;
def DNN_SeluOp : DNN_ActivationOp<"selu", "SELU activation">;
def DNN_SigmoidOp : DNN_ActivationOp<"sigmoid", "Sigmoid activation">;
def DNN_SiluOp : DNN_ActivationOp<"silu", "SiLU activation">;
def DNN_SoftplusOp : DNN_ActivationOp<"softplus", "Softplus activation">;
def DNN_SoftshrinkOp : DNN_ActivationOp<"softshrink", "Softshrink activation">;
def DNN_TanhOp : DNN_ActivationOp<"tanh", "Tanh activation">;
def DNN_ThresholdOp : DNN_ActivationOp<"threshold", "Threshold activation">;
def DNN_SoftmaxOp : DNN_ActivationOp<"softmax", "Softmax activation">;
def DNN_SafeSoftmaxOp : DNN_ActivationOp<"safe_softmax", "Safe softmax activation">;
def DNN_SoftmaxIntOp : DNN_ActivationOp<"softmax_int", "Softmax integer-dimension overload">;
def DNN_LogSoftmaxOp : DNN_ActivationOp<"log_softmax", "Log-softmax activation">;
def DNN_LogSoftmaxIntOp : DNN_ActivationOp<
    "log_softmax_int", "Log-softmax integer-dimension overload">;
def DNN_SoftmaxBackwardDataOp : DNN_ActivationOp<
    "softmax_backward_data", "Softmax backward operation">;
def DNN_LogSoftmaxBackwardDataOp : DNN_ActivationOp<
    "log_softmax_backward_data", "Log-softmax backward operation">;
def DNN_EluBackwardOp : DNN_ActivationOp<"elu_backward", "ELU backward operation">;
def DNN_GeluBackwardOp : DNN_ActivationOp<"gelu_backward", "GELU backward operation">;
def DNN_HardtanhBackwardOp : DNN_ActivationOp<
    "hardtanh_backward", "Hardtanh backward operation">;
def DNN_LeakyReluBackwardOp : DNN_ActivationOp<
    "leaky_relu_backward", "Leaky ReLU backward operation">;
def DNN_RreluWithNoiseBackwardOp : DNN_ActivationOp<
    "rrelu_with_noise_backward", "Randomized ReLU backward operation">;
def DNN_SigmoidBackwardOp : DNN_ActivationOp<
    "sigmoid_backward", "Sigmoid backward operation">;
def DNN_SoftplusBackwardOp : DNN_ActivationOp<
    "softplus_backward", "Softplus backward operation">;
def DNN_TanhBackwardOp : DNN_ActivationOp<"tanh_backward", "Tanh backward operation">;
def DNN_ThresholdBackwardOp : DNN_ActivationOp<
    "threshold_backward", "Threshold backward operation">;

def DNN_ConvolutionOp : DNN_Op<"convolution", [DNN_ParameterMetadata]> {
  let summary = "Preserved neural-network convolution operation";
  let description = [{
    Preserves a selected Torch convolution-family operation as a high-level
    DNN operation. Its `kind` attribute identifies the original `aten.*`
    schema, and the variadic operands/results preserve convolution parameters
    including groups, stride, padding, dilation, and transposition flags.
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$operands,
    StrAttr:$kind,
    OptionalAttr<DenseI32ArrayAttr>:$parameter_indices,
    OptionalAttr<ArrayAttr>:$parameters
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    $operands attr-dict `:` functional-type($operands, $results)
  }];
}

// Recurrent operations use a flattened tensor operand list. The
// `operand_groups` attribute records the original grouping (for example,
// input, hidden/cell state, and parameter lists for aten.lstm.input), while
// scalar and optional arguments are retained in `parameters`.
def DNN_LstmOp : DNN_Op<"lstm", [DNN_ParameterMetadata]> {
  let summary = "Preserved high-level LSTM operation";
  let description = [{
    Preserves a Torch ATen LSTM as a first-class DNN operation. Tensor operands
    are bridged to builtin tensors; scalar and optional arguments remain in
    attributes for a later lowering.
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$operands,
    StrAttr:$kind,
    DenseI32ArrayAttr:$operand_groups,
    OptionalAttr<DenseI32ArrayAttr>:$parameter_indices,
    OptionalAttr<ArrayAttr>:$parameters
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    $operands attr-dict `:` functional-type($operands, $results)
  }];
}

def DNN_GruOp : DNN_Op<"gru", [DNN_ParameterMetadata]> {
  let summary = "Preserved high-level GRU operation";
  let description = [{
    Preserves a Torch ATen gated recurrent unit as a first-class DNN
    operation. Tensor operands are flattened and bridged to builtin tensors;
    scalar configuration is retained in attributes for later lowering.
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$operands,
    StrAttr:$kind,
    DenseI32ArrayAttr:$operand_groups,
    OptionalAttr<DenseI32ArrayAttr>:$parameter_indices,
    OptionalAttr<ArrayAttr>:$parameters
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    $operands attr-dict `:` functional-type($operands, $results)
  }];
}

def DNN_RnnOp : DNN_Op<"rnn", [DNN_ParameterMetadata]> {
  let summary = "Preserved high-level vanilla RNN operation";
  let description = [{
    Preserves a Torch ATen vanilla RNN as a first-class DNN operation. The
    `activation` parameter selects the recurrent nonlinearity (`tanh` or
    `relu`), while `kind` retains the original input or packed-data schema.
    Tensor-list operands are flattened and described by `operand_groups`.
  }];
  let arguments = (ins
    Variadic<AnyRankedTensor>:$operands,
    StrAttr:$activation,
    StrAttr:$kind,
    DenseI32ArrayAttr:$operand_groups,
    OptionalAttr<DenseI32ArrayAttr>:$parameter_indices,
    OptionalAttr<ArrayAttr>:$parameters
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    $operands attr-dict `:` functional-type($operands, $results)
  }];
  let hasVerifier = 1;
}

#endif // DNN_MLIR_DIALECT_DNN_IR_DNNOPS_TD
