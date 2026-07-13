#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_ACTIVATION_ACTIVATIONPATTERNS_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_ACTIVATION_ACTIVATIONPATTERNS_H

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::dnn::torch_to_dnn {

#define DNN_ACTIVATION_PATTERN_LIST(X) \
  X(Celu, "aten.celu", "dnn.celu") \
  X(Elu, "aten.elu", "dnn.elu") \
  X(Gelu, "aten.gelu", "dnn.gelu") \
  X(Glu, "aten.glu", "dnn.glu") \
  X(Hardshrink, "aten.hardshrink", "dnn.hardshrink") \
  X(Hardsigmoid, "aten.hardsigmoid", "dnn.hardsigmoid") \
  X(Hardswish, "aten.hardswish", "dnn.hardswish") \
  X(Hardtanh, "aten.hardtanh", "dnn.hardtanh") \
  X(LeakyRelu, "aten.leaky_relu", "dnn.leaky_relu") \
  X(LogSigmoid, "aten.log_sigmoid", "dnn.log_sigmoid") \
  X(Mish, "aten.mish", "dnn.mish") \
  X(Prelu, "aten.prelu", "dnn.prelu") \
  X(Relu, "aten.relu", "dnn.relu") \
  X(Relu6, "aten.relu6", "dnn.relu6") \
  X(Rrelu, "aten.rrelu", "dnn.rrelu") \
  X(RreluWithNoiseFunctional, "aten.rrelu_with_noise_functional", \
    "dnn.rrelu_with_noise_functional") \
  X(Selu, "aten.selu", "dnn.selu") \
  X(Sigmoid, "aten.sigmoid", "dnn.sigmoid") \
  X(Silu, "aten.silu", "dnn.silu") \
  X(Softplus, "aten.softplus", "dnn.softplus") \
  X(Softshrink, "aten.softshrink", "dnn.softshrink") \
  X(Tanh, "aten.tanh", "dnn.tanh") \
  X(Threshold, "aten.threshold", "dnn.threshold") \
  X(Softmax, "aten._softmax", "dnn.softmax") \
  X(SafeSoftmax, "aten._safe_softmax", "dnn.safe_softmax") \
  X(SoftmaxInt, "aten.softmax.int", "dnn.softmax_int") \
  X(LogSoftmax, "aten._log_softmax", "dnn.log_softmax") \
  X(LogSoftmaxInt, "aten.log_softmax.int", "dnn.log_softmax_int") \
  X(SoftmaxBackwardData, "aten._softmax_backward_data", \
    "dnn.softmax_backward_data") \
  X(LogSoftmaxBackwardData, "aten._log_softmax_backward_data", \
    "dnn.log_softmax_backward_data") \
  X(EluBackward, "aten.elu_backward", "dnn.elu_backward") \
  X(GeluBackward, "aten.gelu_backward", "dnn.gelu_backward") \
  X(HardtanhBackward, "aten.hardtanh_backward", "dnn.hardtanh_backward") \
  X(LeakyReluBackward, "aten.leaky_relu_backward", \
    "dnn.leaky_relu_backward") \
  X(RreluWithNoiseBackward, "aten.rrelu_with_noise_backward", \
    "dnn.rrelu_with_noise_backward") \
  X(SigmoidBackward, "aten.sigmoid_backward", "dnn.sigmoid_backward") \
  X(SoftplusBackward, "aten.softplus_backward", "dnn.softplus_backward") \
  X(TanhBackward, "aten.tanh_backward", "dnn.tanh_backward") \
  X(ThresholdBackward, "aten.threshold_backward", "dnn.threshold_backward")

#define DNN_DECLARE_ACTIVATION_PATTERN(Name, Operation, Target) \
  void populate##Name##Pattern(RewritePatternSet &, llvm::ArrayRef<std::string>);
DNN_ACTIVATION_PATTERN_LIST(DNN_DECLARE_ACTIVATION_PATTERN)
#undef DNN_DECLARE_ACTIVATION_PATTERN

void populateNamedActivationPattern(RewritePatternSet &patterns,
                                    llvm::ArrayRef<std::string> selectedOperations,
                                    llvm::StringRef operationName);
void populateDecomposedGeluFusionPatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);

} // namespace mlir::dnn::torch_to_dnn

#endif // DNN_MLIR_CONVERSION_TORCHTODNN_ACTIVATION_ACTIVATIONPATTERNS_H
