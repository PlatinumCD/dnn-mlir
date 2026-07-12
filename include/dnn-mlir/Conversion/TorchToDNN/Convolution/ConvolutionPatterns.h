#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_CONVOLUTION_CONVOLUTIONPATTERNS_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_CONVOLUTION_CONVOLUTIONPATTERNS_H

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::dnn::torch_to_dnn {

#define DNN_CONVOLUTION_PATTERN_LIST(X) \
  X(Conv1d, "aten.conv1d", "dnn.convolution") \
  X(Conv1dPadding, "aten.conv1d.padding", "dnn.convolution") \
  X(Conv2d, "aten.conv2d", "dnn.convolution") \
  X(Conv2dPadding, "aten.conv2d.padding", "dnn.convolution") \
  X(Conv3d, "aten.conv3d", "dnn.convolution") \
  X(Conv3dPadding, "aten.conv3d.padding", "dnn.convolution") \
  X(ConvTbc, "aten.conv_tbc", "dnn.convolution") \
  X(ConvTranspose1d, "aten.conv_transpose1d", "dnn.convolution") \
  X(ConvTranspose2dInput, "aten.conv_transpose2d.input", "dnn.convolution") \
  X(ConvTranspose3dInput, "aten.conv_transpose3d.input", "dnn.convolution") \
  X(Convolution, "aten.convolution", "dnn.convolution") \
  X(UnderscoreConvolution, "aten._convolution", "dnn.convolution") \
  X(UnderscoreConvolutionDeprecated, "aten._convolution.deprecated", \
    "dnn.convolution") \
  X(ConvolutionBackward, "aten.convolution_backward", "dnn.convolution")

#define DNN_DECLARE_CONVOLUTION_PATTERN(Name, Operation, Capture) \
  void populate##Name##Pattern(RewritePatternSet &, llvm::ArrayRef<std::string>);
DNN_CONVOLUTION_PATTERN_LIST(DNN_DECLARE_CONVOLUTION_PATTERN)
#undef DNN_DECLARE_CONVOLUTION_PATTERN

void populateNamedConvolutionPattern(RewritePatternSet &patterns,
                                     llvm::ArrayRef<std::string> selectedOperations,
                                     llvm::StringRef operationName);

} // namespace mlir::dnn::torch_to_dnn

#endif // DNN_MLIR_CONVERSION_TORCHTODNN_CONVOLUTION_CONVOLUTIONPATTERNS_H
