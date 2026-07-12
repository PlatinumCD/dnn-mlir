#include "dnn-mlir/Conversion/TorchToDNN/Convolution/ConvolutionPatterns.h"

namespace mlir::dnn::torch_to_dnn {

void populateUnderscoreConvolutionDeprecatedPattern(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations) {
  populateNamedConvolutionPattern(patterns, selectedOperations, "aten._convolution.deprecated");
}

} // namespace mlir::dnn::torch_to_dnn

