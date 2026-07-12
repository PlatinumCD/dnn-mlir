#include "dnn-mlir/Conversion/TorchToDNN/Convolution/ConvolutionPatterns.h"

#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {

void populateConvolutionPatterns(RewritePatternSet &patterns,
                                 ArrayRef<std::string> selectedOperations) {
#define DNN_REGISTER_CONVOLUTION(Name, Operation, Capture) \
  populate##Name##Pattern(patterns, selectedOperations);
  DNN_CONVOLUTION_PATTERN_LIST(DNN_REGISTER_CONVOLUTION)
#undef DNN_REGISTER_CONVOLUTION
}

} // namespace mlir::dnn::torch_to_dnn
