#include "dnn-mlir/Conversion/TorchToDNN/Shape/ShapePatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {

void populateShapePatterns(RewritePatternSet &patterns,
                           ArrayRef<std::string> selectedOperations) {
#define DNN_REGISTER_SHAPE(Name, Operation, Capture) \
  populate##Name##Pattern(patterns, selectedOperations);
  DNN_SHAPE_PATTERN_LIST(DNN_REGISTER_SHAPE)
#undef DNN_REGISTER_SHAPE
}

} // namespace mlir::dnn::torch_to_dnn
