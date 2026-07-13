#include "dnn-mlir/Conversion/TorchToDNN/Matrix/MatrixPatterns.h"
#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

#include "dnn-mlir/Dialect/DNN/IR/DNNOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {
namespace {

class ConvertAtenMatmulPattern
    : public OpRewritePattern<torch::Torch::AtenMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      torch::Torch::AtenMatmulOp op,
      PatternRewriter &rewriter) const override {
    auto lhsType = getTensorBridgeType(op.getSelf().getType());
    auto rhsType = getTensorBridgeType(op.getOther().getType());
    auto resultType = getTensorBridgeType(op.getResult().getType());
    if (!lhsType || !rhsType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "requires ranked tensors with known dtypes");
    if (lhsType->builtinType.getRank() < 1 ||
        rhsType->builtinType.getRank() < 1)
      return rewriter.notifyMatchFailure(
          op, "requires operand ranks of at least one");

    Location loc = op.getLoc();
    Value lhs =
        materializeBuiltinTensor(rewriter, loc, op.getSelf(), *lhsType);
    Value rhs =
        materializeBuiltinTensor(rewriter, loc, op.getOther(), *rhsType);
    Value result = MatmulOp::create(rewriter, loc, resultType->builtinType,
                                    lhs, rhs);
    rewriter.replaceOp(
        op, materializeTorchTensor(rewriter, loc, result, *resultType));
    return success();
  }
};

} // namespace

void populateAtenMatmulPattern(
    RewritePatternSet &patterns,
    ArrayRef<std::string> selectedOperations) {
  if (isOperationSelected(selectedOperations, "aten.matmul"))
    patterns.add<ConvertAtenMatmulPattern>(patterns.getContext());
}

} // namespace mlir::dnn::torch_to_dnn
