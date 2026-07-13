#include "dnn-mlir/Conversion/TorchToDNN/Matrix/MatrixPatterns.h"
#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

#include "dnn-mlir/Dialect/DNN/IR/DNNOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {
namespace {

class ConvertAtenMmPattern : public OpRewritePattern<torch::Torch::AtenMmOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      torch::Torch::AtenMmOp op,
      PatternRewriter &rewriter) const override {
    auto lhsType = getTensorBridgeType(op.getSelf().getType());
    auto rhsType = getTensorBridgeType(op.getMat2().getType());
    auto resultType = getTensorBridgeType(op.getResult().getType());
    if (!lhsType || !rhsType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "requires ranked tensors with known dtypes");
    if (lhsType->builtinType.getRank() != 2 ||
        rhsType->builtinType.getRank() != 2 ||
        resultType->builtinType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "requires rank-two tensors");

    Location loc = op.getLoc();
    Value lhs =
        materializeBuiltinTensor(rewriter, loc, op.getSelf(), *lhsType);
    Value rhs =
        materializeBuiltinTensor(rewriter, loc, op.getMat2(), *rhsType);
    Value result =
        MmOp::create(rewriter, loc, resultType->builtinType, lhs, rhs);
    rewriter.replaceOp(
        op, materializeTorchTensor(rewriter, loc, result, *resultType));
    return success();
  }
};

} // namespace

void populateAtenMmPattern(RewritePatternSet &patterns,
                           ArrayRef<std::string> selectedOperations) {
  if (isOperationSelected(selectedOperations, "aten.mm"))
    patterns.add<ConvertAtenMmPattern>(patterns.getContext());
}

} // namespace mlir::dnn::torch_to_dnn
