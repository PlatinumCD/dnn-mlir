#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

#include "dnn-mlir/Dialect/DNN/IR/DNNOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {
namespace {

class ConvertAtenLinearPattern
    : public OpRewritePattern<torch::Torch::AtenLinearOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      torch::Torch::AtenLinearOp op,
      PatternRewriter &rewriter) const override {
    auto inputType = getTensorBridgeType(op.getInput().getType());
    auto weightType = getTensorBridgeType(op.getWeight().getType());
    auto resultType = getTensorBridgeType(op.getResult().getType());
    if (!inputType || !weightType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "requires ranked tensors with known dtypes");

    Value bias = op.getBias();
    std::optional<TensorBridgeType> biasType;
    if (!isa<torch::Torch::NoneType>(bias.getType())) {
      biasType = getTensorBridgeType(bias.getType());
      if (!biasType)
        return rewriter.notifyMatchFailure(
            op, "requires a statically known tensor bias or none");
    }

    Location loc = op.getLoc();
    Value input =
        materializeBuiltinTensor(rewriter, loc, op.getInput(), *inputType);
    Value weight =
        materializeBuiltinTensor(rewriter, loc, op.getWeight(), *weightType);
    Value builtinBias;
    if (biasType)
      builtinBias = materializeBuiltinTensor(rewriter, loc, bias, *biasType);

    Value result = LinearOp::create(rewriter, loc, resultType->builtinType,
                                    input, weight, builtinBias);
    rewriter.replaceOp(
        op, materializeTorchTensor(rewriter, loc, result, *resultType));
    return success();
  }
};

} // namespace

void populateAffinePatterns(RewritePatternSet &patterns,
                            ArrayRef<std::string> selectedOperations) {
  if (isOperationSelected(selectedOperations, "aten.linear"))
    patterns.add<ConvertAtenLinearPattern>(patterns.getContext());
}

} // namespace mlir::dnn::torch_to_dnn
