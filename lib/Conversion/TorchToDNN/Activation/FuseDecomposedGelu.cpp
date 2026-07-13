#include "dnn-mlir/Conversion/TorchToDNN/Activation/ActivationPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include <cmath>
#include <utility>

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {
namespace {

namespace Torch = mlir::torch::Torch;

bool matchesInteger(Value value, int64_t expected) {
  auto attribute = getConstantTorchAttribute(value);
  auto integer = attribute ? dyn_cast<IntegerAttr>(*attribute) : IntegerAttr();
  return integer && integer.getInt() == expected;
}

bool matchesFloat(Value value, double expected, double tolerance = 1.0e-7) {
  auto attribute = getConstantTorchAttribute(value);
  if (!attribute)
    return false;
  if (auto floating = dyn_cast<FloatAttr>(*attribute))
    return std::abs(floating.getValueAsDouble() - expected) <= tolerance;
  if (auto integer = dyn_cast<IntegerAttr>(*attribute))
    return std::abs(static_cast<double>(integer.getInt()) - expected) <=
           tolerance;
  return false;
}

bool matchesAlphaOne(Operation *op, unsigned operandIndex) {
  return op->getNumOperands() > operandIndex &&
         matchesInteger(op->getOperand(operandIndex), 1);
}

void eraseIfUnused(PatternRewriter &rewriter, Operation *op) {
  if (op && op->use_empty())
    rewriter.eraseOp(op);
}

class FuseDecomposedGeluPattern
    : public OpRewritePattern<Torch::AtenMulTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      Torch::AtenMulTensorOp root,
      PatternRewriter &rewriter) const override {
    Torch::AtenMulScalarOp half;
    Torch::AtenAddScalarOp plusOne;
    for (auto [candidateHalf, candidatePlusOne] :
         {std::pair{root->getOperand(0), root->getOperand(1)},
          std::pair{root->getOperand(1), root->getOperand(0)}}) {
      half = candidateHalf.getDefiningOp<Torch::AtenMulScalarOp>();
      plusOne = candidatePlusOne.getDefiningOp<Torch::AtenAddScalarOp>();
      if (half && plusOne)
        break;
    }
    if (!half || !plusOne ||
        !matchesFloat(half->getOperand(1), 0.5) ||
        !matchesFloat(plusOne->getOperand(1), 1.0) ||
        !matchesAlphaOne(plusOne, 2))
      return failure();

    Value input = half->getOperand(0);
    auto tanh = plusOne->getOperand(0).getDefiningOp<Torch::AtenTanhOp>();
    if (!tanh)
      return failure();
    auto scaled =
        tanh->getOperand(0).getDefiningOp<Torch::AtenMulScalarOp>();
    if (!scaled ||
        !matchesFloat(scaled->getOperand(1), 0.7978845608028654))
      return failure();
    auto inner =
        scaled->getOperand(0).getDefiningOp<Torch::AtenAddTensorOp>();
    if (!inner || !matchesAlphaOne(inner, 2))
      return failure();

    Value cubicValue;
    if (inner->getOperand(0) == input)
      cubicValue = inner->getOperand(1);
    else if (inner->getOperand(1) == input)
      cubicValue = inner->getOperand(0);
    else
      return failure();

    auto cubicScale =
        cubicValue.getDefiningOp<Torch::AtenMulScalarOp>();
    if (!cubicScale ||
        !matchesFloat(cubicScale->getOperand(1), 0.044715))
      return failure();
    auto cube = cubicScale->getOperand(0)
                    .getDefiningOp<Torch::AtenPowTensorScalarOp>();
    if (!cube || cube->getOperand(0) != input ||
        !matchesInteger(cube->getOperand(1), 3))
      return failure();

    Value approximate = Torch::ConstantStrOp::create(
        rewriter, root.getLoc(), Torch::StringType::get(getContext()),
        rewriter.getStringAttr("tanh"));
    Value gelu = Torch::AtenGeluOp::create(
        rewriter, root.getLoc(), root->getResult(0).getType(), input,
        approximate);
    rewriter.replaceOp(root, gelu);

    eraseIfUnused(rewriter, half);
    eraseIfUnused(rewriter, plusOne);
    eraseIfUnused(rewriter, tanh);
    eraseIfUnused(rewriter, scaled);
    eraseIfUnused(rewriter, inner);
    eraseIfUnused(rewriter, cubicScale);
    eraseIfUnused(rewriter, cube);
    return success();
  }
};

} // namespace

void populateDecomposedGeluFusionPatterns(
    RewritePatternSet &patterns,
    ArrayRef<std::string> selectedOperations) {
  if (isOperationSelected(selectedOperations, "aten.gelu"))
    patterns.add<FuseDecomposedGeluPattern>(patterns.getContext());
}

} // namespace mlir::dnn::torch_to_dnn
