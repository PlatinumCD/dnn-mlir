#include "dnn-mlir/Conversion/TorchToDNN/Normalization/NormalizationPatterns.h"

#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include <limits>
#include <utility>

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {
namespace {

namespace Torch = mlir::torch::Torch;

std::optional<int64_t> getInteger(Value value) {
  auto attribute = getConstantTorchAttribute(value);
  if (!attribute)
    return std::nullopt;
  if (auto integer = dyn_cast<IntegerAttr>(*attribute))
    return integer.getInt();
  return std::nullopt;
}

std::optional<double> getFloating(Value value) {
  auto attribute = getConstantTorchAttribute(value);
  if (!attribute)
    return std::nullopt;
  if (auto floating = dyn_cast<FloatAttr>(*attribute))
    return floating.getValueAsDouble();
  if (auto integer = dyn_cast<IntegerAttr>(*attribute))
    return static_cast<double>(integer.getInt());
  return std::nullopt;
}

bool matchesInteger(Value value, int64_t expected) {
  auto actual = getInteger(value);
  return actual && *actual == expected;
}

bool matchesTrue(Value value) {
  auto attribute = getConstantTorchAttribute(value);
  auto boolean = attribute ? dyn_cast<BoolAttr>(*attribute) : BoolAttr();
  return boolean && boolean.getValue();
}

bool matchesNone(Value value) {
  auto attribute = getConstantTorchAttribute(value);
  return attribute && isa<UnitAttr>(*attribute);
}

bool matchesAlphaOne(Operation *op, unsigned operandIndex) {
  return op->getNumOperands() > operandIndex &&
         matchesInteger(op->getOperand(operandIndex), 1);
}

std::optional<ArrayAttr> getIntegerList(Value value) {
  auto attribute = getConstantTorchAttribute(value);
  if (!attribute)
    return std::nullopt;
  auto list = dyn_cast<ArrayAttr>(*attribute);
  if (!list)
    return std::nullopt;
  for (Attribute element : list)
    if (!isa<IntegerAttr>(element))
      return std::nullopt;
  return list;
}

bool matchesShape(ArrayAttr shape, ArrayRef<int64_t> expected) {
  if (shape.size() != expected.size())
    return false;
  for (auto [attribute, dimension] : llvm::zip(shape, expected))
    if (cast<IntegerAttr>(attribute).getInt() != dimension)
      return false;
  return true;
}

bool matchesTensorShape(Value value, ArrayRef<int64_t> expected) {
  auto bridge = getTensorBridgeType(value.getType());
  return bridge && bridge->builtinType.getShape() == expected;
}

void eraseIfUnused(PatternRewriter &rewriter, Operation *op) {
  if (op && op->use_empty())
    rewriter.eraseOp(op);
}

class FuseDecomposedLayerNormPattern
    : public OpRewritePattern<Torch::AtenAddTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      Torch::AtenAddTensorOp root,
      PatternRewriter &rewriter) const override {
    if (!matchesAlphaOne(root, 2))
      return failure();

    Torch::AtenMulTensorOp weighted;
    Value bias;
    for (auto [candidateWeighted, candidateBias] :
         {std::pair{root->getOperand(0), root->getOperand(1)},
          std::pair{root->getOperand(1), root->getOperand(0)}}) {
      weighted = candidateWeighted.getDefiningOp<Torch::AtenMulTensorOp>();
      if (weighted) {
        bias = candidateBias;
        break;
      }
    }
    if (!weighted)
      return failure();

    Torch::AtenMulTensorOp normalized;
    Value weight;
    for (auto [candidateNormalized, candidateWeight] :
         {std::pair{weighted->getOperand(0), weighted->getOperand(1)},
          std::pair{weighted->getOperand(1), weighted->getOperand(0)}}) {
      normalized =
          candidateNormalized.getDefiningOp<Torch::AtenMulTensorOp>();
      if (normalized) {
        weight = candidateWeight;
        break;
      }
    }
    if (!normalized)
      return failure();

    Torch::AtenSubTensorOp centered;
    Torch::AtenBroadcastToOp inverseBroadcast;
    for (auto [candidateCentered, candidateInverse] :
         {std::pair{normalized->getOperand(0), normalized->getOperand(1)},
          std::pair{normalized->getOperand(1), normalized->getOperand(0)}}) {
      centered = candidateCentered.getDefiningOp<Torch::AtenSubTensorOp>();
      inverseBroadcast =
          candidateInverse.getDefiningOp<Torch::AtenBroadcastToOp>();
      if (centered && inverseBroadcast)
        break;
    }
    if (!centered || !inverseBroadcast || !matchesAlphaOne(centered, 2))
      return failure();

    Value input = centered->getOperand(0);
    auto meanBroadcast =
        centered->getOperand(1).getDefiningOp<Torch::AtenBroadcastToOp>();
    auto meanDiv = meanBroadcast
                       ? meanBroadcast->getOperand(0)
                             .getDefiningOp<Torch::AtenDivScalarOp>()
                       : Torch::AtenDivScalarOp();
    auto meanSum = meanDiv ? meanDiv->getOperand(0)
                                 .getDefiningOp<Torch::AtenSumDimIntListOp>()
                           : Torch::AtenSumDimIntListOp();
    if (!meanSum || meanSum->getOperand(0) != input ||
        !matchesTrue(meanSum->getOperand(2)) ||
        !matchesNone(meanSum->getOperand(3)))
      return failure();

    auto inverse = inverseBroadcast->getOperand(0)
                       .getDefiningOp<Torch::AtenRsqrtOp>();
    auto epsilonAdd = inverse ? inverse->getOperand(0)
                                    .getDefiningOp<Torch::AtenAddScalarOp>()
                              : Torch::AtenAddScalarOp();
    auto varianceDiv = epsilonAdd
                           ? epsilonAdd->getOperand(0)
                                 .getDefiningOp<Torch::AtenDivScalarOp>()
                           : Torch::AtenDivScalarOp();
    auto varianceSum = varianceDiv
                           ? varianceDiv->getOperand(0)
                                 .getDefiningOp<Torch::AtenSumDimIntListOp>()
                           : Torch::AtenSumDimIntListOp();
    auto square = varianceSum
                      ? varianceSum->getOperand(0)
                            .getDefiningOp<Torch::AtenMulTensorOp>()
                      : Torch::AtenMulTensorOp();
    if (!square || square->getOperand(0) != centered->getResult(0) ||
        square->getOperand(1) != centered->getResult(0) ||
        !matchesTrue(varianceSum->getOperand(2)) ||
        !matchesNone(varianceSum->getOperand(3)) ||
        !matchesAlphaOne(epsilonAdd, 2))
      return failure();

    auto reductionDims = getIntegerList(meanSum->getOperand(1));
    auto varianceDims = getIntegerList(varianceSum->getOperand(1));
    auto meanShape = getIntegerList(meanBroadcast->getOperand(1));
    auto inverseShape = getIntegerList(inverseBroadcast->getOperand(1));
    auto inputBridge = getTensorBridgeType(input.getType());
    if (!reductionDims || !varianceDims || *reductionDims != *varianceDims ||
        !meanShape || !inverseShape || *meanShape != *inverseShape ||
        !inputBridge ||
        !matchesShape(*meanShape, inputBridge->builtinType.getShape()))
      return failure();

    ArrayRef<int64_t> inputShape = inputBridge->builtinType.getShape();
    SmallVector<int64_t> normalizedShape;
    normalizedShape.reserve(reductionDims->size());
    int64_t firstAxis = static_cast<int64_t>(inputShape.size()) -
                        static_cast<int64_t>(reductionDims->size());
    if (firstAxis < 0)
      return failure();
    int64_t elementCount = 1;
    for (auto [index, attribute] : llvm::enumerate(*reductionDims)) {
      int64_t axis = cast<IntegerAttr>(attribute).getInt();
      if (axis < 0)
        axis += inputShape.size();
      if (axis != firstAxis + static_cast<int64_t>(index))
        return failure();
      int64_t dimension = inputShape[axis];
      if (ShapedType::isDynamic(dimension) || dimension <= 0 ||
          elementCount > std::numeric_limits<int64_t>::max() / dimension)
        return failure();
      normalizedShape.push_back(dimension);
      elementCount *= dimension;
    }

    auto meanDivisor = getFloating(meanDiv->getOperand(1));
    auto varianceDivisor = getFloating(varianceDiv->getOperand(1));
    auto epsilon = getFloating(epsilonAdd->getOperand(1));
    if (!meanDivisor || !varianceDivisor || !epsilon || *epsilon <= 0.0 ||
        *meanDivisor != static_cast<double>(elementCount) ||
        *varianceDivisor != static_cast<double>(elementCount) ||
        !matchesTensorShape(weight, normalizedShape) ||
        !matchesTensorShape(bias, normalizedShape))
      return failure();

    SmallVector<Value> normalizedShapeValues;
    normalizedShapeValues.reserve(normalizedShape.size());
    for (int64_t dimension : normalizedShape)
      normalizedShapeValues.push_back(Torch::ConstantIntOp::create(
          rewriter, root.getLoc(), rewriter.getI64IntegerAttr(dimension)));
    Value normalizedShapeList = Torch::PrimListConstructOp::create(
        rewriter, root.getLoc(),
        Torch::ListType::get(Torch::IntType::get(getContext())),
        normalizedShapeValues);
    Value cudnnEnable =
        Torch::ConstantBoolOp::create(rewriter, root.getLoc(), false);
    Value layerNorm = Torch::AtenLayerNormOp::create(
        rewriter, root.getLoc(), root->getResult(0).getType(), input,
        normalizedShapeList, weight, bias, epsilonAdd->getOperand(1),
        cudnnEnable);
    rewriter.replaceOp(root, layerNorm);

    eraseIfUnused(rewriter, weighted);
    eraseIfUnused(rewriter, normalized);
    eraseIfUnused(rewriter, inverseBroadcast);
    eraseIfUnused(rewriter, inverse);
    eraseIfUnused(rewriter, epsilonAdd);
    eraseIfUnused(rewriter, varianceDiv);
    eraseIfUnused(rewriter, varianceSum);
    eraseIfUnused(rewriter, square);
    eraseIfUnused(rewriter, centered);
    eraseIfUnused(rewriter, meanBroadcast);
    eraseIfUnused(rewriter, meanDiv);
    eraseIfUnused(rewriter, meanSum);
    return success();
  }
};

} // namespace

void populateDecomposedLayerNormFusionPatterns(
    RewritePatternSet &patterns,
    ArrayRef<std::string> selectedOperations) {
  if (isOperationSelected(selectedOperations, "aten.layer_norm"))
    patterns.add<FuseDecomposedLayerNormPattern>(patterns.getContext());
}

} // namespace mlir::dnn::torch_to_dnn
