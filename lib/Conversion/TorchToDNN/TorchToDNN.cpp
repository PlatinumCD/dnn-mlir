#include "dnn-mlir/Conversion/TorchToDNN/Passes.h"
#include "dnn-mlir/Conversion/TorchToDNN/Pipelines.h"
#include "dnn-mlir/Conversion/TorchToDNN/CaptureRegistry.h"
#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"

#include "dnn-mlir/Dialect/DNN/IR/DNNDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

using namespace mlir;

namespace mlir::dnn {
namespace {

namespace Torch = mlir::torch::Torch;
namespace TorchConversion = mlir::torch::TorchConversion;

} // namespace

namespace torch_to_dnn {

std::optional<TensorBridgeType> getTensorBridgeType(Type type) {
  auto tensorType = dyn_cast<Torch::BaseTensorType>(type);
  if (!tensorType || !tensorType.hasSizes() || !tensorType.hasDtype())
    return std::nullopt;

  Torch::ValueTensorType valueTensorType =
      tensorType.getWithValueSemantics();
  auto builtinType =
      dyn_cast_or_null<RankedTensorType>(valueTensorType.toBuiltinTensor());
  if (!builtinType)
    return std::nullopt;
  // Torch integer dtypes are represented with signedness, while builtin and
  // Linalg backend types use signless integers. Normalize at the DNN boundary
  // so final backend type conversion does not leave a live signedness cast on
  // captured integer tensors such as packed-sequence batch_sizes.
  if (auto integerType = dyn_cast<IntegerType>(builtinType.getElementType())) {
    if (!integerType.isSignless())
      builtinType = RankedTensorType::get(
          builtinType.getShape(),
          IntegerType::get(type.getContext(), integerType.getWidth()),
          builtinType.getEncoding());
  }
  return TensorBridgeType{type, valueTensorType, builtinType};
}

std::string canonicalizeOperationName(StringRef operation) {
  operation = operation.trim();
  if (operation.starts_with("torch."))
    operation = operation.drop_front(6);
  return operation.str();
}

bool isOperationSelected(ArrayRef<std::string> selectedOperations,
                         StringRef operation) {
  if (selectedOperations.empty())
    return true;

  for (const std::string &selected : selectedOperations)
    if (canonicalizeOperationName(selected) == operation)
      return true;
  return false;
}

std::optional<Attribute> getConstantTorchAttribute(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return std::nullopt;

  StringRef operationName = definingOp->getName().getStringRef();
  if (operationName == "torch.constant.none")
    return UnitAttr::get(definingOp->getContext());
  if (operationName == "torch.constant.int" ||
      operationName == "torch.constant.float" ||
      operationName == "torch.constant.bool" ||
      operationName == "torch.constant.str")
    return definingOp->getAttr("value");

  if (operationName != "torch.prim.ListConstruct")
    return std::nullopt;

  SmallVector<Attribute> elements;
  elements.reserve(definingOp->getNumOperands());
  for (Value element : definingOp->getOperands()) {
    auto attribute = getConstantTorchAttribute(element);
    if (!attribute)
      return std::nullopt;
    elements.push_back(*attribute);
  }
  return ArrayAttr::get(definingOp->getContext(), elements);
}

Value materializeBuiltinTensor(PatternRewriter &rewriter, Location loc,
                               Value value,
                               const TensorBridgeType &bridgeType) {
  if (isa<Torch::NonValueTensorType>(value.getType()))
    value = Torch::CopyToValueTensorOp::create(rewriter, loc, value);
  return TorchConversion::ToBuiltinTensorOp::create(
      rewriter, loc, bridgeType.builtinType, value);
}

Value materializeTorchTensor(PatternRewriter &rewriter, Location loc,
                             Value value,
                             const TensorBridgeType &bridgeType) {
  value = TorchConversion::FromBuiltinTensorOp::create(
      rewriter, loc, bridgeType.valueTensorType, value);
  if (isa<Torch::NonValueTensorType>(bridgeType.originalType))
    value = Torch::CopyToNonValueTensorOp::create(rewriter, loc, value);
  return value;
}

} // namespace torch_to_dnn

namespace {
class ConvertTorchToDNNPass
    : public PassWrapper<ConvertTorchToDNNPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTorchToDNNPass)

  ConvertTorchToDNNPass() = default;
  ConvertTorchToDNNPass(ArrayRef<std::string> queries,
                        ArrayRef<std::string> captures)
      : queries(queries.begin(), queries.end()),
        captures(captures.begin(), captures.end()) {}

  StringRef getArgument() const final { return "convert-torch-to-dnn"; }
  StringRef getDescription() const final {
    return "Recognize selected high-level Torch operations as DNN operations";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<DNNDialect, Torch::TorchDialect,
                    TorchConversion::TorchConversionDialect>();
  }

  void runOnOperation() final {
    if (failed(torch_to_dnn::validateCaptureSelection(
            getOperation(), queries, captures))) {
      signalPassFailure();
      return;
    }

    SmallVector<std::string> selectedOperations =
        torch_to_dnn::resolveCaptureQueries(queries, captures);

    RewritePatternSet patterns(&getContext());
    torch_to_dnn::populateMatrixPatterns(patterns, selectedOperations);
    torch_to_dnn::populateAffinePatterns(patterns, selectedOperations);
    torch_to_dnn::populateActivationPatterns(patterns, selectedOperations);
    torch_to_dnn::populateConvolutionPatterns(patterns, selectedOperations);
    torch_to_dnn::populateRecurrentPatterns(patterns, selectedOperations);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

private:
  SmallVector<std::string> queries;
  SmallVector<std::string> captures;
};

} // namespace

std::unique_ptr<Pass> createConvertTorchToDNNPass() {
  return std::make_unique<ConvertTorchToDNNPass>();
}

std::unique_ptr<Pass>
createConvertTorchToDNNPass(ArrayRef<std::string> queries,
                            ArrayRef<std::string> captures) {
  return std::make_unique<ConvertTorchToDNNPass>(queries, captures);
}

void registerTorchToDNNPasses() {
  PassRegistration<ConvertTorchToDNNPass>();

  struct TorchScriptPipelineOptions
      : public PassPipelineOptions<TorchScriptPipelineOptions> {
    ListOption<std::string> queries{
        *this, "queries",
        llvm::cl::desc("Exact Torch operations to capture.")};
    ListOption<std::string> captures{
        *this, "captures",
        llvm::cl::desc("DNN result operations to capture.")};
  };
  PassPipelineRegistration<TorchScriptPipelineOptions>(
      "torchscript-to-dnn-pipeline",
      "Prepare raw TorchScript IR and recognize high-level DNN operations",
      [](OpPassManager &pm, const TorchScriptPipelineOptions &options) {
        pm.addPass(createSymbolDCEPass());
        pm.addPass(torch::Torch::createPrepareForGlobalizeObjectGraphPass());
        pm.addPass(torch::Torch::createGlobalizeObjectGraphPass());
        pm.addPass(createSymbolDCEPass());
        pm.addPass(createInlinerPass());
        pm.addPass(torch::Torch::createAdjustCallingConventionsPass());
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        pm.addPass(torch::Torch::createInlineGlobalSlotsPass());
        pm.addPass(torch::Torch::createEraseModuleInitializerPass());
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(
            torch::Torch::createRecomposeComplexOpsPass());
        if (!options.queries.empty() || !options.captures.empty())
          pm.addNestedPass<func::FuncOp>(createConvertTorchToDNNPass(
              options.queries, options.captures));
      });

  registerDNNBackendToLinalgPipeline();
}

} // namespace mlir::dnn
