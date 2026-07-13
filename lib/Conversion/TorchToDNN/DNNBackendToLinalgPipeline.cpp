#include "dnn-mlir/Conversion/TorchToDNN/Pipelines.h"

#include "dnn-mlir/Conversion/TorchToDNN/Passes.h"
#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Conversion/TorchConversionToMLProgram/TorchConversionToMLProgram.h"
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Conversion/TorchToTMTensor/TorchToTMTensor.h"
#include "torch-mlir/Conversion/TorchToTensor/TorchToTensor.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

using namespace mlir;

namespace mlir::dnn {
namespace {

struct PipelineOptions : public PassPipelineOptions<PipelineOptions> {
  Option<bool> decompose{
      *this, "decompose-complex-ops",
      llvm::cl::desc("Decompose complex operations."), llvm::cl::init(true)};
  Option<bool> shapeDtypeRefine{
      *this, "shape-dtype-refine",
      llvm::cl::desc("Do shape and dtype refinement."), llvm::cl::init(true)};
  Option<std::string> extraLibrary{
      *this, "extra-library",
      llvm::cl::desc("Filename of MLIR module for splicing into the abstract "
                     "interpretation library.")};
  Option<bool> allowNonFinites{
      *this, "allow-non-finites",
      llvm::cl::desc(
          "Allow lowering patterns that may produce non-finite values."),
      llvm::cl::init(true)};
  ListOption<std::string> queries{
      *this, "queries",
      llvm::cl::desc("Exact Torch operations to capture.")};
  ListOption<std::string> captures{
      *this, "captures",
      llvm::cl::desc("DNN result operations to capture.")};
};

SmallVector<std::string> getQueries(const PipelineOptions &options) {
  return {options.queries.begin(), options.queries.end()};
}

SmallVector<std::string> getCaptures(const PipelineOptions &options) {
  return {options.captures.begin(), options.captures.end()};
}

void addTorchScriptNormalization(OpPassManager &pm) {
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
}

class NormalizeTorchInputPass
    : public PassWrapper<NormalizeTorchInputPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeTorchInputPass)

  StringRef getArgument() const final { return "dnn-normalize-torch-input"; }
  StringRef getDescription() const final {
    return "Normalize TorchScript object graphs while accepting flat FX IR";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (module.getOps<torch::Torch::NnModuleOp>().empty())
      return;

    OpPassManager normalization(ModuleOp::getOperationName());
    addTorchScriptNormalization(normalization);
    if (failed(runPipeline(normalization, module)))
      signalPassFailure();
  }
};

void addTorchLinalgBackendLowering(OpPassManager &pm,
                                   bool allowNonFinites) {
  pm.addNestedPass<func::FuncOp>(
      torch::Torch::createRestructureNonConstantAxesPass());
  pm.addNestedPass<func::FuncOp>(
      torch::Torch::createFuseQuantizedOpsPass());
  pm.addNestedPass<func::FuncOp>(
      torch::createConvertTorchToTMTensorPass(allowNonFinites));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      torch::createConvertTorchToLinalgPass(allowNonFinites));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToSCFPass());
  pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToArithPass());
  pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToTensorPass());
  pm.addPass(torch::createConvertTorchConversionToMLProgramPass());
  pm.addNestedPass<func::FuncOp>(memref::createExpandOpsPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      memref::createResolveShapedTypeResultDimsPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addPass(torch::TorchConversion::createFuncBackendTypeConversionPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      torch::TorchConversion::createFinalizingBackendTypeConversionPass());
}

} // namespace

// Establish the value-semantics boundary before the post-functionalization
// DNN capture. ReduceOpVariants rewrites trailing-underscore ATen operations
// to functional equivalents plus explicit overwrites; MaximizeValueSemantics
// then resolves those overwrites through known aliases. Complex decomposition
// remains in the later backend pipeline so high-level operations stay visible.
void addTorchValueSemanticsNormalization(OpPassManager &pm,
                                         StringRef extraLibrary) {
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      torch::Torch::createRecomposeComplexOpsPass());
  pm.addNestedPass<func::FuncOp>(
      torch::Torch::createReduceOpVariantsPass(extraLibrary));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      torch::Torch::createMaximizeValueSemanticsPass());
  pm.addPass(torch::Torch::createRefinePublicReturnPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}

void registerDNNBackendToLinalgPipeline() {
  PassPipelineRegistration<PipelineOptions>(
      "dnn-backend-to-linalg-on-tensors-backend-pipeline",
      "Lower FX or TorchScript IR through DNN capture to the Linalg-on-tensors "
      "backend (without the upstream final verifier)",
      [](OpPassManager &pm, const PipelineOptions &options) {
        // Capture already-functional operations, normalize mutation and
        // aliases, then capture the newly functionalized forms. Leave complex
        // decomposition to the normal Torch backend pipeline below.
        SmallVector<std::string> queries = getQueries(options);
        SmallVector<std::string> captures = getCaptures(options);

        pm.addPass(std::make_unique<NormalizeTorchInputPass>());

        if (!queries.empty() || !captures.empty()) {
          // Capture value-semantic high-level and opaque operator forms before
          // ReduceOpVariants legalizes them. The shared conversion guard makes
          // this phase ignore every mutable/in-place operation.
          pm.addNestedPass<func::FuncOp>(
              createConvertTorchToDNNPass(queries, captures));
          pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());

          // Functionalize the skipped mutable operations, then give their
          // functional equivalents a second opportunity for DNN capture.
          addTorchValueSemanticsNormalization(pm, options.extraLibrary);
          pm.addNestedPass<func::FuncOp>(
              createConvertTorchToDNNPass(queries, captures));
          pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        }

        torch::Torch::TorchLoweringPipelineOptions torchOptions;
        torchOptions.decompose = options.decompose;
        torchOptions.shapeDtypeRefine = options.shapeDtypeRefine;
        torchOptions.extraLibrary = options.extraLibrary;
        torch::Torch::createTorchSimplificationPipeline(pm, torchOptions);
        addTorchLinalgBackendLowering(pm, options.allowNonFinites);
        pm.addPass(createVerifyDNNBackendContractPass());
      });
}

} // namespace mlir::dnn
