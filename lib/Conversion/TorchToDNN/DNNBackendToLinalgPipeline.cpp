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

void registerDNNBackendToLinalgPipeline() {
  PassPipelineRegistration<PipelineOptions>(
      "dnn-backend-to-linalg-on-tensors-backend-pipeline",
      "Lower TorchScript through DNN capture to the Linalg-on-tensors "
      "backend (without the upstream final verifier)",
      [](OpPassManager &pm, const PipelineOptions &options) {
        // Capture before Torch reduces or legalizes generic torch.operator
        // forms. The remaining Torch operations still run through Torch-MLIR's
        // simplification, refinement, decomposition, and backend conversions.
        SmallVector<std::string> queries = getQueries(options);
        SmallVector<std::string> captures = getCaptures(options);

        addTorchScriptNormalization(pm);

        if (!queries.empty() || !captures.empty()) {
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
      });
}

} // namespace mlir::dnn
