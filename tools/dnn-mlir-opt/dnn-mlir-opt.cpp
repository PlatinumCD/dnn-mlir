#include "dnn-mlir/Conversion/TorchToDNN/Passes.h"
#include "dnn-mlir/Conversion/TorchToDNN/CaptureRegistry.h"
#include "dnn-mlir/Dialect/DNN/IR/DNNDialect.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorDialect.h"

using namespace mlir;

namespace {

llvm::cl::opt<bool> listAvailableQueries(
    "list-available-queries",
    llvm::cl::desc("List DNN captures and their Torch queries by family"),
    llvm::cl::init(false));

llvm::cl::alias listAvailableOps(
    "list-available-ops",
    llvm::cl::desc("Alias for --list-available-queries"),
    llvm::cl::aliasopt(listAvailableQueries));

void printAvailableQueries(llvm::raw_ostream &os) {
  using namespace mlir::dnn::torch_to_dnn;
  os << "Available DNN captures and Torch queries by section:\n";
  llvm::StringRef previousSection;
  llvm::StringRef previousCapture;
  for (const CaptureRegistration &registration : getCaptureRegistry()) {
    if (registration.section != previousSection) {
      os << "\n" << registration.section << ":\n";
      previousSection = registration.section;
      previousCapture = {};
    }
    if (registration.capture != previousCapture) {
      os << "  Capture: " << registration.capture << "\n"
         << "  Queries:\n";
      previousCapture = registration.capture;
    }
    os << "    " << registration.query << "\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  torch::registerTorchPasses();
  dnn::registerTorchToDNNPasses();

  registerCanonicalizerPass();
  registerCSEPass();
  registerInlinerPass();
  registerSymbolDCEPass();

  DialectRegistry registry;
  registry.insert<arith::ArithDialect, dnn::DNNDialect, func::FuncDialect,
                  linalg::LinalgDialect, math::MathDialect,
                  memref::MemRefDialect, ml_program::MLProgramDialect,
                  scf::SCFDialect, tensor::TensorDialect,
                  torch::TMTensor::TMTensorDialect,
                  torch::Torch::TorchDialect,
                  torch::TorchConversion::TorchConversionDialect>();
  func::registerInlinerExtension(registry);

  auto [inputFilename, outputFilename] = registerAndParseCLIOptions(
      argc, argv, "DNN-MLIR modular optimizer\n", registry);
  if (listAvailableQueries) {
    printAvailableQueries(llvm::outs());
    return EXIT_SUCCESS;
  }

  return asMainReturnCode(MlirOptMain(argc, argv, inputFilename,
                                     outputFilename, registry));
}
