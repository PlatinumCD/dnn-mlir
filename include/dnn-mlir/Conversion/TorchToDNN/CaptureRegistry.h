#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_CAPTUREREGISTRY_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_CAPTUREREGISTRY_H

#include <optional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Operation;

namespace dnn::torch_to_dnn {

struct CaptureRegistration {
  llvm::StringRef section;
  llvm::StringRef capture;
  llvm::StringRef query;
};

llvm::ArrayRef<CaptureRegistration> getCaptureRegistry();
std::optional<llvm::StringRef> getCaptureForQuery(llvm::StringRef query);
llvm::SmallVector<std::string> resolveCaptureQueries(
    llvm::ArrayRef<std::string> queries,
    llvm::ArrayRef<std::string> captures);
LogicalResult validateCaptureSelection(
    Operation *anchor, llvm::ArrayRef<std::string> queries,
    llvm::ArrayRef<std::string> captures);

} // namespace dnn::torch_to_dnn
} // namespace mlir

#endif // DNN_MLIR_CONVERSION_TORCHTODNN_CAPTUREREGISTRY_H
