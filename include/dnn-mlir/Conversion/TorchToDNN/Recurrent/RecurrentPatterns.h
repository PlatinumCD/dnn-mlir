#ifndef DNN_MLIR_CONVERSION_TORCHTODNN_RECURRENT_RECURRENTPATTERNS_H
#define DNN_MLIR_CONVERSION_TORCHTODNN_RECURRENT_RECURRENTPATTERNS_H

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::dnn::torch_to_dnn {

#define DNN_RECURRENT_PATTERN_LIST(X)                              \
  X(AtenLstm, "aten.lstm.input", "dnn.lstm")                     \
  X(AtenLstmData, "aten.lstm.data", "dnn.lstm")                 \
  X(AtenGruInput, "aten.gru.input", "dnn.gru")                  \
  X(AtenGruData, "aten.gru.data", "dnn.gru")                    \
  X(AtenRnnTanhInput, "aten.rnn_tanh.input", "dnn.rnn")         \
  X(AtenRnnTanhData, "aten.rnn_tanh.data", "dnn.rnn")           \
  X(AtenRnnReluInput, "aten.rnn_relu.input", "dnn.rnn")         \
  X(AtenRnnReluData, "aten.rnn_relu.data", "dnn.rnn")

#define DNN_DECLARE_RECURRENT_PATTERN(Name, Operation, Capture) \
  void populate##Name##Pattern(RewritePatternSet &, llvm::ArrayRef<std::string>);
DNN_RECURRENT_PATTERN_LIST(DNN_DECLARE_RECURRENT_PATTERN)
#undef DNN_DECLARE_RECURRENT_PATTERN

void populateNamedLstmPattern(RewritePatternSet &patterns,
                              llvm::ArrayRef<std::string> selectedOperations,
                              llvm::StringRef operationName);
void populateNamedGruPattern(RewritePatternSet &patterns,
                             llvm::ArrayRef<std::string> selectedOperations,
                             llvm::StringRef operationName);
void populateNamedRnnPattern(RewritePatternSet &patterns,
                             llvm::ArrayRef<std::string> selectedOperations,
                             llvm::StringRef operationName,
                             llvm::StringRef activation);

void populateRecurrentPatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> selectedOperations);

} // namespace mlir::dnn::torch_to_dnn

#endif // DNN_MLIR_CONVERSION_TORCHTODNN_RECURRENT_RECURRENTPATTERNS_H
