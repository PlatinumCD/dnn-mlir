#include "dnn-mlir/Conversion/TorchToDNN/CaptureRegistry.h"

#include "dnn-mlir/Conversion/TorchToDNN/Activation/ActivationPatterns.h"
#include "dnn-mlir/Conversion/TorchToDNN/Convolution/ConvolutionPatterns.h"
#include "dnn-mlir/Conversion/TorchToDNN/Recurrent/RecurrentPatterns.h"
#include "dnn-mlir/Conversion/TorchToDNN/TorchToDNNPatterns.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace mlir::dnn::torch_to_dnn {
namespace {

constexpr CaptureRegistration captureRegistry[] = {
#define DNN_ACTIVATION_CAPTURE(Name, Query, Capture) \
  {"Activation", Capture, Query},
    DNN_ACTIVATION_PATTERN_LIST(DNN_ACTIVATION_CAPTURE)
#undef DNN_ACTIVATION_CAPTURE
    {"Affine", "dnn.linear", "aten.linear"},
#define DNN_CONVOLUTION_CAPTURE(Name, Query, Capture) \
  {"Convolution", Capture, Query},
    DNN_CONVOLUTION_PATTERN_LIST(DNN_CONVOLUTION_CAPTURE)
#undef DNN_CONVOLUTION_CAPTURE
    {"Matrix", "dnn.mm", "aten.mm"},
#define DNN_RECURRENT_CAPTURE(Name, Query, Capture) \
  {"Recurrent", Capture, Query},
    DNN_RECURRENT_PATTERN_LIST(DNN_RECURRENT_CAPTURE)
#undef DNN_RECURRENT_CAPTURE
};

std::string canonicalizeCaptureName(StringRef capture) {
  capture = capture.trim();
  if (!capture.starts_with("dnn."))
    return ("dnn." + capture).str();
  return capture.str();
}

bool containsQuery(ArrayRef<std::string> queries, StringRef query) {
  return llvm::any_of(queries, [&](const std::string &candidate) {
    return candidate == query;
  });
}

} // namespace

ArrayRef<CaptureRegistration> getCaptureRegistry() { return captureRegistry; }

std::optional<StringRef> getCaptureForQuery(StringRef query) {
  std::string canonical = canonicalizeOperationName(query);
  for (const CaptureRegistration &registration : captureRegistry)
    if (registration.query == canonical)
      return registration.capture;
  return std::nullopt;
}

SmallVector<std::string>
resolveCaptureQueries(ArrayRef<std::string> queries,
                      ArrayRef<std::string> captures) {
  SmallVector<std::string> resolved;
  for (const std::string &query : queries) {
    std::string canonical = canonicalizeOperationName(query);
    if (!containsQuery(resolved, canonical))
      resolved.push_back(std::move(canonical));
  }
  for (const std::string &capture : captures) {
    std::string canonical = canonicalizeCaptureName(capture);
    for (const CaptureRegistration &registration : captureRegistry)
      if (registration.capture == canonical &&
          !containsQuery(resolved, registration.query))
        resolved.push_back(registration.query.str());
  }
  return resolved;
}

LogicalResult validateCaptureSelection(Operation *anchor,
                                       ArrayRef<std::string> queries,
                                       ArrayRef<std::string> captures) {
  for (const std::string &query : queries) {
    std::string canonical = canonicalizeOperationName(query);
    if (llvm::any_of(captureRegistry, [&](const CaptureRegistration &entry) {
          return entry.query == canonical;
        }))
      continue;
    return anchor->emitError()
           << "no DNN query is registered for '" << query << "'";
  }
  for (const std::string &capture : captures) {
    std::string canonical = canonicalizeCaptureName(capture);
    if (llvm::any_of(captureRegistry, [&](const CaptureRegistration &entry) {
          return entry.capture == canonical;
        }))
      continue;
    return anchor->emitError()
           << "no DNN capture is registered for '" << capture << "'";
  }
  return success();
}

} // namespace mlir::dnn::torch_to_dnn
