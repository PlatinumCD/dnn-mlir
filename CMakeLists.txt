cmake_minimum_required(VERSION 3.22)

project(dnn-mlir VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR at: ${MLIR_DIR}")
message(STATUS "Found LLVM at: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH
  "${MLIR_CMAKE_DIR}"
  "${LLVM_CMAKE_DIR}"
  "${PROJECT_SOURCE_DIR}/cmake"
)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(TorchMLIR)

set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_BINARY_DIR}/include")

set(DNN_MLIR_TORCH_MLIR_SOURCE_DIR
    "${PROJECT_SOURCE_DIR}/externals/torch-mlir"
    CACHE PATH "Path to the Torch-MLIR source tree")
dnn_mlir_configure_torch_mlir()

option(DNN_MLIR_ENABLE_TESTS "Build the DNN-MLIR lit test suite" ON)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
if(DNN_MLIR_ENABLE_TESTS)
  add_subdirectory(test)
endif()

set(DNN_MLIR_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/DNNMLIR"
    CACHE STRING "Installation directory for DNN-MLIR CMake package files")

install(TARGETS DNNIR DNNTorchToDNN
  EXPORT DNNMLIRTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(TARGETS dnn-mlir-opt
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/dnn-mlir"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.td"
)
install(FILES
  "${PROJECT_BINARY_DIR}/include/dnn-mlir/Dialect/DNN/IR/DNNDialect.h.inc"
  "${PROJECT_BINARY_DIR}/include/dnn-mlir/Dialect/DNN/IR/DNNDialect.cpp.inc"
  "${PROJECT_BINARY_DIR}/include/dnn-mlir/Dialect/DNN/IR/DNNOps.h.inc"
  "${PROJECT_BINARY_DIR}/include/dnn-mlir/Dialect/DNN/IR/DNNOps.cpp.inc"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/dnn-mlir/Dialect/DNN/IR"
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/DNNMLIRConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/cmake/DNNMLIRConfig.cmake"
  INSTALL_DESTINATION "${DNN_MLIR_INSTALL_CMAKEDIR}"
)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/cmake/DNNMLIRConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

install(EXPORT DNNMLIRTargets
  FILE DNNMLIRTargets.cmake
  NAMESPACE DNNMLIR::
  DESTINATION "${DNN_MLIR_INSTALL_CMAKEDIR}"
)
install(FILES
  "${PROJECT_BINARY_DIR}/cmake/DNNMLIRConfig.cmake"
  "${PROJECT_BINARY_DIR}/cmake/DNNMLIRConfigVersion.cmake"
  DESTINATION "${DNN_MLIR_INSTALL_CMAKEDIR}"
)
install(FILES "${PROJECT_SOURCE_DIR}/cmake/TorchMLIR.cmake"
  DESTINATION "${DNN_MLIR_INSTALL_CMAKEDIR}"
  RENAME DNNMLIRTorchMLIR.cmake
)
