option(DNN_MLIR_ENABLE_SEMANTIC_TESTS
  "Run numerical FX capture tests through Torch-MLIR's reference backend" ON)
set(DNN_MLIR_PYTHON_EXECUTABLE "" CACHE FILEPATH
  "Python interpreter containing PyTorch and NumPy for semantic tests")
set(DNN_MLIR_TORCH_MLIR_PYTHON_PACKAGES_DIR "" CACHE PATH
  "Directory containing the built torch_mlir Python package")

if(NOT DNN_MLIR_PYTHON_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    set(DNN_MLIR_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
  endif()
endif()

if(NOT DNN_MLIR_TORCH_MLIR_PYTHON_PACKAGES_DIR)
  get_filename_component(_torch_mlir_build_root "${MLIR_DIR}/../../.." ABSOLUTE)
  set(_torch_mlir_python_candidate
    "${_torch_mlir_build_root}/tools/torch-mlir/python_packages/torch_mlir")
  if(EXISTS "${_torch_mlir_python_candidate}/torch_mlir")
    set(DNN_MLIR_TORCH_MLIR_PYTHON_PACKAGES_DIR
      "${_torch_mlir_python_candidate}")
  endif()
endif()

set(DNN_MLIR_SEMANTIC_TESTS_ENABLED 0)
if(DNN_MLIR_ENABLE_SEMANTIC_TESTS AND DNN_MLIR_PYTHON_EXECUTABLE AND
   DNN_MLIR_TORCH_MLIR_PYTHON_PACKAGES_DIR)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${DNN_MLIR_TORCH_MLIR_PYTHON_PACKAGES_DIR}"
      "${DNN_MLIR_PYTHON_EXECUTABLE}" -c
      "import numpy, torch, torch_mlir, torch_mlir_e2e_test"
    RESULT_VARIABLE _semantic_import_result
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(_semantic_import_result EQUAL 0)
    set(DNN_MLIR_SEMANTIC_TESTS_ENABLED 1)
    message(STATUS "DNN-MLIR numerical semantic tests enabled")
  else()
    message(STATUS
      "DNN-MLIR numerical semantic tests disabled: Python dependencies unavailable")
  endif()
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-dnn-mlir
  "Running the DNN-MLIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS dnn-mlir-opt
)

set(_installed_consumer_prefix
  "${CMAKE_CURRENT_BINARY_DIR}/installed-consumer/prefix")
set(_installed_consumer_build
  "${CMAKE_CURRENT_BINARY_DIR}/installed-consumer/build")
add_custom_target(check-dnn-mlir-install
  COMMAND "${CMAKE_COMMAND}" -E remove_directory
    "${CMAKE_CURRENT_BINARY_DIR}/installed-consumer"
  COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
    --prefix "${_installed_consumer_prefix}"
  COMMAND "${_installed_consumer_prefix}/bin/dnn-mlir-opt"
    /dev/null -o /dev/null
  COMMAND "${CMAKE_COMMAND}"
    -S "${CMAKE_CURRENT_SOURCE_DIR}/InstalledConsumer"
    -B "${_installed_consumer_build}"
    -G "${CMAKE_GENERATOR}"
    "-DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
    "-DCMAKE_PREFIX_PATH=${_installed_consumer_prefix}"
    "-DMLIR_DIR=${MLIR_DIR}"
    "-DLLVM_DIR=${LLVM_DIR}"
    "-DDNN_MLIR_TORCH_MLIR_SOURCE_DIR=${DNN_MLIR_TORCH_MLIR_SOURCE_DIR}"
    "-DDNN_MLIR_TORCH_MLIR_BINARY_DIR=${DNN_MLIR_TORCH_MLIR_BINARY_DIR}"
  COMMAND "${CMAKE_COMMAND}" --build "${_installed_consumer_build}"
  COMMAND "${_installed_consumer_build}/dnn-mlir-installed-consumer"
  DEPENDS DNNIR DNNTorchToDNN dnn-mlir-opt
  COMMENT "Testing the installed DNN-MLIR CMake package"
  VERBATIM
)
add_dependencies(check-dnn-mlir check-dnn-mlir-install)
