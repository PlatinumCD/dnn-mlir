import os
import lit.formats

from lit.llvm import llvm_config

config.name = "DNN-MLIR"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir", ".py"]
config.excludes = ["lit.cfg.py", "lit.site.cfg.py"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.dnn_mlir_obj_root, "test")

llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.dnn_mlir_tools_dir,
                             append_path=True)

config.substitutions.append(
    ("%dnn-mlir-opt", os.path.join(config.dnn_mlir_tools_dir,
                                    "dnn-mlir-opt")))
if config.dnn_mlir_python_executable:
    config.substitutions.append(("%python", config.dnn_mlir_python_executable))
if config.dnn_mlir_semantic_tests_enabled:
    config.available_features.add("dnn-semantic-tests")
    llvm_config.with_environment(
        "PYTHONPATH", config.dnn_mlir_torch_mlir_python_packages_dir,
        append_path=True)
