# RUN: %python %s %dnn-mlir-opt | FileCheck %s
# REQUIRES: dnn-semantic-tests

import subprocess
import sys

import numpy as np
import torch
from torch_mlir import fx, ir
from torch_mlir.compiler_utils import OutputType, lower_mlir_module
from torch_mlir.dialects import torch as torch_dialect
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)


DNN_MLIR_OPT = sys.argv[1]


class Mm(torch.nn.Module):
    def forward(self, lhs, rhs):
        return torch.mm(lhs, rhs)


class Relu(torch.nn.Module):
    def forward(self, value):
        return torch.relu(value)


class MmRelu(torch.nn.Module):
    def forward(self, lhs, rhs):
        return torch.relu(torch.mm(lhs, rhs))


def run_optimizer(module_text, *arguments):
    completed = subprocess.run(
        [DNN_MLIR_OPT, *arguments],
        input=module_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"dnn-mlir-opt {' '.join(arguments)} failed:\n{completed.stderr}"
        )
    return completed.stdout


def execute_linalg(module_text, inputs):
    context = ir.Context()
    torch_dialect.register_dialect(context)
    with context:
        module = ir.Module.parse(module_text)
        lower_mlir_module(False, OutputType.LINALG_ON_TENSORS, module)
        backend = RefBackendLinalgOnTensorsBackend()
        invoker = backend.load(backend.compile(module))
        return invoker.main(*(value.detach().numpy() for value in inputs))


def check_capture(name, program, inputs, expected_operations):
    program.eval()
    with torch.no_grad():
        expected = program(*inputs).detach().numpy()

    imported = fx.export_and_import(
        program, *inputs, output_type=OutputType.TORCH, func_name="main"
    )
    captured = run_optimizer(str(imported), "-convert-torch-to-dnn")
    for operation in expected_operations:
        if operation not in captured:
            raise AssertionError(f"{name} did not capture {operation}")

    restored = run_optimizer(
        captured, "-test-restore-dnn-to-torch", "-canonicalize"
    )
    if "dnn." in restored or "torch_c." in restored:
        raise AssertionError(f"{name} restoration left capture artifacts")

    actual = execute_linalg(restored, inputs)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6)
    print(f"PASS: {name}")


torch.manual_seed(13)
lhs = torch.randn(2, 3, dtype=torch.float32)
rhs = torch.randn(3, 4, dtype=torch.float32)
values = torch.tensor(
    [[-4.0, -0.0, 0.25, 3.5], [2.0, -1.0, -2.0, 7.0]],
    dtype=torch.float32,
)

check_capture("dnn.mm", Mm(), (lhs, rhs), ("dnn.mm",))
check_capture("dnn.relu", Relu(), (values,), ("dnn.relu",))
check_capture(
    "dnn.mm + dnn.relu", MmRelu(), (lhs, rhs), ("dnn.mm", "dnn.relu")
)

# CHECK: PASS: dnn.mm
# CHECK: PASS: dnn.relu
# CHECK: PASS: dnn.mm + dnn.relu

