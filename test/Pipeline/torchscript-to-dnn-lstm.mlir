// RUN: dnn-mlir-opt -torchscript-to-dnn-pipeline='captures=dnn.lstm' %s | FileCheck %s
// RUN: dnn-mlir-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=dnn.lstm' %s | FileCheck %s --check-prefix=LINALG

module {
  func.func private @__torch__.LSTM.forward(
      %self: !torch.nn.Module<"__torch__.LSTM">,
      %input: !torch.vtensor<[5,2,4],f32>,
      %data: !torch.vtensor<[7,4],f32>,
      %batch_sizes: !torch.vtensor<[5],si64>,
      %h0: !torch.vtensor<[1,2,3],f32>,
      %c0: !torch.vtensor<[1,2,3],f32>,
      %wih: !torch.vtensor<[12,4],f32>,
      %whh: !torch.vtensor<[12,3],f32>,
      %bih: !torch.vtensor<[12],f32>,
      %bhh: !torch.vtensor<[12],f32>)
      -> (!torch.vtensor<[5,2,3],f32>,
          !torch.vtensor<[1,1,2,3],f32>,
          !torch.vtensor<[1,1,2,3],f32>,
          !torch.vtensor<[7,3],f32>,
          !torch.vtensor<[1,1,2,3],f32>,
          !torch.vtensor<[1,1,2,3],f32>) {
    %hx = torch.prim.ListConstruct %h0, %c0
        : (!torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
        -> !torch.list<vtensor>
    %params = torch.prim.ListConstruct %wih, %whh, %bih, %bhh
        : (!torch.vtensor<[12,4],f32>, !torch.vtensor<[12,3],f32>,
           !torch.vtensor<[12],f32>, !torch.vtensor<[12],f32>)
        -> !torch.list<vtensor>
    %has_biases = torch.constant.bool true
    %layers = torch.constant.int 1
    %dropout = torch.constant.float 0.0
    %false = torch.constant.bool false
    %result:3 = torch.operator "torch.aten.lstm.input"(
        %input, %hx, %params, %has_biases, %layers, %dropout, %false,
        %false, %false)
        : (!torch.vtensor<[5,2,4],f32>, !torch.list<vtensor>,
           !torch.list<vtensor>, !torch.bool, !torch.int, !torch.float,
           !torch.bool, !torch.bool, !torch.bool)
        -> (!torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,1,2,3],f32>,
            !torch.vtensor<[1,1,2,3],f32>)
    %packed:3 = torch.operator "torch.aten.lstm.data"(
        %data, %batch_sizes, %hx, %params, %has_biases, %layers, %dropout,
        %false, %false)
        : (!torch.vtensor<[7,4],f32>, !torch.vtensor<[5],si64>,
           !torch.list<vtensor>, !torch.list<vtensor>, !torch.bool, !torch.int,
           !torch.float, !torch.bool, !torch.bool)
        -> (!torch.vtensor<[7,3],f32>, !torch.vtensor<[1,1,2,3],f32>,
            !torch.vtensor<[1,1,2,3],f32>)
    %activated = torch.aten.relu %result#0
        : !torch.vtensor<[5,2,3],f32> -> !torch.vtensor<[5,2,3],f32>
    return %activated, %result#1, %result#2,
           %packed#0, %packed#1, %packed#2
        : !torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,1,2,3],f32>,
          !torch.vtensor<[1,1,2,3],f32>, !torch.vtensor<[7,3],f32>,
          !torch.vtensor<[1,1,2,3],f32>, !torch.vtensor<[1,1,2,3],f32>
  }

  torch.class_type @__torch__.LSTM {
    torch.attr private "training" : !torch.bool
    torch.method "forward", @__torch__.LSTM.forward
  }

  %module_false = torch.constant.bool false
  %module = torch.nn_module {
    torch.slot "training", %module_false : !torch.bool
  } : !torch.nn.Module<"__torch__.LSTM">
}

// CHECK-LABEL: func.func @forward
// CHECK: dnn.lstm
// CHECK-SAME: kind = "aten.lstm.input"
// CHECK-SAME: operand_groups = array<i32: 1, 2, 4, 0, 0, 0, 0, 0, 0>
// CHECK: dnn.lstm
// CHECK-SAME: kind = "aten.lstm.data"
// CHECK-SAME: operand_groups = array<i32: 1, 1, 2, 4, 0, 0, 0, 0, 0>
// CHECK-NOT: torch.operator "torch.aten.lstm.input"
// CHECK-NOT: torch.operator "torch.aten.lstm.data"

// LINALG-LABEL: func.func @forward
// LINALG: dnn.lstm
// LINALG-SAME: kind = "aten.lstm.input"
// LINALG: dnn.lstm
// LINALG-SAME: kind = "aten.lstm.data"
// LINALG: linalg.generic
// LINALG-NOT: torch.
// LINALG-NOT: torch_c.
