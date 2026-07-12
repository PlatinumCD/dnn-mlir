// RUN: dnn-opt -torchscript-to-dnn-pipeline='captures=dnn.gru' %s | FileCheck %s
// RUN: dnn-opt --dnn-backend-to-linalg-on-tensors-backend-pipeline='captures=dnn.gru' %s | FileCheck %s --check-prefix=LINALG

module {
  func.func private @__torch__.GRU.forward(
      %self: !torch.nn.Module<"__torch__.GRU">,
      %input: !torch.vtensor<[5,2,4],f32>,
      %data: !torch.vtensor<[7,4],f32>,
      %batch_sizes: !torch.vtensor<[5],si64>,
      %h0: !torch.vtensor<[1,2,3],f32>,
      %wih: !torch.vtensor<[9,4],f32>,
      %whh: !torch.vtensor<[9,3],f32>,
      %bih: !torch.vtensor<[9],f32>,
      %bhh: !torch.vtensor<[9],f32>)
      -> (!torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,2,3],f32>,
          !torch.vtensor<[7,3],f32>, !torch.vtensor<[1,2,3],f32>) {
    %params = torch.prim.ListConstruct %wih, %whh, %bih, %bhh
        : (!torch.vtensor<[9,4],f32>, !torch.vtensor<[9,3],f32>,
           !torch.vtensor<[9],f32>, !torch.vtensor<[9],f32>)
        -> !torch.list<vtensor>
    %has_biases = torch.constant.bool true
    %layers = torch.constant.int 1
    %dropout = torch.constant.float 0.0
    %false = torch.constant.bool false
    %padded:2 = torch.operator "torch.aten.gru.input"(
        %input, %h0, %params, %has_biases, %layers, %dropout, %false,
        %false, %false)
        : (!torch.vtensor<[5,2,4],f32>, !torch.vtensor<[1,2,3],f32>,
           !torch.list<vtensor>, !torch.bool, !torch.int, !torch.float,
           !torch.bool, !torch.bool, !torch.bool)
        -> (!torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
    %packed:2 = torch.operator "torch.aten.gru.data"(
        %data, %batch_sizes, %h0, %params, %has_biases, %layers, %dropout,
        %false, %false)
        : (!torch.vtensor<[7,4],f32>, !torch.vtensor<[5],si64>,
           !torch.vtensor<[1,2,3],f32>, !torch.list<vtensor>, !torch.bool,
           !torch.int, !torch.float, !torch.bool, !torch.bool)
        -> (!torch.vtensor<[7,3],f32>, !torch.vtensor<[1,2,3],f32>)
    %activated = torch.aten.relu %padded#0
        : !torch.vtensor<[5,2,3],f32> -> !torch.vtensor<[5,2,3],f32>
    return %activated, %padded#1, %packed#0, %packed#1
        : !torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,2,3],f32>,
          !torch.vtensor<[7,3],f32>, !torch.vtensor<[1,2,3],f32>
  }

  torch.class_type @__torch__.GRU {
    torch.attr private "training" : !torch.bool
    torch.method "forward", @__torch__.GRU.forward
  }

  %false = torch.constant.bool false
  %module = torch.nn_module {
    torch.slot "training", %false : !torch.bool
  } : !torch.nn.Module<"__torch__.GRU">
}

// CHECK-LABEL: func.func @forward
// CHECK: dnn.gru
// CHECK-SAME: kind = "aten.gru.input"
// CHECK-SAME: operand_groups = array<i32: 1, 1, 4, 0, 0, 0, 0, 0, 0>
// CHECK: dnn.gru
// CHECK-SAME: kind = "aten.gru.data"
// CHECK-SAME: operand_groups = array<i32: 1, 1, 1, 4, 0, 0, 0, 0, 0>
// CHECK-NOT: torch.operator "torch.aten.gru"

// LINALG-LABEL: func.func @forward
// LINALG: dnn.gru
// LINALG-SAME: kind = "aten.gru.input"
// LINALG: dnn.gru
// LINALG-SAME: kind = "aten.gru.data"
// LINALG: linalg.generic
// LINALG-NOT: torch.
// LINALG-NOT: torch_c.
