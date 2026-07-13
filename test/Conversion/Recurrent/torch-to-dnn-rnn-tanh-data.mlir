// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_rnn_tanh_data(
    %data: !torch.vtensor<[7,4],f32>,
    %batch_sizes: !torch.vtensor<[5],si64>,
    %h0: !torch.vtensor<[1,2,3],f32>,
    %wih: !torch.vtensor<[3,4],f32>,
    %whh: !torch.vtensor<[3,3],f32>,
    %bih: !torch.vtensor<[3],f32>,
    %bhh: !torch.vtensor<[3],f32>)
    -> (!torch.vtensor<[7,3],f32>, !torch.vtensor<[1,2,3],f32>) {
  %params = torch.prim.ListConstruct %wih, %whh, %bih, %bhh
      : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,3],f32>,
         !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>)
      -> !torch.list<vtensor>
  %has_biases = torch.constant.bool true
  %layers = torch.constant.int 1
  %dropout = torch.constant.float 0.0
  %false = torch.constant.bool false
  %result:2 = torch.operator "torch.aten.rnn_tanh.data"(
      %data, %batch_sizes, %h0, %params, %has_biases, %layers, %dropout,
      %false, %false)
      : (!torch.vtensor<[7,4],f32>, !torch.vtensor<[5],si64>,
         !torch.vtensor<[1,2,3],f32>, !torch.list<vtensor>, !torch.bool,
         !torch.int, !torch.float, !torch.bool, !torch.bool)
      -> (!torch.vtensor<[7,3],f32>, !torch.vtensor<[1,2,3],f32>)
  return %result#0, %result#1
      : !torch.vtensor<[7,3],f32>, !torch.vtensor<[1,2,3],f32>
}

// CHECK-LABEL: func.func @convert_rnn_tanh_data
// CHECK: dnn.rnn
// CHECK-SAME: activation = "tanh"
// CHECK-SAME: kind = "aten.rnn_tanh.data"
// CHECK-SAME: operand_groups = array<i32: 1, 1, 1, 4, 0, 0, 0, 0, 0>
// CHECK-SAME: parameter_indices = array<i32: 4, 5, 6, 7, 8>
// CHECK-SAME: parameters = [true, 1, 0.000000e+00, false, false]
// CHECK-NOT: torch.operator "torch.aten.rnn_tanh.data"
