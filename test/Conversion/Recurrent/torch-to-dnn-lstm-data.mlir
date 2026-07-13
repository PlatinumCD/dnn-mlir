// RUN: dnn-mlir-opt -convert-torch-to-dnn %s | FileCheck %s

func.func @convert_fx_aten_lstm_data(
    %data: !torch.vtensor<[7,4],f32>,
    %batch_sizes: !torch.vtensor<[5],si64>,
    %h0: !torch.vtensor<[1,2,3],f32>,
    %c0: !torch.vtensor<[1,2,3],f32>,
    %wih: !torch.vtensor<[12,4],f32>,
    %whh: !torch.vtensor<[12,3],f32>,
    %bih: !torch.vtensor<[12],f32>,
    %bhh: !torch.vtensor<[12],f32>)
    -> (!torch.vtensor<[7,3],f32>, !torch.vtensor<[1,2,3],f32>,
        !torch.vtensor<[1,2,3],f32>) {
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
  %result:3 = torch.operator "torch.aten.lstm.data"(
      %data, %batch_sizes, %hx, %params, %has_biases, %layers, %dropout,
      %false, %false)
      : (!torch.vtensor<[7,4],f32>, !torch.vtensor<[5],si64>,
         !torch.list<vtensor>, !torch.list<vtensor>, !torch.bool, !torch.int,
         !torch.float, !torch.bool, !torch.bool)
      -> (!torch.vtensor<[7,3],f32>, !torch.vtensor<[1,2,3],f32>,
          !torch.vtensor<[1,2,3],f32>)
  return %result#0, %result#1, %result#2
      : !torch.vtensor<[7,3],f32>, !torch.vtensor<[1,2,3],f32>,
        !torch.vtensor<[1,2,3],f32>
}

// CHECK-LABEL: func.func @convert_fx_aten_lstm_data
// CHECK: dnn.lstm
// CHECK-SAME: kind = "aten.lstm.data"
// CHECK-SAME: operand_groups = array<i32: 1, 1, 2, 4, 0, 0, 0, 0, 0>
// CHECK-SAME: parameter_indices = array<i32: 4, 5, 6, 7, 8>
// CHECK-SAME: parameters = [true, 1, 0.000000e+00, false, false]
// CHECK-NOT: torch.operator "torch.aten.lstm.data"
