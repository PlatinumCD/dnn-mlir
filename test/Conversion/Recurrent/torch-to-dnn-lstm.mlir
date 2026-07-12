// RUN: dnn-opt -convert-torch-to-dnn %s | FileCheck %s

module {
  func.func @convert_fx_aten_lstm(
      %input: !torch.vtensor<[5,2,4],f32>,
      %h0: !torch.vtensor<[1,2,3],f32>,
      %c0: !torch.vtensor<[1,2,3],f32>,
      %wih: !torch.vtensor<[12,4],f32>,
      %whh: !torch.vtensor<[12,3],f32>,
      %bih: !torch.vtensor<[12],f32>,
      %bhh: !torch.vtensor<[12],f32>)
      -> (!torch.vtensor<[5,2,3],f32>,
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
    %train = torch.constant.bool false
    %bidirectional = torch.constant.bool false
    %batch_first = torch.constant.bool false
    %result:3 = torch.operator "torch.aten.lstm.input"(
        %input, %hx, %params, %has_biases, %layers, %dropout, %train,
        %bidirectional, %batch_first)
        : (!torch.vtensor<[5,2,4],f32>, !torch.list<vtensor>,
           !torch.list<vtensor>, !torch.bool, !torch.int, !torch.float,
           !torch.bool, !torch.bool, !torch.bool)
        -> (!torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,1,2,3],f32>,
            !torch.vtensor<[1,1,2,3],f32>)
    return %result#0, %result#1, %result#2
        : !torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,1,2,3],f32>,
          !torch.vtensor<[1,1,2,3],f32>
  }

}

// CHECK-LABEL: func.func @convert_fx_aten_lstm
// CHECK: dnn.lstm
// CHECK-SAME: kind = "aten.lstm.input"
// CHECK-SAME: operand_groups = array<i32: 1, 2, 4, 0, 0, 0, 0, 0, 0>
// CHECK-SAME: parameter_indices = array<i32: 3, 4, 5, 6, 7, 8>
// CHECK-SAME: parameters = [true, 1, 0.000000e+00, false, false, false]
// CHECK-NOT: torch.operator "torch.aten.lstm.input"
