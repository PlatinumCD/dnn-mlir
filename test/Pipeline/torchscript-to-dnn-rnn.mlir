// RUN: dnn-opt -torchscript-to-dnn-pipeline='captures=dnn.rnn' %s | FileCheck %s

module {
  func.func private @__torch__.RNN.forward(
      %self: !torch.nn.Module<"__torch__.RNN">,
      %input: !torch.vtensor<[5,2,4],f32>,
      %h0: !torch.vtensor<[1,2,3],f32>,
      %wih: !torch.vtensor<[3,4],f32>,
      %whh: !torch.vtensor<[3,3],f32>,
      %bih: !torch.vtensor<[3],f32>,
      %bhh: !torch.vtensor<[3],f32>)
      -> (!torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,2,3],f32>,
          !torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,2,3],f32>) {
    %params = torch.prim.ListConstruct %wih, %whh, %bih, %bhh
        : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[3,3],f32>,
           !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>)
        -> !torch.list<vtensor>
    %has_biases = torch.constant.bool true
    %layers = torch.constant.int 1
    %dropout = torch.constant.float 0.0
    %false = torch.constant.bool false
    %tanh:2 = torch.operator "torch.aten.rnn_tanh.input"(
        %input, %h0, %params, %has_biases, %layers, %dropout, %false,
        %false, %false)
        : (!torch.vtensor<[5,2,4],f32>, !torch.vtensor<[1,2,3],f32>,
           !torch.list<vtensor>, !torch.bool, !torch.int, !torch.float,
           !torch.bool, !torch.bool, !torch.bool)
        -> (!torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
    %relu:2 = torch.operator "torch.aten.rnn_relu.input"(
        %input, %h0, %params, %has_biases, %layers, %dropout, %false,
        %false, %false)
        : (!torch.vtensor<[5,2,4],f32>, !torch.vtensor<[1,2,3],f32>,
           !torch.list<vtensor>, !torch.bool, !torch.int, !torch.float,
           !torch.bool, !torch.bool, !torch.bool)
        -> (!torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,2,3],f32>)
    return %tanh#0, %tanh#1, %relu#0, %relu#1
        : !torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,2,3],f32>,
          !torch.vtensor<[5,2,3],f32>, !torch.vtensor<[1,2,3],f32>
  }

  torch.class_type @__torch__.RNN {
    torch.attr private "training" : !torch.bool
    torch.method "forward", @__torch__.RNN.forward
  }

  %module_false = torch.constant.bool false
  %module = torch.nn_module {
    torch.slot "training", %module_false : !torch.bool
  } : !torch.nn.Module<"__torch__.RNN">
}

// CHECK-LABEL: func.func @forward
// CHECK: dnn.rnn
// CHECK-SAME: activation = "tanh"
// CHECK-SAME: kind = "aten.rnn_tanh.input"
// CHECK: dnn.rnn
// CHECK-SAME: activation = "relu"
// CHECK-SAME: kind = "aten.rnn_relu.input"
// CHECK-NOT: torch.operator "torch.aten.rnn_
