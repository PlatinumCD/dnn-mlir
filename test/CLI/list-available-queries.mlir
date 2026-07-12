// RUN: dnn-opt --list-available-queries | FileCheck %s
// RUN: dnn-opt --list-available-ops | FileCheck %s

// CHECK: Available DNN captures and Torch queries by section:
// CHECK: Activation:
// CHECK:   Capture: dnn.relu
// CHECK:   Queries:
// CHECK:     aten.relu
// CHECK: Affine:
// CHECK:   Capture: dnn.linear
// CHECK:     aten.linear
// CHECK: Convolution:
// CHECK:   Capture: dnn.convolution
// CHECK:     aten.conv2d
// CHECK: Matrix:
// CHECK:   Capture: dnn.mm
// CHECK:     aten.mm
// CHECK: Recurrent:
// CHECK:   Capture: dnn.lstm
// CHECK:     aten.lstm.input
// CHECK:     aten.lstm.data
// CHECK:   Capture: dnn.gru
// CHECK:     aten.gru.input
// CHECK:     aten.gru.data
// CHECK:   Capture: dnn.rnn
// CHECK:     aten.rnn_tanh.input
// CHECK:     aten.rnn_tanh.data
// CHECK:     aten.rnn_relu.input
// CHECK:     aten.rnn_relu.data
