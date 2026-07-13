// RUN: dnn-mlir-opt --list-available-queries | FileCheck %s
// RUN: dnn-mlir-opt --list-available-ops | FileCheck %s

// CHECK: Available DNN captures and Torch queries by section:
// CHECK: Activation:
// CHECK:   Capture: dnn.relu
// CHECK:   Queries:
// CHECK:     aten.relu
// CHECK: Attention:
// CHECK:   Capture: dnn.scaled_dot_product_attention
// CHECK:     aten.scaled_dot_product_attention
// CHECK: Affine:
// CHECK:   Capture: dnn.linear
// CHECK:     aten.linear
// CHECK: Convolution:
// CHECK:   Capture: dnn.convolution
// CHECK:     aten.conv2d
// CHECK: Embedding:
// CHECK:   Capture: dnn.embedding
// CHECK:     aten.embedding
// CHECK: Elementwise:
// CHECK:   Capture: dnn.add
// CHECK:     aten.add.Tensor
// CHECK:   Capture: dnn.mul
// CHECK:     aten.mul.Tensor
// CHECK: Matrix:
// CHECK:   Capture: dnn.mm
// CHECK:     aten.mm
// CHECK:   Capture: dnn.matmul
// CHECK:     aten.matmul
// CHECK: Normalization:
// CHECK:   Capture: dnn.batch_norm
// CHECK:     aten.batch_norm
// CHECK:     aten.native_batch_norm
// CHECK:     aten._native_batch_norm_legit_no_training
// CHECK:   Capture: dnn.layer_norm
// CHECK:     aten.layer_norm
// CHECK: Pooling:
// CHECK:   Capture: dnn.max_pool2d
// CHECK:     aten.max_pool2d
// CHECK:   Capture: dnn.adaptive_avg_pool2d
// CHECK:     aten.adaptive_avg_pool2d
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
// CHECK: Shape:
// CHECK:   Capture: dnn.flatten
// CHECK:     aten.flatten.using_ints
