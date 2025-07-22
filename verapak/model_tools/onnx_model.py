from .model_base import *
import tensorflow as tf
from .onnx_utils import *
import onnx
from onnx_tf.backend import prepare
import numpy as np


class ONNXModel(ModelBase):
    def __init__(self, path):
        self.path = path
        onnx_model = onnx.load(path)
        input_shape = get_onnx_input_shape(onnx_model)
        output_shape = get_onnx_output_shape(onnx_model)
        input_dtype = get_onnx_input_dtype(onnx_model)
        output_dtype = get_onnx_output_dtype(onnx_model)
        self.input_node = get_onnx_input_node(onnx_model)
        self.output_node = get_onnx_output_node(onnx_model)

        if input_shape[0] == -1:
            input_shape = [1] + input_shape[1:]

        super().__init__(prepare(onnx_model), input_shape,
                         output_shape, input_dtype, output_dtype)

    def get_path(self):
        return self.path

    def evaluate(self, point):
        return self.model_internal.run(self._cast_point_input(point))[self.output_node.name]

    def gradient_of_loss_wrt_input(self, point, label):
        label_tf = tf.constant(self._cast_point_output(label))
        in_tf = tf.Variable(self._cast_point_input(point))
        with tf.GradientTape() as tape:
            tape.watch(in_tf)
            output = self.model_internal.tf_module(
                **{self.input_node.name: in_tf})[self.output_node.name]
            loss = output_to_loss(output, label_tf)
        return tape.gradient(loss, in_tf)

