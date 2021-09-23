from .model_base import *
import tensorflow as tf
from .onnx_utils import *
import onnx
import numpy as np


class ONNXModel(ModelBase):
    def __init__(self, path):
        onnx_model = onnx.load(path)
        input_shape = get_onnx_input_shape(onnx_model)
        output_shape = get_onnx_output_shape(onnx_model)
        input_dtype = get_onnx_input_dtype(onnx_model)
        output_dtype = get_onnx_output_dtype(onnx_model)
        super().__init__(prepare(onnx_model), input_shape,
                         output_shape, input_dtype, output_dtype)
        self.model_internal.tensor_dict = self.model_internal.tf_module.gen_tensor_dict(
            {self.model_internal.inputs[0]: self._cast_point_input(np.zeros(self.input_shape))})

    def _cast_point_input(self, point):
        return point.reshape(self.input_shape).astype(self.input_dtype)

    def _cast_point_output(self, point):
        return point.reshape(self.output_shape).astype(self.output_dtype)

    def evaluate(self, point):
        return self.model_internal.run(self._cast_point_input(point))[0]

    def gradient_of_loss_wrt_input(self, point, label):
        label_tf = tf.constant(self._cast_point_output(label))
        in_tf = tf.Variable(self._cast_point_input(point))
        with tf.GradientTape() as g:
            g.watch(in_tf)
            output = self.model_internal.tf_module(
                **{self.model_internal.inputs[0]: in_tf})[self.model_internal.outputs[0]]
            max_out = np.max(output.numpy())
            min_out = np.min(output.numpy())
            from_logits = max_out > 1 or min_out < 0
            loss = tf.losses.categorical_crossentropy(
                label_tf, output, from_logits=from_logits)
        return g.gradient(loss, in_tf)
