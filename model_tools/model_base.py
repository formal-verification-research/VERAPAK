import tensorflow as tf
from tensorflow import keras
from .onnx_utils import *
import onnx
from onnx_tf.backend import prepare


class ModelBase:
    def __init__(self, model, input_shape, output_shape, input_dtype, output_dtype):
        self.model_internal = model
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    def evaluate(self, point):
        raise NotImplementedError(
            "Model did not implement the evaluate function")

    def gradient_wrt_input(self, point):
        raise NotImplementedError(
            "Model did not implement the gradient_wrt_input function")


class ONNXModel(ModelBase):
    def __init__(self, path):
        onnx_model = onnx.load(path)
        input_shape = get_onnx_input_shape(onnx_model)
        output_shape = get_onnx_output_shape(onnx_model)
        input_dtype = get_onnx_input_dtype(onnx_model)
        output_dtype = get_onnx_output_dtype(onnx_model)
        super().__init__(prepare(onnx_model), input_shape,
                         output_shape, input_dtype, output_dtype)

    def evaluate(self, point):
        return self.model_internal.run(point.reshape(self.input_shape).astype(np.float32))[0]

    def gradient_wrt_input(self, point):
        pass
