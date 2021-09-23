from .model_base import *
import tensorflow as tf
from .onnx_utils import *
import onnx


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
        return self.model_internal.run(point.reshape(self.input_shape).astype(self.input_dtype))[0]

    def gradient_wrt_input(self, point):
        pass
