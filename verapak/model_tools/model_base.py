import numpy as np
import tensorflow as tf

class ModelBase:
    def __init__(self, model, input_shape, output_shape, input_dtype, output_dtype):
        self.model_internal = model
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    def get_path(self):
        raise NotImplementedError()

    def _cast_point_input(self, point):
        return np.array(point).reshape(self.input_shape).astype(self.input_dtype)

    def _cast_point_output(self, point):
        return np.array(point).reshape(self.output_shape).astype(self.output_dtype)

    def evaluate(self, point):
        raise NotImplementedError(
            "Model did not implement the evaluate function")

    def gradient_of_loss_wrt_input(self, point, label):
        raise NotImplementedError(
            "Model did not implement the gradient_wrt_input function")


def load_graph_by_type(graph_path, graph_type):
    if graph_type == "ONNX":
        from .onnx_model import ONNXModel
        return ONNXModel(graph_path)
    elif graph_type == "KERAS":
        from .keras_model import KerasModel
        return KerasModel(graph_path)
    #elif graph_type == "PT":
    #    from .pytorch_model import PyTorchModel
    #    return PyTorchModel(graph_path)

    raise NotImplementedError(
        f"Graph type {graph_type} is not implemented")

def output_to_loss(output, label_tf):
    max_out = np.max(output.numpy())
    min_out = np.min(output.numpy())
    from_logits = max_out > 1 or min_out < 0
    loss = tf.losses.categorical_crossentropy(
        label_tf, output, from_logits=from_logits)
    return loss

