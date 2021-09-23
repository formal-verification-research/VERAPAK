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


