from .model_base import *
import tensorflow as tf
import numpy as np

# TODO
class PyTorchModel(ModelBase):
    def __init__(self, path):
        self.path = path
        keras_model = tf.keras.models.load_model(path)
        inputs = keras_model.layers[0]
        outputs = keras_model.layers[-1]

        if input_shape[0] == -1:
            input_shape = [1] + input_shape[1:]

        super().__init__(keras_model, inputs.shape, outputs.shape
                         inputs.dtype, outputs.dtype)

    def get_path(self):
        return self.path

    def evaluate(self, point):
        return self.model_internal.run(self._cast_point_input(point))[0]

    def gradient_of_loss_wrt_input(self, point, label):
        label_tf = tf.constant(self._cast_point_output(label))
        in_tf = tf.Variable(self._cast_point_input(point))
        with tf.GradientTape() as tape:
            tape.watch(in_tf)
            output = self.model_internal.run(in_tf)
            loss = output_to_loss(output, label_tf)
        return tape.gradient(loss, in_tf)

