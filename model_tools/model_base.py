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

    def gradient_of_loss_wrt_input(self, point, label):
        raise NotImplementedError(
            "Model did not implement the gradient_wrt_input function")

def load_graph_by_type(graph_path, graph_type):
    if graph_type == "ONNX":
        import onnx_model
        config["Graph"] = onnx_model.ONNXModel(graph_path)
    else:
        raise NotImplementedError(f"Graph type {graph_type} is not implemented")

