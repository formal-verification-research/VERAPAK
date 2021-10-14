import numpy as np
import sys
from verapak.model_tools import model_base
from verapak.parse_arg_tools import parse_args
from verapak.utilities.point_tools import granularity_to_array
import tensorflow as tf


def setup(config):
    graph_path = config["graph"]
    config["graph"] = model_base.load_graph_by_type(graph_path, 'ONNX')
    config["input_shape"] = config["graph"].input_shape
    config["input_dtype"] = config["graph"].input_dtype
    config["output_shape"] = config["graph"].output_shape
    config["output_dtype"] = config["graph"].output_dtype

    if len(config["granularity"]) == 1:
        config['granularity'] = np.full(
            config['input_shape'], config['granularity'][0]).astype(config['input_dtype'])
    else:
        config['granularity'] = np.array(
            config['granularity'], dtype=config['input_dtype']).reshape(config['input_shape'])

    config["initial_point"] = np.array(
        config["initial_point"], dtype=config["input_dtype"]).reshape(config["input_shape"])

    if config["label"] is None:
        config["label"] = np.argmax(
            config["graph"].evaluate(config["initial_point"]).flatten())

    config['label'] = tf.keras.utils.to_categorical(
        config['label'], num_classes=np.prod(config['output_shape']), dtype=config['output_dtype'])

    if len(config["radius"]) == 1:
        config["radius"] = np.full_like(
            config["initial_point"], config["radius"][0])
    else:
        config["radius"] = np.array(config["radius"]).reshape(
            config["input_shape"]).astype(config["input_dtype"])

    def gradient_function(point):
        return config['graph'].gradient_of_loss_wrt_input(point, config['label'])

    config['gradient_function'] = gradient_function
    config['abstraction_strategy'] = config['abstraction_strategy'](**config)

    if config['partitioning_num_dimensions'] > config['initial_point'].size:
        config['partitioning_num_dimensions'] = config['initial_point'].size

    config['partitioning_strategy'] = config['partitioning_strategy'](**config)
    config['dimension_ranking_strategy'] = config['dimension_ranking_strategy'](
        **config)

    config['verification_strategy'] = config['verification_strategy'](**config)


def main(config):
    setup(config)
    print(config)


if __name__ == "__main__":
    config = parse_args(sys.argv[1:], prog=sys.argv[0])
    main(config)
