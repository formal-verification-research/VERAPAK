import numpy as np
import queue
import sys
from verapak.model_tools import model_base
from verapak.parse_arg_tools import parse_args
from verapak.utilities.point_tools import granularity_to_array, get_amount_valid_points
import tensorflow as tf
import verapak_utils
from verapak import snap


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
    if len(config['domain_upper_bound']) == 1:
        dub = np.full_like(
            config['initial_point'], config['domain_upper_bound'][0])
    else:
        dub = np.array(config['domain_upper_bound']).reshape(
            config['input_shape']).astype(config['input_dtype'])

    if len(config['domain_lower_bound']) == 1:
        dlb = np.full_like(
            config['initial_point'], config['domain_lower_bound'][0])
    else:
        dlb = np.array(config['domain_lower_bound']).reshape(
            config['input_shape']).astype(config['input_dtype'])

    config['domain'] = [dlb, dub]
    config['initial_region'] = snap.region_to_domain([config['initial_point'] - config['radius'],
                                                      config['initial_point'] + config['radius']], config['domain'])


def main(config):
    setup(config)

    unknown_set = verapak_utils.RegionSet()
    adversarial_examples = verapak_utils.PointSet()
    unsafe_set = queue.Queue()
    safe_set = verapak_utils.RegionSet()

    unknown_set.insert(*config['initial_region'])


if __name__ == "__main__":
    config = parse_args(sys.argv[1:], prog=sys.argv[0])
    main(config)
