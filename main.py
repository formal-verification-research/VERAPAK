import numpy as np
import sys
import verapak.model_tools.model_base
from verapak.cmdline import parse_args


def setup(config):
    graph_path, graph_type = config["Graph"]
    config["Graph"] = model_base.load_graph_by_type(graph_path, graph_type)

    config["Point"] = np.fromstring(config["Point"],
                                    dtype=config["Graph"].input_dtype, sep=",").reshape(config["Graph"].input_shape)

    if config["Label"] is None:
        config["Label"] = np.argmax(config["Graph"].evaluate(config["Point"]))

    if len(config["Granularity"]) == 1:
        config["Granularity"] = np.full_like(
            config["Point"], config["Granularity"][0])
    else:
        config["Granularity"] = config["Granularity"].reshape(
            config["Point"].shape).astype(config["Graph"].input_dtype)

    if len(config["Radius"]) == 1:
        config["Radius"] = np.full_like(config["Point"], config["Radius"][0])
    else:
        config["Radius"] = config["Radius"].reshape(config["Point"].shape).astype(config["Graph"].input_dtype)

    config["Verification Strategy"].set_config(config)
    config["Dimension-Ranking Strategy"].set_config(config)
    config["Abstraction Strategy"].set_config(config)
    config["Partitioning Strategy"].set_config(config)


def main(config):
    pass


if __name__ == "__main__":
    config = parse_args(sys.argv[1:], prog=sys.argv[0])
    setup(config)
    main(config)
