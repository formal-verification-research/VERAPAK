from verapak.partitioning.pe import PartitioningEngine
from verapak.partitioning.tools import *
from verapak.dimension_ranking.largest_first import LargestFirstDimSelection
from functools import reduce


class LargestFirstPartitioningStrategy(PartitioningEngine):
    @classmethod
    def get_config_parameters(cls):
        return [{
            "name": "partitioning_divisor",
            "arg_params": {
                "type": "int",
                "help": "Number of divisions on each dimension during partitioning",
                "default": 2 # TODO: Automatically determing based on PC specs, NN size, gradient amplitude, verifier used, and region size?
            }
        },
        {
            "name": "partitioning_dimensions",
            "arg_params": {
                "type": "int",
                "help": "Number of dimensions to partition",
                "default": 3
            }
        }]

    @classmethod
    def evaluate_args(cls, args, v, errors):
        v["partitioning_divisor"] = args["partitioning_divisor"]
        v["partitioning_dimensions"] = args["partitioning_dimensions"]
        
        if len(errors) == 0:
            graph_input_dims = reduce(lambda a, b: a*b, v["graph"].input_shape, 1)
            if v["partitioning_dimensions"] > graph_input_dims:
                v["partitioning_dimensions"] = graph_input_dims

    def __init__(self):
        self.dim_selection_strat = LargestFirstDimSelection()

    def partition_impl(self, region):
        return hierarchicalDimensionRefinement(region, self.dim_selection_strat.rank_indices_impl, self.num_dimensions, self.divisor)
    
    def set_config(self, v):
        self.divisor = v["partitioning_divisor"]
        self.num_dimensions = v["partitioning_dimensions"]
        self.dim_selection_strat.set_config(v)

