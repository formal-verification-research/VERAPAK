from .pe import PartitioningEngine
from .tools import *
from ..dimension_ranking.largest_first import LargestFirstDimSelection


class LargestFirstPartitioningStrategy(PartitioningEngine):
    @staticmethod
    def get_config_parameters():
        return [{
            "name": "partitioning_divisor",
            "arg_params": {
                "type": int,
                "help": "Number of divisions on each dimension during partitioning",
                "default": 2 # TODO: Automatically determing based on PC specs, NN size, gradient amplitude, verifier used, and region size?
            }
        },
        {
            "name": "partitioning_dimensions",
            "arg_params": {
                "type": int,
                "help": "Number of dimensions to partition",
                "default": 3
            }
        }]

    def __init__(self):
        self.dim_selection_strat = LargestFirstDimSelection()

    def partition_impl(self, region):
        return hierarchicalDimensionRefinement(region, self.dim_selection_strat.rank_indices_impl, self.num_dimensions, self.divisor)
    
    def set_config(self, config, data):
        self.divisor = config["partitioning_divisor"]
        self.num_dimensions = min(config["partitioning_dimensions"], data["initial_point"].size)
        self.dim_selection_strat.set_config(config, data)

