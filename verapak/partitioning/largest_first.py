from .pe import PartitioningEngine
from .tools import *
from ..dimension_ranking.largest_first import LargestFirstDimSelection


class LargestFirstPartitioningStrategy(PartitioningEngine):
    def __init__(self, divisor, num_dimensions):
        self.dim_selection_strat = LargestFirstDimSelection()
        self.divisor = divisor
        self.num_dimensions = num_dimensions

    def set_config(self, config):
        pass

    def partition_impl(self, region):
        return hierarchicalDimensionRefinement(region, self.dim_selection_strat.rank_indices_impl, self.num_dimensions, self.divisor)


def IMPL():
    return LargestFirstPartitioningStrategy(None, None)
