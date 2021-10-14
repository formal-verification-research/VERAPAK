from .pe import PartitioningEngine
from .tools import *
from ..dimension_ranking.largest_first import LargestFirstDimSelection


class LargestFirstPartitioningStrategy(PartitioningEngine):
    def __init__(self, partitioning_divisor, partitioning_num_dimensions, **kwargs):
        self.dim_selection_strat = LargestFirstDimSelection()
        self.divisor = partitioning_divisor
        self.num_dimensions = partitioning_num_dimensions

    def partition_impl(self, region):
        return hierarchicalDimensionRefinement(region, self.dim_selection_strat.rank_indices_impl, self.num_dimensions, self.divisor)


def IMPL():
    return LargestFirstPartitioningStrategy(None, None)
