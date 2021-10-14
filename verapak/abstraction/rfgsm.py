from .modfgsm import ModFGSM, _min_dim
from .random_point import RandomPoint
from ..dimension_ranking.gradient_based import GradientBasedDimSelection


class RFGSM(ModFGSM):
    def __init__(self, gradient_function, granularity, balance_factor, **kwargs):
        def fallback_predicate(region):
            min_val, min_idx = _min_dim(region)
            return min_val < (2 * granularity[min_idx])

        dim_strat = GradientBasedDimSelection(
            gradient_function)
        super().__init__(gradient_function=gradient_function, granularity=granularity,
                         balance_factor=balance_factor, fallback_strategy=RandomPoint(),
                         fallback_predicate=fallback_predicate, dimension_ranking_strategy=dim_strat)
