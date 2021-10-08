from .modfgsm import ModFGSM, _min_dim
from .random_point import RandomPoint
from ..dimension_ranking.gradient_based import GradientBasedDimSelection


class RFGSM(ModFGSM):
    def __init__(self, grad_func, granularity, percent_fgsm=0.9):
        def fallback_predicate(region):
            min_val, min_idx = _min_dim(region)
            return min_val < (3 * granularity[min_idx])

        dim_selection = GradientBasedDimSelection(grad_func)
        super().__init__(grad_func=grad_func, granularity=granularity,
                         percent_fgsm=percent_fgsm, fallback_strategy=RandomPoint(),
                         fallback_predicate=fallback_predicate, dim_selection_strategy=dim_selection)
