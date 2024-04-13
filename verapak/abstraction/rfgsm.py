from .modfgsm import ModFGSM
from .random_point import RandomPoint
from ..dimension_ranking.gradient_based import GradientBasedDimSelection


class RFGSM(ModFGSM):
    @classmethod
    def get_config_parameters(cls):
        parent = ModFGSM.get_config_parameters()
        non_strat = list(filter(lambda item: item["name"] != "dimension_ranking_strategy", parent))
        strat = next(filter(lambda item: item["name"] == "dimension_ranking_strategy", parent))
        strat["arg_params"]["default"] = GradientBasedDimSelection
        return [*non_strat, strat]

    @classmethod
    def evaluate_args(cls, args, v):
        super().evaluate_args(args, v)

    def __init__(self):
        def fallback_predicate(region):
            return False
            #min_val, min_idx = _min_dim(region)
            #return min_val < (2 * granularity[min_idx])
        super().__init__(fallback_strategy=RandomPoint(), fallback_predicate=fallback_predicate)

