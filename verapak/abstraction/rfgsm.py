from .modfgsm import ModFGSM
from .random_point import RandomPoint


class RFGSM(ModFGSM):
    @classmethod
    def get_config_parameters(cls):
        from ..dimension_ranking.gradient_based import GradientBasedDimSelection
        parent = ModFGSM.get_config_parameters()
        non_strat = list(filter(lambda item: item["name"] != "dimension_ranking_strategy", parent))
        strat = next(filter(lambda item: item["name"] == "dimension_ranking_strategy", parent))
        strat["arg_params"]["default"] = GradientBasedDimSelection
        return [*non_strat, strat]

    @classmethod
    def evaluate_args(cls, args, v, errors):
        super().evaluate_args(args, v, errors)

    def __init__(self):
        def fallback_predicate(region):
            return False
            #min_val, min_idx = _min_dim(region)
            #return min_val < (2 * granularity[min_idx])
        super().__init__(fallback_strategy=RandomPoint(), fallback_predicate=fallback_predicate)

