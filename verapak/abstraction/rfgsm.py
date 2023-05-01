from .modfgsm import ModFGSM, _min_dim
from .random_point import RandomPoint
from ..dimension_ranking.gradient_based import GradientBasedDimSelection


class RFGSM(ModFGSM):
    @staticmethod
    def get_config_parameters():
        parent = ModFGSM.get_config_parameters()
        non_strat = list(filter(lambda item: item["name"] != "dimension_ranking_strategy", parent))
        strat = next(filter(lambda item: item["name"] == "dimension_ranking_strategy", parent))
        strat["arg_params"]["default"] = GradientBasedDimSelection
        return [*non_strat, strat]

    def __init__(self):
        def fallback_predicate(region):
            return False
            #min_val, min_idx = _min_dim(region)
            #return min_val < (2 * granularity[min_idx])
        super().__init__(fallback_strategy=RandomPoint(), fallback_predicate=fallback_predicate)

