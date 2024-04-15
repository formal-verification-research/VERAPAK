from .modfgsm import ModFGSM


class FGSM(ModFGSM):
    @classmethod
    def get_config_parameters(cls):
        parent = ModFGSM.get_config_parameters()
        return list(filter( \
                lambda item: item["name"] not in ["balance_factor", "dimension_ranking_strategy"], \
                parent))

    @classmethod
    def evaluate_args(cls, args, v, errors):
        v["granularity"] = args.get("granularity")

    def __init__(self, fallback_strategy=None, fallback_predicate=lambda region: False):
        super().__init__(fallback_strategy=fallback_strategy, fallback_predicate=fallback_predicate)

    def set_config(self, v):
        new_v = {**v}
        new_v["balance_factor"] = 1
        super().set_config(new_v)

