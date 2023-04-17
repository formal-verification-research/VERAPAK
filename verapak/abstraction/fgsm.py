from .modfgsm import ModFGSM


class FGSM(ModFGSM):
    @staticmethod
    def get_config_parameters():
        parent = ModFGSM.get_config_parameters()
        return list(filter( \
                lambda item: item["name"] not in ["balance_factor", "dimension_ranking_strategy"], \
                parent))

    def __init__(self, fallback_strategy=None, fallback_predicate=lambda region: False):
        super().__init__(fallback_strategy=fallback_strategy, fallback_predicate=fallback_predicate)

    def set_config(self, config, data):
        new_config = {**config}
        new_config["balance_factor"] = 1
        super().set_config(new_config, data)

