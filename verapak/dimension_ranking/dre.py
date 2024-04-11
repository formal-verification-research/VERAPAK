class DimensionRankingEngine:
    @classmethod
    def get_config_parameters(cls):
        return []

    @classmethod
    def evaluate_args(cls, v):
        return v

    def rank_indices_impl(self, region):
        raise NotImplementedError("DimensionRankingEngine did not implement rank_indices_impl(region)")

    def set_config(self, config, data):
        pass # Do nothing with the config by default

    def shutdown(self):
        pass # Do nothing during shutdown by default

