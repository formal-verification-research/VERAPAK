class DimensionRankingEngine:
    @staticmethod
    def get_config_parameters():
        return []

    def rank_indices_impl(self, region):
        raise NotImplementedError("DimensionRankingEngine did not implement rank_indices_impl(region)")

    def set_config(self, config, data):
        pass # Do nothing with the config by default

    def shutdown(self):
        pass # Do nothing during shutdown by default

