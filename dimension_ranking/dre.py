class DimensionRankingEngine:

    def rank_indices_impl(self, region):
        raise NotImplementedError("DimensionRankingEngine did not implement rank_indices_impl(region)")

    def set_config(self, config):
        pass # Do nothing with the config by default
