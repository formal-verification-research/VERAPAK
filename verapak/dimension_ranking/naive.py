from .dre import DimensionRankingEngine


class NaiveDimRanking(DimensionRankingEngine):
    def __init__(self):
        pass

    def rank_indices_impl(self, region):
        return [i for i in range(region[0].size)]
