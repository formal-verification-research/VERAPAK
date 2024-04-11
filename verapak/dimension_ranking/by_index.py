from .dre import DimensionRankingEngine


class ByIndexDimSelection(DimensionRankingEngine):
    def __init__(self, **kwargs):
        pass

    def rank_indices_impl(self, region):
        return [i for i in range(region[0].size)]
