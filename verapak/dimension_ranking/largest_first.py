from verapak.dimension_ranking.dre import DimensionRankingEngine
import numpy as np


class LargestFirstDimSelection(DimensionRankingEngine):
    def rank_indices_impl(self, region):
        res = list(np.argsort(np.abs(region.high - region.low).flatten()))
        res.reverse()
        return res
