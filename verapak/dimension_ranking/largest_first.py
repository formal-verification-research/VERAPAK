from .dre import *
import numpy as np


class LargestFirstDimSelection(DimensionRankingEngine):
    def __init__(self, **kwargs):
        pass

    def rank_indices_impl(self, region):
        res = list(np.argsort(np.abs(region[1] - region[0]).flatten()))
        res.reverse()
        return res
