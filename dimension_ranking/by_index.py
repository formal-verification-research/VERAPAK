import dimension_ranking.dre as dre

class ByIndex(dre.DimensionRankingEngine):
    def __init__(self):
        pass

    def rank_indices_impl(self, region):
        return [i for i in range(region[0].size)]

# IMPORT INTERFACE
def IMPL():
    return ByIndex()
