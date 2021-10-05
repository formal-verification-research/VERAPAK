import dimension_ranking.dre as dre

class ByIndex(dre.DimensionRankingEngine):

    def set_config(self, config):
        pass # Does not need any info from config

    def rank_indices_impl(self, region):
        return [i for i in range(region[0].size)]

# IMPORT INTERFACE
def IMPL():
    return ByIndex()
