class DimSelectionBase:
    def __init__(self):
        pass

    def rank_indices(self, region):
        return [i for i in range(region[0].size)]
