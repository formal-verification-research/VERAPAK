from .ae import AbstractionEngine
import random
import numpy as np

class RandomPoint(AbstractionEngine):

    def set_config(self, config):
        pass # Does not need any info from config

    def abstraction_impl(self, region, num_abstractions): # num_abstractions = number of split points
        retVal = []
        for i in range(0, num_abstractions):
            point = np.empty(region[0].size)
            for j in range(0, region[0].size):
                # Find a random split point, such that as N increases they will make a uniform distribution
                point[j] = random.uniform(region[0][j], region[1][j])
                # region[0][j] = Starting point's jth dimension
                # region[1][j] = Ending point's jth dimension
            retVal.append(point)
        return retVal


# IMPORT INTERFACE
def IMPL():
    return RandomPoint()

