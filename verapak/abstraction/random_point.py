from .ae import AbstractionEngine
import numpy as np


class RandomPoint(AbstractionEngine):
    # num_abstractions = number of split points
    def abstraction_impl(self, region, num_abstractions):
        retVal = []
        for i in range(0, num_abstractions):
            point = np.empty_like(region[0])
            for j in range(0, region[0].size):
                unraveled = np.unravel_index(j, region[0].shape)
                # Find a random split point, such that as N increases they will make a uniform distribution
                point[unraveled] = np.random.uniform(
                    region[0][unraveled], region[1][unraveled])
                # region[0][j] = Starting point's jth dimension
                # region[1][j] = Ending point's jth dimension
            retVal.append(point)
        return retVal
