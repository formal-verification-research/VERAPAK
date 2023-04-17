from .ae import AbstractionEngine
import numpy as np

class CenterPoint(AbstractionEngine):

    def abstraction_impl(self, region, num_abstractions): # num_abstractions = number of split points (1 gives centerpoint)
        retVal = []
        for i in range(1, num_abstractions + 1):
            point = np.empty_like(region[0])
            for j in range(0, region[0].size):
                reg_idx = np.unravel_index(j, region[0].shape)
                point[reg_idx] = (region[0][reg_idx] + region[1][reg_idx]) * i / (num_abstractions + 1)
                # Find the ith even division point on dimension j
                # region[0][j] = Starting point's jth dimension
                # region[j][1] = Ending point's jth dimension
            retVal.append(point)
        return retVal

