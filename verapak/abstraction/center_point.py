from .ae import AbstractionEngine
import numpy as np

class CenterPoint(AbstractionEngine):

    def abstraction_impl(self, region, num_abstractions): # num_abstractions = number of split points (1 gives centerpoint)
        retVal = []
        for i in range(1, num_abstractions + 1):
            point = np.empty_like(region.low)
            for j in range(0, region.size):
                reg_idx = np.unravel_index(j, region.shape)
                point[reg_idx] = (region.low[reg_idx] + region.high[reg_idx]) * i / (num_abstractions + 1)
                # Find the ith even division point on dimension j
                # region.low[j] = Starting point's jth dimension
                # region.high[j] = Ending point's jth dimension
            retVal.append(point)
        return retVal

