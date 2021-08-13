import abstraction.ae
import numpy as np

class CenterPoint(ae.AbstractionEngine):

    def abstraction_impl(self, region, num_abstractions): # num_abstractions = number of split points (1 gives centerpoint)
        retVal = []
        for i in range(1, num_abstractions + 1):
            point = np.empty(region.size())
            for j in range(0, region.size()):
                point[j] = (region[j][0] + region[j][1]) * i / (num_abstractions + 1)
                # Find the ith even division point on dimension j
                # region[j][0] = Starting point of Region j
                # region[j][1] = Ending point of Region j
            retVal.append(point)
        return retVal

