import abstraction.ae
import random
import numpy as np

class RandomPoint(ae.AbstractionEngine):

    def abstraction_impl(self, region, num_abstractions): # num_abstractions = number of split points
        retVal = []
        for i in range(0, num_abstractions):
            point = np.empty(region.size())
            for j in range(0, region.size()):
                # Find a random split point, such that as N increases they will make a uniform distribution
                point[j] = random.uniform(region[j][0], region[j][1])
                # region[j][0] = Starting point of Region j
                # region[j][1] = Ending point of Region j
            retVal.append(point)
        return retVal

