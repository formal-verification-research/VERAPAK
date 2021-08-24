import abstraction.ae as ae
import numpy as np

class CenterPoint(ae.AbstractionEngine):

    def abstraction_impl(self, region, num_abstractions): # num_abstractions = number of split points (1 gives centerpoint)
        retVal = []
        for i in range(1, num_abstractions + 1):
            point = np.empty(region[0].size)
            for j in range(0, region[0].size):
                point[j] = (region[0][j] + region[1][j]) * i / (num_abstractions + 1)
                # Find the ith even division point on dimension j
                # region[0][j] = Starting point's jth dimension
                # region[j][1] = Ending point's jth dimension
            retVal.append(point)
        return retVal

