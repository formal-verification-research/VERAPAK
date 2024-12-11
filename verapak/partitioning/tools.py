import numpy as np

def hierarchicalDimensionRefinement(region, dim_select_strategy, num_dims, divisor):
    sorted_indices = dim_select_strategy(region)
    # Starting region (i.e. what should be returned if num_dims == 0)
    retVal = [region]
    for i in range(num_dims):
        # Shaped index location
        curIndex = np.unravel_index(sorted_indices[i], region.low.shape)
        # Size of each region
        sizeIncrement = (region.high[curIndex] - region.low[curIndex]) / divisor

        newRetVal = []
        for r in retVal:
            for d in range(divisor):
                # Low = region.low + sizeIncrement * i
                low = r.low.copy()
                low[curIndex] = low[curIndex] + (sizeIncrement * d)
                # High = low + sizeIncrement
                high = r.high.copy()
                high[curIndex] = low[curIndex] + sizeIncrement

                newRetVal.append(Region(low, high, RegionData.make_child(region.data)))
        retVal = newRetVal
    return retVal

