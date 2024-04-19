import numpy as np

def enumerateAllRegions(partition, region, dim_index, sorted_indices, original_region, num_dimension):
    if dim_index >= num_dimension:
        return True
    newr = [region[0].copy(), region[1].copy()]
    curIndex = np.unravel_index(sorted_indices[dim_index], region[0].shape)
    diff = newr[1][curIndex] - newr[0][curIndex]
    while newr[1][curIndex] <= original_region[1][curIndex]:
        if enumerateAllRegions(partition, newr, dim_index+1, sorted_indices, original_region, num_dimension):
            partition.append([newr[0].copy(), newr[1].copy()])
        if diff <= 0:
            break
        newr[0][curIndex] = newr[1][curIndex]
        newr[1][curIndex] += diff
    return False


def hierarchicalDimensionRefinement(region, dim_select_strategy, num_dims, divisor):
    sorted_indices = dim_select_strategy(region)
    retVal = [Region(region[0].copy(), region[1].copy(), region[2])]
#    firstRegion = [region[0].copy(), region[1].copy()]
    for i in range(num_dims):
        curIndex = np.unravel_index(sorted_indices[i], region[0].shape)
        sizeIncrement = (region[1][curIndex] - region[0][curIndex]) / divisor
        newRetVal = []
        for r in retVal:
            for d in range(divisor):
                r0 = r[0].copy()
                r1 = r[1].copy()
                r0[curIndex] = r0[curIndex] + (sizeIncrement * d)
                r1[curIndex] = r0[curIndex] + sizeIncrement
                newRetVal.append(Region(r0, r1, parent=r))
        retVal = newRetVal
#        if sizeIncrement <= 0:
#            return []
#        firstRegion[1][curIndex] = firstRegion[0][curIndex] + sizeIncrement
#    enumerateAllRegions(retVal, firstRegion, 0,
#                        sorted_indices, region, num_dims)
    return retVal

def Region(lower, upper, data=None, parent=None):
    if data is None and parent is None:
        data = ()
    elif data is None:
        data = parent[2]
    return (lower, upper, data)

