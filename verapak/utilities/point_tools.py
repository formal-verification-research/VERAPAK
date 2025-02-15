import numpy as np
import math


def point_in_region(region, point):
    return np.all(point >= region.low) and np.all(point < region.high)


def granularity_to_array(granularity, point=None):
    if point is None and not isinstance(granularity, np.ndarray) and not isinstance(granularity, list):
        raise ValueError(
            'shape or length must be specified if granularity is not list or np array')
    if isinstance(granularity, np.ndarray):
        return granularity
    if isinstance(granularity, list):
        converted = np.array(granularity)
        if point is None:
            return converted
        return converted.reshape(point.shape)
    if isinstance(granularity, int) or isinstance(granularity, float):
        return np.full_like(point, granularity)
    if isinstance(granularity, AnyToSingle):
        return np.full_like(point, granularity.value)
    raise ValueError('granularity must be either a list, numpy array, or int')


def is_point_valid(point, granularity, valid_point):
    converted_granularity = granularity_to_array(granularity, valid_point)
    return np.all(np.abs(point - valid_point) % converted_granularity == 0)


def get_valid_point_in_region(region, granularity, valid_point):
    converted_granularity = granularity_to_array(granularity, valid_point)
    if region.size != valid_point.size or converted_granularity.size != valid_point.size:
        raise ValueError('region, granularity and valid point sizes do not match up: {}, {}, {}'.format(
            region.size, converted_granularity.size, valid_point.size))
    retVal = np.empty_like(valid_point)
    for i in range(valid_point.size):
        reg_idx = np.unravel_index(i, region.shape)
        pnt_idx = np.unravel_index(i, valid_point.shape)
        gran_idx = np.unravel_index(i, converted_granularity.shape)
        multiplier = math.ceil(
            (region.low[reg_idx] - valid_point[pnt_idx]) / converted_granularity[gran_idx])
        value = valid_point[pnt_idx] + \
            multiplier * converted_granularity[gran_idx]
        if value < region.low[reg_idx]: # Below the lower bound
            while value < region.low[reg_idx] and value < region.high[reg_idx]: # Below both the lower and upper bounds
                multiplier += 1
                value = valid_point[pnt_idx] + \
                        multiplier * converted_granularity[gran_idx]
        elif value >= region.high[reg_idx]: # Above the upper bound
            while value >= region.high[reg_idx] and value >= region.low[reg_idx]: # Above both the upper and lower bound
                multiplier -= 1
                value = valid_point[pnt_idx] + \
                        multiplier * converted_granularity[gran_idx]
        if not(value >= region.low[reg_idx] and value < region.high[reg_idx]): # Still out of bounds -- no interior point(s)
            #print(f"{region.low[reg_idx]} <= {value} < {region.high[reg_idx]} failed for i={i}, reg_idx={reg_idx}\n\treg={region.low} -> {region.high}\n\tvalid_point={valid_point},\n\tgran={converted_granularity}")
            return None
        retVal[pnt_idx] = value
    return retVal


def get_amount_valid_points(region, granularity, valid_point):
    converted_granularity = granularity_to_array(granularity, valid_point)
    inside_vp = get_valid_point_in_region(region, granularity, valid_point)
    if inside_vp is None:
        return 0
    retVal = 1
    for i in range(region.size):
        reg_idx = np.unravel_index(i, region.shape)
        pnt_idx = np.unravel_index(i, inside_vp.shape)
        gran_idx = np.unravel_index(i, converted_granularity.shape)
        if region.low[reg_idx] == region.high[reg_idx]:
            continue
        numPointsInDim = math.ceil(
            (region.high[reg_idx] - inside_vp[pnt_idx]) / converted_granularity[gran_idx])
        if numPointsInDim == 0:
            return 0
        retVal *= numPointsInDim
    return retVal


# TODO: Implement in a more memory-efficient way
# TODO: Make sure this works for dimensions of width 0
#       (i.e. only one point along that dimension)
def _enumerate_impl(curPoint, curIndex, region, granularity):
    if curIndex >= region.size:
        #yield None
        return
    cp_point = curPoint.copy()
    reg_idx = np.unravel_index(curIndex, region.shape)
    while cp_point[reg_idx] < region.high[reg_idx]:
        last_val = False
        for i in _enumerate_impl(cp_point, curIndex + 1, region, granularity):
            if not (i is None):
                yield i
            else:
                last_val = True
        if last_val:
            yield cp_point.copy()
        if granularity[reg_idx] <= 0.0:
            break
        cp_point[reg_idx] += granularity[reg_idx]


def enumerate_valid_points(region, granularity, valid_point):
    vp = get_valid_point_in_region(region, granularity, valid_point)
    if vp is None:
        return
    yield from _enumerate_impl(vp, 0, region, granularity)
