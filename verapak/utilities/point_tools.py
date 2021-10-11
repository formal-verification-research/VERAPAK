import numpy as np
import math


class AnyToSingle:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, idx):
        return self.value

    def __len__(self):
        return 1


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
    if region[0].size != valid_point.size or converted_granularity.size != valid_point.size:
        raise ValueError('region, granularity and valid point sizes do not match up: {}, {}, {}'.format(
            region[0].size, converted_granularity.size, valid_point.size))
    retVal = np.empty_like(valid_point)
    for i in range(valid_point.size):
        reg_idx = np.unravel_index(i, region[0].shape)
        pnt_idx = np.unravel_index(i, valid_point.shape)
        gran_idx = np.unravel_index(i, converted_granularity.shape)
        multiplier = math.ceil(
            (region[0][reg_idx] - valid_point[pnt_idx]) / converted_granularity[gran_idx])
        value = valid_point[pnt_idx] + \
            multiplier * converted_granularity[gran_idx]
        if value >= region[1][reg_idx] or value < region[0][reg_idx]:
            return None
        retVal[pnt_idx] = value
    return retVal


def get_amount_valid_points(region, granularity, valid_point):
    converted_granularity = granularity_to_array(granularity, valid_point)
    inside_vp = get_valid_point_in_region(region, granularity, valid_point)
    if inside_vp is None:
        return 0
    retVal = 1
    for i in range(region[0].size):
        reg_idx = np.unravel_index(i, region[0].shape)
        pnt_idx = np.unravel_index(i, inside_vp.shape)
        gran_idx = np.unravel_index(i, converted_granularity.shape)
        if region[0][reg_idx] == region[1][reg_idx]:
            continue
        numPointsInDim = math.ceil(
            (region[1][reg_idx] - inside_vp[pnt_idx]) / converted_granularity[gran_idx])
        if numPointsInDim == 0:
            return 0
        retVal *= numPointsInDim
    return retVal


def _enumerate_impl(curPoint, curIndex, region, granularity):
    if curIndex >= region[0].size:
        yield None
        return
    cp_point = curPoint.copy()
    reg_idx = np.unravel_index(curIndex, region[0].shape)
    while cp_point[reg_idx] < region[1][reg_idx]:
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
