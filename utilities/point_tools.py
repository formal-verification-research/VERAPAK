import numpy as np
import math


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
    if isinstance(granularity, int):
        return np.full_like(point, granularity)
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
