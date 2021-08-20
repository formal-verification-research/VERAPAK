import numpy as np

def _amax_of(*points):
    current_max = points[0]
    for i in range(1, len(points)):
        current_max = np.amax(np.vstack([current_max, points[i]]), axis=0)
    return current_max

def _amin_of(*points):
    current_min = points[0]
    for i in range(1, len(points)):
        current_min = np.amin(np.vstack([current_min, points[i]]), axis=0)
    return current_min

def region_to_domain(region, domain):
    return _amin_of(_amax_of(region[0], domain[0]), domain[1]), _amax_of(_amin_of(region[1], domain[1]), domain[0])

def point_to_domain(point, domain):
    return _amin_of(_amax_of(point, domain[0]), domain[1])

def to_nearest_valid_point(point, initial_point, granularity):
    nearest = point[:]
    if type(granularity) == type([]):
        granularity = np.array(granularity)
    elif type(granularity) != type(np.empty(0)):
        granularity = np.full(point.size, granularity)

    nearest -= initial_point
    nearest /= granularity
    nearest = np.around(nearest)
    nearest *= granularity
    nearest += initial_point

    return nearest
