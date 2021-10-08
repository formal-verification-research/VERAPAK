import numpy as np


def region_to_domain(region, domain):
    return [np.clip(region[0], domain[0], domain[1]), np.clip(region[1], domain[0], domain[1])]
    # domain[0] <= region[0] <= domain[1]
    # domain[0] <= region[1] <= domain[1]


def point_to_domain(point, domain):
    # domain[0] <= point <= domain[1]
    return np.clip(point, domain[0], domain[1])


def to_nearest_valid_point(point, initial_point, granularity):
    nearest = point[:]
    if type(granularity) == type([]):
        granularity = np.array(granularity)
    elif type(granularity) != type(np.empty(0)):
        granularity = np.full(point.size, granularity)

    nearest -= initial_point            # Set the initial point as the origin
    # Scale the grid such that the granularity is an integer step in any direction
    nearest /= granularity
    nearest = np.around(nearest)        # Round to the nearest grid point
    nearest *= granularity              # Undo the scaling
    nearest += initial_point            # Reset the origin

    return nearest
