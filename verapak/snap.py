import numpy as np


def region_to_domain(region, domain):
    return [np.clip(region[0], domain[0], domain[1]), np.clip(region[1], domain[0], domain[1])]
    # domain[0] <= region[0] <= domain[1]
    # domain[0] <= region[1] <= domain[1]


def point_to_domain(point, domain):
    # domain[0] <= point <= domain[1]
    return np.clip(point, domain[0], domain[1])

