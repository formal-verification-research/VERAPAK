import numpy as np

def Region(starting_point, ending_point):
    return np.asarray(starting_point), np.asarray(ending_point)

def make_region(initial_point, radii):
    radii = np.asarray(radii)
    return Region(initial_point - (radii / 2), initial_point + (radii / 2))

