from .ve import *
from ..utilities.point_tools import *


class DiscreteSearch(VerificationEngine):

    def __init__(self, should_attempt_checker, discrete_point_generator):
        self.should_attempt_checker = should_attempt_checker
        self.discrete_point_generator = discrete_point_generator

    def set_config(self, config):
        self.granularity = config["Granularity"]
        self.initial_point = config["Point"]

    def verification_impl(self, region, safety_predicate):
        if not self.should_attempt_checker(region, 
                self.initial_point, self.granularity):      # If we shouldn't attempt:
            return UNKNOWN, None                         # Return UNKNOWN
        else:                                               # Otherwise
            points = self.discrete_point_generator(region,
                    self.granularity, self.initial_point)   # Get discretized points
            for point in points:                            # For each point
                if not safety_predicate(point):             # Check if it's safe
                    return ve.UNSAFE, point                 # If not, return UNSAFE
            return SAFE, None                            # Otherwise, return SAFE


# IMPORT INTERFACE
def IMPL():
    should_attempt_checker = LessPointsThanThreshold(1000)
    discrete_point_generator = iter_discrete_points

    return DiscreteSearch(should_attempt_checker,
            discrete_point_generator)

