from .ve import *
from ..utilities.point_tools import *


class DiscreteSearch(VerificationEngine):

    def __init__(self, verification_point_threshold, granularity, initial_point, **kwargs):
        def should_attempt_predicate(region):
            num_points = get_amount_valid_points(
                region, granularity, initial_point)
            return num_points <= verification_point_threshold
        self.should_attempt_checker = should_attempt_predicate
        self.discrete_point_generator = enumerate_valid_points
        self.granularity = granularity
        self.valid_point = initial_point

    def verification_impl(self, region, safety_predicate):
        if not self.should_attempt_checker(region):
            return UNKNOWN, None
        try:
            points = self.discrete_point_generator(
                region, self.granularity, self.valid_point)
            for point in points:
                if not safety_predicate(point):
                    return UNSAFE, point
            return SAFE, None
        except RecursionError: # Tried to process something too big
            return UNKNOWN, None
