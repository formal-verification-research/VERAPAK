from .ve import *
from ..utilities.point_tools import *


class DiscreteSearch(VerificationEngine):

    def __init__(self, should_attempt_checker, discrete_point_generator, granularity, valid_point):
        self.should_attempt_checker = should_attempt_checker
        self.discrete_point_generator = discrete_point_generator
        self.granularity = granularity
        self.valid_point = valid_point

    def verification_impl(self, region, safety_predicate):
        if not self.should_attempt_checker(region):
            return UNKNOWN, None
        points = self.discrete_point_generator(
            region, self.granularity, self.valid_point)
        for point in points:
            if not safety_predicate(point):
                return UNSAFE, point
        return SAFE, None
