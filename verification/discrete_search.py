import verification.ve as ve

class DiscreteSearch(ve.VerificationEngine):

    def __init__(self, should_attempt_checker, discrete_point_generator):
        self.should_attempt_checker = should_attempt_checker
        self.discrete_point_generator = discrete_point_generator

    def verification_impl(self, region, safety_predicate):
        if not self.should_attempt_checker(region):         # If we shouldn't attempt:
            return ve.UNKNOWN, None                         # Return UNKNOWN
        else:                                               # Otherwise
            points = self.discrete_point_generator(region)  # Get discretized points
            for point in points:                            # For each point
                if not safety_predicate(point):             # Check if it's safe
                    return ve.UNSAFE, point                 # If not, return UNSAFE
            return ve.SAFE, None                            # Otherwise, return SAFE


# IMPORT INTERFACE
def IMPL():
    pass
#    return DiscreteSearch(
#            lambda region: True,
#            lambda region: [],
#    )

