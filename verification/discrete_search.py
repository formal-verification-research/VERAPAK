import verification.ve

class DiscreteSearch(ve.VerificationEngine):

    def __init__(self, should_attempt_checker, discrete_point_generator, point_checker):
        self.should_attempt_checker = should_attempt_checker
        self.discrete_point_generator = discrete_point_generator
        self.point_is_safe = point_checker

    def verification_impl(self, region):
        if not self.should_attempt_checker():               # If we shouldn't attempt:
            return ve.UNKNOWN, None                         # Return UNKNOWN
        else:                                               # Otherwise
            points = self.discrete_point_generator(region)  # Get discretized points
            for point in points:                            # For each point
                if not point_is_safe(point):                # Check if it's safe
                    return ve.UNSAFE, point                 # If not, return UNSAFE
            return ve.SAFE, None                            # Otherwise, return SAFE

