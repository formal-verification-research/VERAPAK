import verification.ve

class DiscreteSearch(ve.VerificationEngine):

    def __init__(self, shouldAttemptChecker, discretePointGenerator, pointChecker):
        self.shouldAttemptChecker = shouldAttemptChecker
        self.discretePointGenerator = discretePointGenerator
        self.pointIsSafe = pointChecker

    def verification_impl(self, region):
        if not self.shouldAttemptChecker():                 # If we shouldn't attempt:
            return ve.UNKNOWN, None                         # Return UNKNOWN
        else:                                               # Otherwise
            points = self.discretePointGenerator(region)    # Get discretized points
            for point in points:                            # For each point
                if not pointIsSafe(point):                  # Check if it's safe
                    return ve.UNSAFE, point                 # If not, return UNSAFE
            return ve.SAFE, None                            # Otherwise, return SAFE

