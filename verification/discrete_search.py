import verification.ve as ve
import utilities.point_tools


class DiscreteSearch(ve.VerificationEngine):

    def __init__(self, should_attempt_checker, discrete_point_generator):
        self.should_attempt_checker = should_attempt_checker
        self.discrete_point_generator = discrete_point_generator

    def set_config(self, config):
        self.granularity = config["Granularity"]
        self.initial_point = config["Point"]

    def verification_impl(self, region, safety_predicate):
        if not self.should_attempt_checker(region, 
                self.initial_point, self.granularity):      # If we shouldn't attempt:
            return ve.UNKNOWN, None                         # Return UNKNOWN
        else:                                               # Otherwise
            points = self.discrete_point_generator(region,
                    self.granularity, self.initial_point)   # Get discretized points
            for point in points:                            # For each point
                if not safety_predicate(point):             # Check if it's safe
                    return ve.UNSAFE, point                 # If not, return UNSAFE
            return ve.SAFE, None                            # Otherwise, return SAFE


# IMPORT INTERFACE
def IMPL():
    class LessPointsThanThreshold:
        def __init__(self, default_threshold):
            self.threshold = default_threshold
        
        def set_threshold(self, threshold):
            self.threshold = threshold

        def __call__(self, region, granularity, initial_point):
            return point_tools.get_amount_valid_points(region,
                    granularity, initial_point) < self.threshold

    should_attempt_checker = LessPointsThanThreshold(1000)
    discrete_point_generator = point_tools.iter_discrete_points

    return DiscreteSearch(should_attempt_checker,
            discrete_point_generator)

