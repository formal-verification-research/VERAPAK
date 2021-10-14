from .point_tools import get_amount_valid_points


class LessPointsThanThresholdDiscrete:
    def __init__(self,  granularity, initial_point, threshold=10000):
        self.threshold = threshold
        self.granularity = granularity
        self.initial_point = initial_point

    def __call__(self, region):
        return get_amount_valid_points(region, self.granularity, self.valid_point) <= self.threshold
