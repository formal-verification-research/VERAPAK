from .ve import *
from ..utilities.point_tools import *
from ..parse_arg_types import xNumArrayType

class DiscreteSearch(VerificationEngine):
    @staticmethod
    def get_config_parameters():
        return [{
            "name": "verification_point_threshold",
            "arg_params": {
                "type": int,
                "help": "Threshold number of discrete points under which verification should occur",
                "default": 10000
            }
        },
        {
            "name": "granularity",
            "arg_params": {
                "type": xNumArrayType,
                "help": "Granularity (single value or per dimension array): a valid discretization of the input space (8 bit image -> 1/255)"
            }
        }]

    def __init__(self):
        self.discrete_point_generator = enumerate_valid_points

    def set_safety_predicate(self, safety_predicate):
        self.safety_predicate = safety_predicate

    def verification_impl(self, region, safety_predicate):
        if not self.should_attempt_checker(region):
            return TOO_BIG, None
        try:
            points = self.discrete_point_generator(
                region, self.granularity, self.valid_point)
            print(region[1] - region[0])
            print(self.granularity)
            print(self.valid_point)
            c = 0
            for point in points:
                c += 1
                if not safety_predicate(point):
                    if len(points) == 1:
                        return ALL_UNSAFE, point
                    return SOME_UNSAFE, point
            print(f"Size: {c}")
            return ALL_SAFE, None
        except RecursionError: # Tried to process something too big
            return TOO_BIG, None

    def set_config(self, config, data):
        self.granularity = config["granularity"]
        self.valid_point = initial_point
        self.point_threshold = config["verification_point_threshold"]
        
        def should_attempt_predicate(region):
            num_points = get_amount_valid_points(
                region, self.granularity, self.valid_point)
            return num_points <= self.point_threshold
        self.should_attempt_checker = should_attempt_predicate

