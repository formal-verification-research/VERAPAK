from verapak.verification.ve import *
from verapak.utilities.point_tools import *

class DiscreteSearch(VerificationEngine):
    @classmethod
    def get_config_parameters(cls):
        return [{
            "name": "verification_point_threshold",
            "arg_params": {
                "type": "int",
                "help": "Threshold number of discrete points under which verification should occur",
                "default": 10000
            }
        },
        {
            "name": "granularity",
            "arg_params": {
                "type": "x_num_array",
                "help": "Granularity (single value or per dimension array): a valid discretization of the input space (8 bit image -> 1/255)"
            }
        }]
    
    @classmethod
    def evaluate_args(cls, args, v, errors):
        v["granularity"] = args.get("granularity"), # RESHAPED
        v["verification_point_threshold"] = args["verification_point_threshold"]

        if v["granularity"] is None:
            errors.missing("granularity", reason="Discrete search requires granularity")

        # Reshape granularity
        if len(errors) == 0:
            if v["radius"] is not None:
                v["granularity"] = np.array(v["granularity"], dtype=np.string_).reshape(v["graph"].input_shape)
                v["granularity"] *= np.where(np.char.endswith(v["granularity"], "x"), radius, 1)
                v["granularity"] = v["granularity"].reshape(v["graph"].input_shape)
            else:
                v["granularity"] = np.array(v["granularity"], dtype=v["graph"].input_dtype).reshape(v["graph"].input_shape)

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

    def set_config(self, v):
        self.granularity = v["granularity"]
        self.valid_point = v["initial_point"]
        self.point_threshold = v["verification_point_threshold"]
        self.safety_predicate = v["safety_predicate"]
        
        def should_attempt_predicate(region):
            num_points = get_amount_valid_points(
                region, self.granularity, self.valid_point)
            return num_points <= self.point_threshold
        self.should_attempt_checker = should_attempt_predicate

