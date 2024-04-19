class VerificationEngine:
    @classmethod
    def get_config_parameters(cls):
        return []

    @classmethod
    def evaluate_args(cls, args, v, errors):
        return {}

    def verification_impl(self, region, safety_predicate):
        raise NotImplementedError("VerificationEngine did not implement verification_impl(region, safety_predicate)")

    def set_config(self, v):
        pass # Do nothing with the config by default

    def shutdown(self):
        pass # Do nothing during shutdown by default

ALL_SAFE = 0
ALL_UNSAFE = 1
SOME_UNSAFE = 2
TOO_BIG = 3
UNKNOWN = 3 # NOTE: UNKNOWN currently implies TOO_BIG
BOUNDARY = 4

