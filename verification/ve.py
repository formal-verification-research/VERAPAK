class VerificationEngine:

    def verification_impl(self, region, safety_predicate):
        raise NotImplementedError("VerificationEngine did not implement verification_impl(region, safety_predicate)")

    def set_config(self, config):
        pass # Do nothing with the config by default

UNKNOWN = 0
UNSAFE = 1
SAFE = 2

