class VerificationEngine:
    @staticmethod
    def get_config_parameters():
        return []

    def verification_impl(self, region, safety_predicate):
        raise NotImplementedError("VerificationEngine did not implement verification_impl(region, safety_predicate)")

    def set_config(self, config, data):
        pass # Do nothing with the config by default

    def shutdown(self):
        pass # Do nothing during shutdown by default

ALL_SAFE = 0
ALL_UNSAFE = 1
SOME_UNSAFE = 2
TOO_BIG = 3

