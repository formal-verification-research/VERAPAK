class VerificationEngine:
    @classmethod
    def get_config_parameters(cls):
        return []

    @classmethod
    def evaluate_args(cls, args, v, errors):
        return {}

    def verify(self, region, use_cache=True):
        percent, adversarial_example = self.get_result(
                region, use_cache=use_cache)
        if percent == 1:
            assert adversarial_example is None, "Cannot be 100% safe and have an adversarial example!"
            return ALL_SAFE, None
        elif percent == 0:
            return ALL_UNSAFE, adversarial_example
        elif adversarial_example is not None:
            return SOME_UNSAFE, adversarial_example
        else:
            return UNKNOWN, None

    _last_region = None
    _last_result = None
    def get_result(self, region, use_cache=True):
        if use_cache and self._last_region == region:
            return self._last_result
        result = self.verification_impl(region)
        self._last_region = region
        self._last_result = result
        return result
    def get_percent(self, region, use_cache=True):
        return self.get_result(region, use_cache=use_cache)[0]
    def get_adv_example(self, region, use_cache=True):
        return self.get_result(region, use_cache=use_cache)[1]

    def verification_impl(self, region):
        raise NotImplementedError("VerificationEngine did not implement verification_impl(region)")

    def set_config(self, v):
        self.safety_predicate = v["safety_predicate"]

    def shutdown(self):
        pass # Do nothing during shutdown by default

ALL_SAFE = 0
ALL_UNSAFE = 1
SOME_UNSAFE = 2
TOO_BIG = 3
UNKNOWN = 3 # NOTE: UNKNOWN currently implies TOO_BIG
BOUNDARY = 4

