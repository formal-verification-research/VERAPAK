class VerificationEngine:

    def verification_impl(self, region, safety_predicate):
        raise NotImplementedError("VerificationEngine did not implement verification_impl(region, safety_predicate)")

UNKNOWN = 0
UNSAFE = 1
SAFE = 2

