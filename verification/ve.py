class VerificationEngine:

    def verification_impl(self, region):
        raise NotImplementedError("VerificationEngine did not implement verification_impl(region)")

UNKNOWN = 0
UNSAFE = 1
SAFE = 2

