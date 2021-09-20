from abstraction.fgsm_engine import FGSMEngine
import numpy as np

class RplusFGSM(FGSMEngine):

    def __init__(self, gradient_function, granularity, epsilon, fallback_strategy=None):
        super().__init__(gradient_function, granularity, fallback_strategy=fallback_strategy)

    def _setup(self, region, num_abstractions):
        pass

    def _next_point(self, region, center, gradient_sign, epsilon):
        generated_point = np.empty(center.size())
        for i in range(0, center.size()):
            generated_point[i] = center + (epsilon * gradient_sign[i])
        return generated_point


# IMPORT INTERFACE
def IMPL():
    return RplusFGSM()

