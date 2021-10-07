from .fgsm_engine import FGSMEngine
import numpy as np

class RplusFGSM(FGSMEngine):

    def set_config(self, config):
        pass # Does not need any info from config

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

