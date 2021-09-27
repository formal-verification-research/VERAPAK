from abstraction.fgsm_engine import FGSMEngine
import numpy as np

class ModFGSM(FGSMEngine):

    def set_config(self, config):
        super().set_config(config)
        self.dim_select_strategy = config["Dimension-Ranking Strategy"]
        self.percent_fgsm = config["FGSM Balance"]

    def _setup(self, region, num_abstractions):
        dims = self.dim_select_strategy(region, region[0].size)
        num_dims_fgsm = self.percent_fgsm * region[0].size

        self.M = np.zeros(region[0].size)
        self.M_not = np.ones(region[0].size)

        for i in range(0, num_dims_fgsm):
            self.M[dims[i]] = 1
            self.M_not[dims[i]] = 0

    def _next_point(self, region, center, gradient_sign, epsilon):
        R = np.random.Generator.integers(0, 1, center.size, endpoint=True)
        for i in range(0, center.size):
            if (R[i] == 0):
                R[i] = -1

        fgsm_part = gradient_sign * self.M
        mod_part = R * self.M_not
        scaled_part = epsilon * (fgsm_part + mod_part)
        generated_point = center + scaled_part

        return generated_point


# IMPORT INTERFACE
def IMPL():
    return ModFGSM()

