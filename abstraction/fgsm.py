import abstraction.ae as ae
from abstraction.center_point import CenterPoint
import numpy as np

def _min_dim(region):
    minimum_index = None
    for i in range(region[0].size):
        if minimum_index == None or region[1][i] - region[0][i] < region[1][minimum_index] - region[0][minimum_index]:
            minimum_index = i
    return [region[0][i], region[1][i]], minimum_index


class FGSMEngine(ae.AbstractionEngine):

    def __init__(self, gradient_function, granularity, fallback_strategy=None):
        self.gradient = gradient_function
        self.granularity = granularity
        self.fallback_strategy = fallback_strategy

    def abstraction_impl(self, region, num_abstractions):
        if self.fallback_strategy is not None and self.fallback_strategy.should_fallback(region, num_abstractions): # Should fallback?
            return self.fallback_strategy(region, num_abstractions)                                                 # Then fall back

        center = CenterPoint().abstraction_impl(region, 1)              # Get center point
        if len(center) == 0:                                            # If no input:
            return []                                                       # Return no abstractions
        else:                                                           # Otherwise:
            center = center[0] # Should only be one point                   # Get actual center *point*
            gradient_sign = np.sign(self.gradient(center))                  # Get normalized gradient

            min_dimension, min_dimension_index = _min_dim(region) # TODO: In original, uses std::distance --- Why?
            max_radius = (min_dimension[1] - min_dimension[0]) / 1.25 # TODO: Why divide by 1.25

            self._setup(region, num_abstractions)

            e1_lowerbound = 1
            e1_upperbound = int(max_radius / granularity[min_dimension_index])

            retVal = []

            for i in range(0, num_abstractions):
                current_epsilon = granularity[min_dimension_index] * np.random.Generator.integers(e1_lowerbound, e1_upperbound)
                retVal.append(self._next_point(region, center, gradient_sign, current_epsilon))
            
            return retVal

    def _setup(self, region, num_abstractions, center):
        raise NotImplementedError("FGSMEngine did not implement _setup(self, region, num_abstractions)")
    
    def _next_point(self, region, center, gradient_sign, epsilon):
        raise NotImplementedError("FGSMEngine did not implement _next_point(self, region, center, gradient_sign, epsilon)")


class ModFGSM(FGSMEngine):

    def __init__(self, gradient_function, granularity, dim_select_strategy, percent_fgsm, fallback_strategy=None):
        super().__init__(gradient_function, granularity, fallback_strategy=fallback_strategy)

        self.dim_select_strategy = dim_select_strategy
        self.percent_fgsm = percent_fgsm

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

