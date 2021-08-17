import abstraction.ae
import CenterPoint from abstraction.center_point
import numpy as np

def _min_dim(region):
    minimum = None
    minimum_index = None
    for i, subregion in enumerate(region):
        if minimum == None or subregion.second - subregion.first < minimum.second - minimum.first:
            minimum = subregion
            minimum_index = i
    return minimum, minimum_index


class FGSMEngine(ae.AbstractionEngine):

    def __init__(self, gradient_function, granularity):
        self.gradient = gradient_function
        self.granularity = granularity

    def abstraction_impl(self, region, num_abstractions):
        center = CenterPoint().abstraction_impl(region, 1)
        if len(center) == 0:
            return []
        else:
            center = center[0] # Should only be one point
            gradient_sign = np.sign(self.gradient(center))

            min_dimension, min_dimension_index = _min_dim(region) # TODO: In original, uses std::distance --- Why?
            max_radius = min_dimension.second - min_dimension.first / 1.25 # TODO: Why divide by 1.25

            e1_lowerbound = 1
            e1_upperbound = int(max_radius / granularity[min_dimension_index])

            retVal = []

            for i in range(0, num_abstractions):
                current_epsilon = granularity[min_dimension_index] * np.random.Generator.integers(e1_lowerbound, e1_upperbound)
                retVal.append(self._fgsm_method(region, center, gradient_sign, current_epsilon))
            
            return retVal

    def _setup(self, region, num_abstractions):
        raise NotImplementedError("FGSMEngine did not implement _setup(self, region, num_abstractions)")
    
    def _next_point(self, region, ):
        raise NotImplementedError("FGSMEngine did not implement _next_point(self, region, ...)")


class ModFGSM(FGSMEngine):

    def __init__(self, gradient_function, granularity, dim_select_strategy, percent_fgsm, fallback_strategy):
        super.__init__(gradient_function, granularity)
        self.dim_select_strategy = dim_select_strategy
        self.percent_fgsm = percent_fgsm
        self.fallback_strategy = fallback_strategy

    def _setup(self, region, num_abstractions):
        if region[1] == min_dimension || max_radius <= granularity[min_dimension_index]:
            return self.fallback_strategy(region, num_abstractions)

        dims = self.dim_select_strategy(region, region.size())
        num_dims_fgsm = self.percent_fgsm * region.size()

        self.M = np.zeros(center.size())
        self.M_not = np.one(center.size)

        for i in range(0, num_dims_fgsm):
            self.M[dims[i]] = 1
            self.M_not[dims[i]] = 0

    def _next_point(self, region, center, gradient_sign, epsilon):
        R = np.random.Generator.integers(0, 1, region.size(), endpoint=True)
        for i in range(0, center.size()):
            if (R[i] == 0):
                R[i] = -1

        fgsm_part = gradient_sign * self.M
        mod_part = R * self.M_not
        scaled_part = epsilon * (fgsm_part + mod_part)
        generated_point = center + scaled_part

        return generated_point


class RplusFGSM(ae.AbstractionEngine):

    def __init__(max_points, gradient, granularity, epsilon):
        self.max_points = max_points
        self.gradient = gradient
        self.granularity = granularity
        self.epsilon = epsilon

    def _setup(self, region, num_abstractions):
        pass

    def _next_point(self, region, center, gradient_sign, epsilon):
        generated_point = np.empty(center.size())
        for i in range(0, center.size()):
            generated_point[i] = center + (epsilon * gradient_sign[i])
        return generated_point

