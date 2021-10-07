from .ae import AbstractionEngine
from .center_point import CenterPoint
import numpy as np

def _min_dim(region):
    minimum_index = None
    for i in range(region[0].size):
        if minimum_index == None or region[1][i] - region[0][i] < region[1][minimum_index] - region[0][minimum_index]:
            minimum_index = i
    return [region[0][i], region[1][i]], minimum_index


class FGSMEngine(AbstractionEngine):

    def __init__(self, fallback_strategy=None):
        self.fallback_strategy = fallback_strategy

    def set_config(config):
        self.config = config
        self.gradient = lambda point: config["Graph"].gradient_of_loss_wrt_input(point, LABEL) # TODO: Where to get label?
        self.granularity = config["Granularity"]

    def abstraction_impl(self, region, num_abstractions):
        if not hasattr(self, "config") or self.config is None:
            raise RuntimeError("FGSMEngine requires a call to set_config(config) before calling abstraction_impl(region, num_abstractions)")

        if self.fallback_strategy is not None and self.fallback_strategy.should_fallback(region, num_abstractions): # Should fallback?
            return self.fallback_strategy(region, num_abstractions)                                                 # Then fall back

        center = CenterPoint().abstraction_impl(region, 1)              # Get center point
        if len(center) == 0:                                            # If no input:
            return []                                                       # Return no abstractions
        else:                                                           # Otherwise:
            center = center[0] # Should only be one point                   # Get actual center *point*
            gradient_sign = np.sign(self.gradient(center))                  # Get normalized gradient

            min_dimension, min_dimension_index = _min_dim(region)
            max_radius = (min_dimension[1] - min_dimension[0]) / 1.25 # Divide by 1.25 probably to keep points within the region, not on the borders.

            self._setup(region, num_abstractions)

            e1_lowerbound = 1
            e1_upperbound = int(max_radius / self.granularity[min_dimension_index])

            retVal = []

            for i in range(0, num_abstractions):
                current_epsilon = self.granularity[min_dimension_index] * np.random.Generator.integers(e1_lowerbound, e1_upperbound)
                retVal.append(self._next_point(region, center, gradient_sign, current_epsilon))
            
            return retVal

    def _setup(self, region, num_abstractions, center):
        raise NotImplementedError("FGSMEngine did not implement _setup(self, region, num_abstractions, center)")
    
    def _next_point(self, region, center, gradient_sign, epsilon):
        raise NotImplementedError("FGSMEngine did not implement _next_point(self, region, center, gradient_sign, epsilon)")
