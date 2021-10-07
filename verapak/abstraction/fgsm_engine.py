from .ae import AbstractionEngine
from .center_point import CenterPoint
import numpy as np


def _min_dim(region):
    diff = np.abs(region[1] - region[0])
    min_index = np.unravel_index(np.argmin(diff.flatten()), diff.shape)
    return diff[min_index], min_index


class ModFGSM(AbstractionEngine):

    def __init__(self, fallback_strategy=None, fallback_predicate=lambda region: False, percent_fgsm=1):
        self.fallback_strategy = fallback_strategy
        self.fallback_predicate = fallback_predicate
        self.percent_fgsm = percent_fgsm

    def set_config(config):
        self.config = config
        self.gradient = lambda point: config["Graph"].gradient_of_loss_wrt_input(
            point, LABEL)  # TODO: Where to get label?
        self.granularity = config["Granularity"]

    def abstraction_impl(self, region, num_abstractions):
        if not hasattr(self, "config") or self.config is None:
            raise RuntimeError(
                "FGSMEngine requires a call to set_config(config) before calling abstraction_impl(region, num_abstractions)")

        # Should fallback?
        if self.fallback_strategy is not None and self.fallback_predicate(region):
            # Then fall back
            return self.fallback_strategy.abstraction_impl(region, num_abstractions)

        center = CenterPoint().abstraction_impl(region, 1)[0]  # Get center point
        # Get sign of gradient
        gradient_sign = np.sign(self.gradient(center))

        min_dimension_range, min_dimension_index = _min_dim(region)
        # Divide by 1.25 probably to keep points within the region, not on the borders.
        max_radius = min_dimension_range / 1.25

        e1_lowerbound = 1
        e1_upperbound = int(
            max_radius / self.granularity[min_dimension_index])

        retVal = []

        for i in range(0, num_abstractions):
            current_epsilon = self.granularity[min_dimension_index] * \
                np.random.Generator.integers(e1_lowerbound, e1_upperbound)
            retVal.append(self._next_point(region, center,
                                           gradient_sign, current_epsilon))

        return retVal

    def _setup(self, region, num_abstractions, center):
        raise NotImplementedError(
            "FGSMEngine did not implement _setup(self, region, num_abstractions, center)")

    def _next_point(self, region, center, gradient_sign, epsilon):
        raise NotImplementedError(
            "FGSMEngine did not implement _next_point(self, region, center, gradient_sign, epsilon)")
