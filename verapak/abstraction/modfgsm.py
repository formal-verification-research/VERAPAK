from .ae import AbstractionEngine
from .center_point import CenterPoint
import numpy as np
from ..dimension_ranking.by_index import ByIndex
import math
from ..snap import point_to_domain


def _min_dim(region):
    diff = np.abs(region[1] - region[0])
    min_index = np.unravel_index(np.argmin(diff.flatten()), diff.shape)
    return diff[min_index], min_index


class ModFGSM(AbstractionEngine):

    def __init__(self, gradient_function, granularity, fallback_strategy=None, fallback_predicate=lambda region: False, balance_factor=1, dimension_ranking_strategy=ByIndex(), **kwargs):
        self.fallback_strategy = fallback_strategy
        self.fallback_predicate = fallback_predicate
        self.balance_factor = balance_factor
        self.gradient_function = gradient_function
        self.granularity = granularity
        self.dimension_ranking_strategy = dimension_ranking_strategy

    def abstraction_impl(self, region, num_abstractions):
        # Should fallback?
        if self.fallback_strategy is not None and self.fallback_predicate(region):
            # Then fall back
            return self.fallback_strategy.abstraction_impl(region, num_abstractions)

        center = CenterPoint().abstraction_impl(
            region, 1)[0]  # Get center point
        # Get sign of gradient
        grad = self.gradient_function(center)
        gradient_sign = np.sign(grad)

        min_dimension_range, min_dimension_index = _min_dim(region)
        # Divide by 1.25 probably to keep points within the region, not on the borders.
        max_radius = min_dimension_range / 1.25

        e1_lowerbound = 1
        e1_upperbound = int(
            max_radius / self.granularity[min_dimension_index])

        sorted_dims = self.dimension_ranking_strategy.rank_indices_impl(region)
        unraveled_sorted_dims = [np.unravel_index(
            x, region[0].shape) for x in sorted_dims]
        if len(unraveled_sorted_dims[0]) == 1:
            unraveled_sorted_dims = [x[0] for x in unraveled_sorted_dims]
        M = np.full(region[0].shape, 0.0)
        num_dims_fgsm = math.floor(self.balance_factor * region[0].size)
        unraveled_sorted_dims = unraveled_sorted_dims[:num_dims_fgsm]
        M[unraveled_sorted_dims] = 1.0
        fgsm_part = gradient_sign * M
        Mnot = M * -1.0 + 1.0

        retVal = []
        for i in range(0, num_abstractions):
            current_epsilon = self.granularity[min_dimension_index] * \
                np.random.randint(e1_lowerbound, e1_upperbound)
            R = np.random.randint(0, 2, size=center.shape) * 2 - 1
            mod_part = R * Mnot
            scaled_part = current_epsilon * (fgsm_part + mod_part)
            generated_point = center + scaled_part
            retVal.append(generated_point)

        return retVal
