from verapak.abstraction.ae import AbstractionEngine
from verapak.abstraction.center_point import CenterPoint
import numpy as np
from verapak.dimension_ranking.by_index import ByIndexDimSelection
import math


def _min_dim(region):
    diff = np.abs(region[1] - region[0])
    min_index = np.unravel_index(np.argmin(diff.flatten()), diff.shape)
    return diff[min_index], min_index


class ModFGSM(AbstractionEngine):
    @classmethod
    def get_config_parameters(cls):
        return [{
            "name": "balance_factor",
            "arg_params": {
                "type": "float",
                "help": "RFGSM balance factor (1.0 -> all FGSM, 0.0 -> pure random)",
                "default": 0.95
            }
        },
        {
            "name": "granularity",
            "arg_params": {
                "type": "x_num_array",
                "help": "Granularity (single value or per dimension array): a valid discretization of the input space (8 bit image -> 1/255)"
            }
        },
        {
            'name': 'dimension_ranking_strategy',
            'arg_params':
                {
                    'type': "strategy:dimension_ranking",
                    'help': "RFGSM dimension ranking strategy: (oneof gradient_based, by_index, largest_first)",
                    'default': ByIndexDimSelection
                }
        }]

    @classmethod
    def evaluate_args(cls, args, v, errors):
        v["balance_factor"] = args["balance_factor"]
        v["granularity"] = args.get("granularity")
        v["strategy"]["dimension_ranking"] = args["dimension_ranking_strategy"]()

        # Reshape and compute each strategy's optional arguments
        v["strategy"]["dimension_ranking"].evaluate_args(args, v, errors)


    def __init__(self, fallback_strategy=None, fallback_predicate=lambda region: False):
        self.fallback_strategy = fallback_strategy
        self.fallback_predicate = fallback_predicate

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

        if self.granularity is None:
            e1_lowerbound = np.nextafter(1, 0)
            e1_upperbound = max_radius
        else:
            e1_lowerbound = 1
            e1_upperbound = max_radius / self.granularity[min_dimension_index]
        #print(max_radius)
        #print(self.granularity[min_dimension_index])
        #print(max_radius / self.granularity[min_dimension_index])

        if self.balance_factor >= 1:
            M = np.full(region[0].shape, 1.0)
        elif self.balance_factor <= 0:
            M = np.full(region[0].shape, 0.0)
        else:
            sorted_dims = self.dimension_ranking_strategy.rank_indices_impl(region)
            unraveled_sorted_dims = [np.unravel_index(
                x, region[0].shape) for x in sorted_dims]
            if len(unraveled_sorted_dims[0]) == 1:
                unraveled_sorted_dims = [x[0] for x in unraveled_sorted_dims]
            M = np.full(region[0].shape, 0.0)
            num_dims_fgsm = math.floor(self.balance_factor * region[0].size)
            unraveled_sorted_dims = unraveled_sorted_dims[:num_dims_fgsm]
            for dim_index in unraveled_sorted_dims:
                M[dim_index] = 1.0
        fgsm_part = gradient_sign * M
        Mnot = M * -1.0 + 1.0

        retVal = []
        for _ in range(0, num_abstractions):
            if self.granularity is None:
                current_epsilon = np.random.uniform(e1_lowerbound, e1_upperbound)
            else:
                current_epsilon = self.granularity[min_dimension_index] * np.random.randint(e1_lowerbound, e1_upperbound)
            R = np.random.randint(0, 2, size=center.shape) * 2 - 1
            mod_part = R * Mnot
            scaled_part = current_epsilon * (fgsm_part + mod_part)
            generated_point = center + scaled_part
            retVal.append(generated_point)

        return retVal

    def set_config(self, v):
        self.balance_factor = v["balance_factor"]
        self.gradient_function = v["gradient_function"]
        self.granularity = v["granularity"]

        if self.balance_factor > 0 and self.balance_factor < 1:
            self.dimension_ranking_strategy = v["strategy"]["dimension_ranking"]
        else:
            self.dimension_ranking_strategy = None

        if self.fallback_strategy is not None:
            self.fallback_strategy.set_config(v)

    def shutdown(self):
        if self.dimension_ranking_strategy is not None:
            self.dimension_ranking_strategy.shutdown()
        if self.fallback_strategy is not None:
            self.fallback_strategy.shutdown()

