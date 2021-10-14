from .dre import *
from ..abstraction import center_point as cp_abstract
import numpy as np


class GradientBasedDimSelection(DimensionRankingEngine):
    def __init__(self, gradient_function, **kwargs):
        self.gradient_func = gradient_function
        self.center_point_abstraction = cp_abstract.CenterPoint()

    def rank_indices_impl(self, region):
        if self.gradient_func is None:
            raise NotImplementedError(
                "GradientBasedDimSelection requires a valid gradient function. Use set_config(config) with a valid value for \"Graph\" to set.")

        center_point = self.center_point_abstraction.abstraction_impl(region, 1)[
            0]
        abs_grad = np.abs(self.gradient_func(center_point))
        retVal = list(np.argsort(abs_grad.flatten()))
        retVal.reverse()
        return retVal
