from .dim_select_base import *
from abstraction import center_point as cp_abstract
import numpy as np


class GradientBasedDimSelection(DimSelectionBase):
    def __init__(self, grad_func):
        self.gradient_func = grad_func
        self.center_point_abstraction = cp_abstract.CenterPoint()

    def rank_indices(self, region):
        center_point = self.center_point_abstraction.abstraction_impl(region, 1)[
            0]
        abs_grad = np.abs(self.gradient_func(center_point))
        retVal = list(np.argsort(abs_grad.flatten()))
        retVal.reverse()
        return retVal
