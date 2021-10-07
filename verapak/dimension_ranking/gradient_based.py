from .dre import *
from ..abstraction import center_point as cp_abstract
import numpy as np


class GradientBasedDimSelection(DimensionRankingEngine):
    def __init__(self, grad_func):
        self.gradient_func = grad_func
        self.center_point_abstraction = cp_abstract.CenterPoint()

    def set_config(self, config):
        label = None # TODO: Where to get label from?
        self.gradient_func = lambda point: config["Graph"].gradient_of_loss_wrt_input(point, label)

    def rank_indices_impl(self, region):
        if self.gradient_func is None:
            raise NotImplementedError("GradientBasedDimSelection requires a valid gradient function. Use set_config(config) with a valid value for \"Graph\" to set.")

        center_point = self.center_point_abstraction.abstraction_impl(region, 1)[
            0]
        abs_grad = np.abs(self.gradient_func(center_point))
        retVal = list(np.argsort(abs_grad.flatten()))
        retVal.reverse()
        return retVal

# IMPORT INTERFACE
def IMPL():
    return GradientBasedDimSelection(None)
