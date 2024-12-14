import math
from .ve import *
from eran import ERAN_factory

class ERAN(VerificationEngine):
    @classmethod
    def get_config_parameters(cls):
        return [{
            "name": "eran_timeout",
            "arg_params": {
                "type": "float",
                "help": "ERAN timeout. 0 for no timeout, negative timeout is a multiple of the full timeout",
                "default": -0.1
            }
        }]

    @classmethod
    def evaluate_args(cls, args, v, errors):
        v["eran_timeout"] = args["eran_timeout"]
        if v["eran_timeout"] < 0:
            v["eran_timeout"] = -v["eran_timeout"] / v["timeout"]
        v["eran_timeout"] = math.ceil(v["eran_timeout"])

    def __init__(self):
        pass

    def verification_impl(self, region):
        specLB = region.low
        specUB = region.high
        percent, nn, nlb, nub, _, _ = self.eran.analyze_box(specLB, specUB, "deeppoly", 1, 1, True, self.safety_predicate.constraints)
        return percent

    def set_config(self, v):
        super().set_config(v)
        self.graph_path = v["graph"].get_path()
        self.eran = ERAN_factory(self.graph_path)
        self.timeout = v["eran_timeout"]

