from .center_point import CenterPoint
from .fgsm import FGSM
from .random_point import RandomPoint
from .rfgsm import RFGSM

STRATEGIES = {
    "center": CenterPoint,
    "fgsm": FGSM,
    "random": RandomPoint,
    "rfgsm": RFGSM,
}
