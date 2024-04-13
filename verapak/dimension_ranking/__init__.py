from .by_index import ByIndexDimSelection
from .gradient_based import GradientBasedDimSelection
from .largest_first import LargestFirstDimSelection

STRATEGIES = {
    "gradient_based": GradientBasedDimSelection,
    "by_index": ByIndexDimSelection,
    "largest_first": LargestFirstDimSelection,
}
