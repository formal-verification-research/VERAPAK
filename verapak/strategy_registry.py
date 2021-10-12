from .verification import discrete_search
from .partitioning import largest_first as largest_first_p
from .abstraction import center_point, fgsm, random_point, rfgsm
from .dimension_ranking import by_index, gradient_based, largest_first


VERIFICATION_STRATEGIES = {
    "discrete_search": discrete_search.DiscreteSearch}

PARTITIONING_STRATEGIES = {
    "largest_first": largest_first_p.LargestFirstPartitioningStrategy}

DIMENSION_RANKING_STRATEGIES = {
    "gradient_based": gradient_based.GradientBasedDimSelection,
    "by_index": by_index.ByIndex,
    "largest_first": largest_first.LargestFirstDimSelection}

ABSTRACTION_STRATEGIES = {
    "center": center_point.CenterPoint,
    "fgsm": fgsm.FGSM,
    "random": random_point.RandomPoint,
    "rfgsm": rfgsm.RFGSM}
