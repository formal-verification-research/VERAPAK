from .discrete_search import DiscreteSearch
from .eran import ERAN

STRATEGIES = {
    "discrete_search": DiscreteSearch,
    "eran": ERAN,
}
