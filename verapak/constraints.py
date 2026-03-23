import numpy as np
from ml_constraints import Constraint

__all__ = [
    "Constraint",
    "SafetyPredicate"
]

class SafetyPredicate:
    def __init__(self, evaluate_function, constraint=None):
        self.evaluate_function = evaluate_function
        self.constraint = constraint

    def add_constraint(self, labels, constraint, other=None):
        constraint = Constraint(labels, constraint, other)
        if self.constraint is None:
            self.constraint = constraint
        else:
            self.constraint &= constraint

    def __invert__(self):
        return SafetyPredicate(
                self.evaluate_function,
                constraint=~self.constraint if self.constraint is not None else None
        )

    def __call__(self, point):
        if self.constraint is not None:
            return self.constraint(self.evaluate_function(point))
        else:
            return True

    def __repr__(self):
        return str(self.num_labels) + "\n" + repr(self.constraints)

    def best_case_scenario(self, output):
        """
        Creates a reasonable desired/'safe' output, even for non-classification networks
        """
        if not self.constraint(output):
            return self.constraint.nearest_satisfactory_point(output)
        else:
            return self.constraint.get_optimal_points[0]

