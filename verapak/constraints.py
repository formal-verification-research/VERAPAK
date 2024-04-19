import numpy as np
from ml_constraints import Constraints, Constraint

__all__ = [
    "Constraints",
    "Constraint",
    "SafetyPredicate"
]

class SafetyPredicate:
    def __init__(self, num_labels, evaluate_function, constraints=None):
        self.num_labels = num_labels
        self.evaluate_function = evaluate_function
        if constraints is None:
            self.constraints = Constraints()
        else:
            self.constraints = constraints

    def add_constraint(self, labels, constraint, other=None):
        self.constraints.add(Constraint(labels, constraint, other))

    def __invert__(self):
        s = SafetyPredicate(self.num_labels, self.evaluate_function)
        s.constraints = ~self.constraints
        return s

    def __call__(self, point):
        return self.constraints(self.evaluate_function(point))

    def __repr__(self):
        return str(self.num_labels) + "\n" + repr(self.constraints)

    def best_case_scenario(self, output):
        """
        Creates a reasonable desired/'safe' output, even for non-classification networks
        """
        desired = np.zeros(self.num_labels)
        for constraint in self.constraints.constraints:
            if constraint.constraint == "min":
                # We don't know the scale of the output space, so just take the minimum value
                v = np.amin(output)
                for label in constraint.labels:
                    desired[int(label)] = v
            elif constraint.constraint == "max":
                # We don't know the scale of the output space, so just take the maximum value
                v = np.amax(output)
                for label in constraint.labels:
                    desired[int(label)] = v
            elif constraint.constraint in ["notmin", "notmax"]:
                # We don't know the scale of the output space, so just put it somewhere in the middle
                a = np.amin(output)
                b = np.amax(output)
                for label in constraint.labels:
                    desired[int(label)] = (a + b) / 2
            elif constraint.constraint in [">", "<", ">=", "<="]:
                # If a value fails a comparison with a constant, set it to that constant.
                if constraint.other_const:
                    f = max if ">" in constraint.constraint else min
                    v = constraint.other
                    if constraint.constraint[-1] == "=":
                        v *= 0.99 if f is min else 1.01
                    desired[int(self.labels[0])] = f(desired[int(self.labels[0])], v)

                # If a comparison constraint fails, flip the operands - because then it will succeed
                elif not Constraint.F[constraint.constraint](desired[self.labels[0]], desired[self.other]):
                    v = desired[self.other]
                    desired[self.other] = desired[self.labels[0]]
                    desired[self.labels[0]] = v
        return desired

