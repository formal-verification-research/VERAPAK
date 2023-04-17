import numpy as np

class Constraint:
    def __init__(self, labels, constraint, other=None):
        if constraint not in [">", "<", "min", "max", "notmin", "notmax", "<="]:
            raise ValueError(f"Bad constraint `{constraint}`")
        elif constraint in [">", "<"] and other is None:
            raise ValueError(f"Constraint `{constraint}` requires a second value")
        elif constraint in [">", "<"] and len(labels) != 1:
            raise ValueError(f"Only one label is allowed on either side of constraint `{constraint}`")
        if constraint == "<=":
            other = float(other)
        if labels is not list:
            labels = [labels]
        self.labels = labels
        self.constraint = constraint
        self.other = other
        self.repr = " ".join(list(map(lambda n: f"y{int(n)}", labels))) + " " + constraint
        if self.other is not None and self.other is not float:
            self.repr += f" y{int(other)}"
        elif self.other is float:
            self.repr += " " + str(other)

    def __repr__(self):
        return self.repr

    def __call__(self, values):
        if self.constraint == "min":
            return np.argmin(values) in self.labels
        elif self.constraint == "max":
            return np.argmax(values) in self.labels
        elif self.constraint == "notmin":
            return np.argmin(values) not in self.labels
        elif self.constraint == "notmax":
            return np.argmax(values) not in self.labels
        elif self.constraint == ">":
            return values[self.labels[0]] > values[self.other]
        elif self.constraint == "<":
            return values[self.labels[0]] < values[self.other]
        elif self.constraint == "<=":
            return values[self.labels[0]] <= self.other

class Constraints:
    def __init__(self):
        self.constraints = []

    def add(self, *constraints):
        self.constraints.extend(constraints)

    def __repr__(self):
        return '\n'.join(map(repr, self.constraints))

    def __call__(self, values):
        for constraint in self.constraints:
            if not constraint(values):
                return False
        return True

class SafetyPredicate:
    def __init__(self, num_labels, evaluate_function):
        self.num_labels = num_labels
        self.evaluate_function = evaluate_function
        self.constraints = Constraints()

    def add_constraint(self, labels, constraint, other=None):
        self.constraints.add(Constraint(labels, constraint, other))

    def __call__(self, point):
        return self.constraints(self.evaluate_function(point))

    def __repr__(self):
        return str(self.num_labels) + "\n" + repr(self.constraints)

