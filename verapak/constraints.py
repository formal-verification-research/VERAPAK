import numpy as np

class Constraint:
    INVERT_MAP = {
        "min": "notmin",
        "max": "notmax",
        "notmin": "min",
        "notmax": "max",
        ">": "<=",
        "<": ">=",
        ">=": "<",
        "<=": ">"
    }
    F = {
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
    }
    def __init__(self, labels, constraint, other=None):
        self.vnnlib = False
        self.other_const = False
        if other is not None:
            if other.endswith("f"):
                other = float(other[:-1])
                self.other_const = True
            elif "." in other:
                other = float(other)
                self.other_const = True
            else:
                other = int(other)

        if len(labels) == 0:
            raise ValueError("Every constraint requires at least one label")
        if constraint in ["min", "max", "notmin", "notmax"]:
            if other is not None:
                raise ValueError("Min/max constraints should be of the format `<labels> [not]<min|max>`")
            self.vnnlib = True
        elif constraint in [">", "<", ">=", "<="]:
            if other is None or len(labels) == 0:
                raise ValueError("Greater/less than constraints require a label on the left and a label or value on the right")
            if len(labels) > 1:
                raise ValueError("Greater/less than constraints can only have one label")
            if (self.other_const and constraint == ">=") or (not self.other_const and constraint in [">", "<"]):
                self.vnnlib = True
        else:
            raise ValueError(f"Bad constraint `{constraint}`")

        self.repr = " ".join(list(map(lambda n: f"y{int(n)}", labels))) + " " + constraint
        if other_float:
            self.repr += " " + str(other)
        elif other is not None:
            self.repr += f" y{int(other)}"

        self.labels = labels
        self.constraint = constraint
        self.other = other

    def __invert__(self):
        return Constraint(self.labels, INVERT_MAP[self.constraint], other=self.other)

    def is_vnnlib(self, force=False):
        if self.constraint == ">=" and not force:
            return False
        if self.other is not None:
            if self.constraint == "<=" and type(self.other) is not float:
                return False
            if self.constraint != "<=" and type(self.other) is float:
                return False
        return True

    def force_vnnlib(self):
        if self.is_vnnlib():
            return self
        elif not self.is_vnnlib(force=True):
            raise ValueError(f"Cannot convert constraint `{self.repr}` to vnnlib format")
        return Constraint(self.labels, self.constraint.replace("=", ""), other=other)


    def __repr__(self):
        return self.repr

    def percent_true(self, lower_bound, upper_bound):
        if self.constraint in ["min", "max", "notmin", "notmax"]:
            chosen = self.labels
            unchosen = list(filter(lambda x: x not in chosen, range(len(lower_bound))))

            f = max if "max" in self.constraint else min

            chosen_low = f(map(lambda x: lower_bound[x], chosen))
            chosen_high = f(map(lambda x: upper_bound[x], chosen))
            unchosen_low = f(map(lambda x: lower_bound[x], unchosen))
            unchosen_high = f(map(lambda x: upper_bound[x], unchosen))

            # Invert mins so we only have to check once
            if "min" in self.constraint:
                chosen_low, chosen_high, unchosen_low, unchosen_high = unchosen_low, unchosen_high, chosen_low, chosen_high

            # ALWAYS a max: low >= high
            if chosen_low >= unchosen_high:
                result = 1
            # SOMETIMES a max: high >= low
            elif chosen_high >= unchosen_low:
                range_chosen = chosen_high - chosen_low
                range_unchosen = unchosen_high - unchosen_low
                overlap = chosen_high - unchosen_low
                result = (overlap / range_chosen) * (overlap / range_unchosen)
            # NEVER a max
            else:
                result = 0

            if invert:
                return 1 - result
            return result

        elif self.constraint in [">", "<", ">=", "<="]:
            v1_low = lower_bound[self.labels[0]]
            v1_high = upper_bound[self.labels[0]]
            if not self.other_const:
                v2_low = lower_bound[self.other]
                v2_high = upper_bound[self.other]
            else:
                v2_low = self.other
                v2_high = self.other

            # Invert less-than so we only have to check once (keep `=`, if present)
            if self.constraint[0] == "<":
                v1_low, v1_high, v2_low, v2_high = v2_low, v2_high, v1_low, v1_high
            f = Constraint.F[self.constraint.replace("<", ">")]

            # ALL greater: low >(=) high
            if f(v1_low, v2_high):
                return 1
            # SOME greater: high >(=) low
            elif f(v1_high, v2_low):
                v1_range = v1_high - v1_low
                v2_range = v2_high - v2_low
                overlap = v1_high - v2_low
                return (overlap / v1_range) * (overlap / v2_range)
            # NONE greater
            else:
                return 0

    def __call__(self, values, upper_bound=None):
        if upper_bound is not None:
            return self.percent_true(values, upper_bound)

        elif self.constraint in ["min", "max", "notmin", "notmax"]:
            invert = "not" in self.constraint
            f = np.argmax if "max" in self.constraint else np.argmin
            result = f(values) in self.labels
            if invert:
                return not result
            return result

        elif self.constraint in [">", "<", ">=", "<="]:
            v1 = values[self.labels[0]]
            v2 = self.other
            if not self.other_const:
                v2 = values[v2]
            return Constraint.F[self.constraint](v1, v2)

class Constraints:
    @classmethod
    def from_text(cls, text):
        c = cls()
        if type(text) is str:
            text = text.split("\n")
        for line in text:
            parts = line.split(" ")
            labels = []
            i = 0
            while parts[i] not in Constraint.INVERT_MAP:
                labels.append(parts[i])
                i += 1
            if len(parts) == i + 1: # constraint is last element
                c.add(Constraint(labels, parts[i]))
            else:
                c.add(Constraint(labels, parts[i], parts[i+1]))
        return c
    @classmethod
    def from_constraint_file(cls, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        return cls.from_text(lines)
    @classmethod
    def from_label(cls, label):
        c = cls()
        c.add(Constraint(label, "max"))
        return c


    def __init__(self):
        self.constraints = []

    def add(self, *constraints):
        self.constraints.extend(constraints)

    def is_vnnlib(self, force=False):
        for constraint in self.constraints:
            if not constraint.is_vnnlib(force=force):
                return False
        return True

    def force_vnnlib(self):
        c = Constraints()
        for constraint in self.constraints:
            c.add(constraint.force_vnnlib())
        return c

    def __invert__(self):
        c = Constraints()
        for constraint in self.constraints:
            c.add(~constraint)
        return c

    def __repr__(self):
        return '\n'.join(map(repr, self.constraints))

    def __call__(self, values):
        if upper_bound is None:
            for constraint in self.constraints:
                if not constraint(values):
                    return False
            return True
        elif len(self.constraints) == 0:
            return 1
        else:
            total = 0
            for constraint in self.constraints:
                total += constraint.percent_true(values, upper_bound)
            else:
                return total / len(self.constraints)

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

