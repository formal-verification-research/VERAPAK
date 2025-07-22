import numpy as np
from verapak import snap
from verapak.constraints import SafetyPredicate, Constraints
from verapak.model_tools.model_base import load_graph_by_type
from verapak_utils import Region, RegionData

class ConfigValueError(Exception):
    def __init__(self, message, path=None, key=None):
        self.message = message
        self.path = path
        self.key = key
        self.use_color = False
    def set_key(self, key):
        self.key = key
    def colorize(self, use_color=True, colors=(9, 14, 219)):
        if use_color is not None:
            self.use_color = use_color
        self.colors = colors
        return self
    def _color(self, i):
        if not self.use_color:
            return ""
        return f"\033[38;5;{self.colors[i]}m"
    def _no_color(self):
        if not self.use_color:
            return ""
        return f"\033[39m"
    def __str__(self):
        no_color = self._no_color()
        error_color = self._color(0)
        key_color = self._color(1)
        special_color = self._color(2)
        s = ""
        if self.key is not None:
            s += f"{key_color}{self.key}{no_color}: "
        s += f"{error_color}{self.message}{no_color}"
        if self.path is not None:
            s += f"\n    Path: {special_color}{self.path}{no_color}"
        return s

class ConfigError(Exception):
    REASON_MISSING = "missing"
    REASON_INVALID = "invalid"
    REASON_MISSING_ANY = "missing group"
    REASON_CONFLICTING = "conflicting"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bad_key_dict = {}
        self.missing_errors = {}
        self.invalid_errors = {}
        self.missing_group_errors = []
        self.conflicting_errors = []
        self.use_color = False
    def missing(self, key, valid=None, reason=None):
        self.bad_key_dict[key] = ConfigError.REASON_MISSING
        self.missing_errors[key] = (valid, reason)
        return self
    def invalid(self, key, value, valid=None, reason=None):
        self.bad_key_dict[key] = ConfigError.REASON_INVALID
        self.invalid_errors[key] = (value, valid, reason)
        return self
    def missing_any(self, keys, reason=None):
        for key in keys:
            self.bad_key_dict[key] = ConfigError.REASON_MISSING_ANY
        self.missing_group_errors.append((keys, reason))
        return self
    def conflicting(self, keys, values, reason=None):
        for key in keys:
            self.bad_key_dict[key] = ConfigError.REASON_CONFLICTING
        self.conflicting_errors.append((keys, values, reason))
        return self
    def colorize(self, use_color=True, colors=(9, 14, 219)):
        if use_color is not None:
            self.use_color = use_color
        self.colors = colors
        return self

    def __len__(self):
        return len(self.bad_key_dict)
    def _color(self, i):
        if not self.use_color:
            return ""
        return f"\033[38;5;{self.colors[i]}m"
    def _no_color(self):
        if not self.use_color:
            return ""
        return f"\033[39m"
    def __str__(self):
        no_color = self._no_color()
        error_color = self._color(0)
        key_color = self._color(1)
        special_color = self._color(2)

        s = error_color + "Error in keys:\n"
        for bad_key, reason in self.bad_key_dict.items():
            s += f"  {key_color}{bad_key}{no_color}: {error_color}{reason}\n"

        if len(self.missing_errors) > 0:
            s += f"{no_color}REASON: {error_color}{ConfigError.REASON_MISSING}\n"
            for bad_key, (valid, reason) in self.missing_errors.items():
                if reason is None:
                    s += f"  {key_color}{bad_key}\n"
                else:
                    s += f"  {key_color}{bad_key} {no_color}({special_color}{reason}{no_color})\n"
                if valid is not None:
                    s += f"    {error_color}Valid: {special_color}{valid}\n"
        if len(self.invalid_errors) > 0:
            s += f"{no_color}REASON: {error_color}{ConfigError.REASON_INVALID}\n"
            for bad_key, (value, valid, reason) in self.invalid_errors.items():
                if reason is None:
                    s += f"  {key_color}{bad_key}{no_color}: {error_color}{value}\n"
                else:
                    s += f"  {key_color}{bad_key}{no_color}: {error_color}{value} {no_color}({special_color}{reason}{no_color})\n"
                if valid is not None:
                    s += f"    {error_color}Valid: {special_color}{valid}\n"
        if len(self.missing_group_errors) > 0:
            s += f"{no_color}REASON: {error_color}{ConfigError.REASON_MISSING_ANY}\n"
            for (keys, reason) in self.missing_group_errors:
                s += f"  {key_color}" + f"{no_color}, {key_color}".join(keys[:-1]) + f"{no_color} or {key_color}" + keys[-1] + "\n"
                if reason is not None:
                    s += f"    {special_color}{reason}\n"
        if len(self.conflicting_errors) > 0:
            s += f"{no_color}REASON: {error_color}{ConfigError.REASON_CONFLICTING}\n"
            for (keys, values, reason) in self.conflicting_errors:
                if reason is None:
                    s += f"  {error_color}GROUP\n"
                else:
                    s += f"  {error_color}GROUP {no_color}({special_color}{reason}{no_color})\n"
                for key, value in zip(keys, values):
                    s += f"    {key_color}{key}{no_color}: {error_color}{value}\n"
        return s[:-1] + no_color # Cut off the trailing newline

class Config(dict):
    def __init__(self, config_in):
        self.raw = config_in
        super().__init__(**evaluate_args(config_in))
    def to_binary(self):
        pass

def evaluate_args(args):
    v = {
        "graph": args.get("graph"), # COMPUTED
        "initial_point": args.get("initial_point"), # RESHAPED
        "radius": args.get("radius"), # RESHAPED, REMOVED
        "strategy": {
            "abstraction": args["abstraction_strategy"](),
            "partitioning": args["partitioning_strategy"](),
            "verification": args["verification_strategy"](),
        },
        "num_abstractions": args["num_abstractions"],
        "output_dir": args["output_dir"],
        "timeout": args["timeout"],
        "halt_on_first": args["halt_on_first"],
        "load": args.get("load"),
        "initial_region": None, # COMPUTED
        "safety_predicate": None, # COMPUTED
        "gradient_function": None, # COMPUTED
        "ignored_dimensions": None, # COMPUTED
    }

    errors = ConfigError()
    if args["color"]:
        errors.colorize()

    # Load graph
    if v["graph"] is None:
        errors.missing("graph")
    else:
        try:
            v["graph"] = load_graph_by_type(v["graph"], v["graph"].split(".")[-1].upper())
        except NotImplementedError as ex:
            errors.invalid("graph", v["graph"], reason=ex.args[0] if ex.args else None)
        except FileNotFoundError:
            errors.invalid("graph", v["graph"], reason="No such file or directory")

    # Reshape initial_point
    if v["initial_point"] is not None:
        v["initial_point"] = np.array(v["initial_point"], dtype=v["graph"].input_dtype).reshape(v["graph"].input_shape)

    # Reshape radius
    if v["radius"] is not None:
        v["radius"] = np.array(v["radius"], dtype=v["graph"].input_dtype).reshape(v["graph"].input_shape)

    # Compute initial_region
    radius = v["radius"]
    region_bounds = [args.get("region_lower_bound"), args.get("region_upper_bound")]
    load = args.get("load")

    if radius is not None and v["initial_point"] is None:
        errors.missing("initial_point", reason="Cannot set a radius without an initial point")
    if region_bounds[0] is None and region_bounds[1] is not None:
        errors.missing("region_lower_bound", reason="Cannot set a region_upper_bound without a region_lower_bound")
    elif region_bounds[0] is not None and region_bounds[1] is None:
        errors.missing("region_upper_bound", reason="Cannot set a region_lower_bound without a region_upper_bound")
    if radius is None and region_bounds[0] is None and load is None:
        errors.missing_any(("radius", "region_lower_bound", "region_upper_bound", "load"), reason="Must set exactly one of radius, region bounds, or load file")
    elif int(radius is not None) + int(region_bounds[0] is not None) + int(load is not None) > 1:
        conflicting = ([],[])
        for k, v in [("radius", radius), ("region_lower_bound", region_bounds[0]), ("region_upper_bound", region_bounds[1]), ("load", load)]:
            if v is not None:
                conflicting[0].append(k)
                conflicting[1].append(v)
        errors.conflicting(conflicting[0], conflicting[1], reason="Must set only one of radius, region bounds, or load file")

    if len(errors) == 0:
        if radius is not None:
            if len(radius) == 1:
                radius = np.full(v["graph"].input_shape, radius[0])
            else:
                radius = np.array(radius).reshape(v["graph"].input_shape).astype(v["graph"].input_dtype)

            v["initial_region"] = Region(v["initial_point"] - radius, v["initial_point"] + radius, RegionData.EMPTY)
        elif region_bounds[0] is not None:
            if len(region_bounds[0]) == 1:
                region_bounds[0] = np.full(v["graph"].input_shape, region_bounds[0])
            else:
                region_bounds[0] = np.array(region_bounds[0]).reshape(v["graph"].input_shape).astype(v["graph"].input_dtype)
            if len(region_bounds[1]) == 1:
                region_bounds[1] = np.full(v["graph"].input_shape, region_bounds[1])
            else:
                region_bounds[1] = np.array(region_bounds[1]).reshape(v["graph"].input_shape).astype(v["graph"].input_dtype)

            v["initial_region"] = Region(region_bounds[0], region_bounds[1], RegionData.EMPTY)
        elif load is not None:
            v["load"] = load

    # Compute safety_predicate
    initial_point = args.get("initial_point")
    constraint_file = args.get("constraint_file")
    label = args.get("label")
    constraints = args.get("constraints")

    if initial_point is None and constraint_file is None and label is None and constraints is None:
        errors.missing_any(("constraint_file", "label", "initial_point", "constraints"), reason="Must set at least one of constraint_file, label, initial_point, or constraints (in order to derive validity constraints)")
    elif constraint_file is not None and label is not None:
        errors.conflicting(("constraint_file", "label"), (constraint_file, label), reason="Cannot set both constraint_file and label")
    elif int(constraint_file is not None) + int(label is not None) + int(constraints is not None) > 1:
        conflicting = ([],[])
        for k, v in [("constraint_file", constraint_file), ("label", label), ("constraints", constraints)]:
            if v is not None:
                conflicting[0].append(k)
                conflicting[1].append(v)
        errors.conflicting(conflicting[0], conflicting[1], reason="Must set only one of constraint_file, label, or constraints")

    if len(errors) == 0:
        if label is not None:
            constraints = Constraints.from_label(label)
        elif constraint_file is not None:
            constraints = Constraints.from_constraint_file(constraint_file)
        elif constraints is not None:
            if type(constraints) == str:
                constraints = Constraints.from_text(constraints)
            else:
                constraints = Constraints.from_object(constraints)
        else:
            constraints = Constraints.from_label(np.argmax(v["graph"].evaluate(v["initial_point"])))

        v["safety_predicate"] = SafetyPredicate(
            np.prod(v["graph"].output_shape),
            v["graph"].evaluate,
            constraints=constraints)

    # Compute gradient_function
    if len(errors) == 0:
        v["gradient_function"] = lambda point: v["graph"].gradient_of_loss_wrt_input(point, v["safety_predicate"].best_case_scenario(point))

    # Compute ignored_dimensions
    def is_ignored(x): return x == 0
    is_ignored = np.vectorize(is_ignored)
    v["ignored_dimensions"] = is_ignored(v["initial_region"].high - v["initial_region"].low)

    # Reshape and compute each strategy's optional arguments
    #   Added strategies must be evaluated *by the parent strategy*
    for strategy in list(v["strategy"].values()):
        strategy.evaluate_args(args, v, errors)

    # Remove radius
    del v["radius"]

    if len(errors) > 0:
        raise errors

    return v

