import numpy as np
from verapak import snap
from verapak.constraints import SafetyPredicate, Constraints
from verapak.model_tools.model_base import load_graph_by_type

class Config(dict):
    def __init__(self, config_in):
        self.raw = config_in
        self.dict = evaluate_args(config_in)
    def to_binary(self):
        pass

def evaluate_args(args):
    v = {
        "graph": load_graph_by_type(args["graph"], "ONNX"),
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
    }

    # Reshape initial_point
    if v["initial_point"] is not None:
        v["initial_point"] = np.array(v["initial_point"], dtype=v["graph"].input_dtype).reshape(v["graph"].input_shape)

    # Reshape radius
    if v["radius"] is not None:
        v["radius"] = np.array(v["radius"], dtype=v["graph"].input_dtype).reshape(v["graph"].input_shape)
        
    # Compute initial_region
    region_bounds = [args.get("region_lower_bound"), args.get("region_upper_bound")]
    load = args.get("load")

    if radius is not None and v["initial_point"] is None:
        raise ValueError("Cannot set a radius without an initial point")
    elif region_bounds[0] is None and region_bounds[1] is not None:
        raise ValueError("Cannot set a region_upper_bound without a region_lower_bound")
    elif region_bounds[0] is not None and region_bounds[1] is None:
        raise ValueError("Cannot set a region_lower_bound without a region_upper_bound")
    elif radius is None and region_bounds[0] is None and load is None:
        raise ValueError("Must set exactly one of radius, region bounds, or load file")
    elif int(radius is not None) + int(region_bounds[0] is not None) + int(load is not None) > 1:
        raise ValueError("Must set only one of radius, region bounds, or load file")

    if radius is not None:
        if len(radius) == 1:
            radius = np.full(v["graph"].input_shape, radius[0])
        else:
            radius = np.array(radius).reshape(v["graph"].input_shape).astype(v["graph"].input_dtype)
        
        v["initial_region"] = (v["initial_point"] - radius, v["initial_point"] + radius, ())
    elif region_bounds[0] is not None:
        if len(region_bounds[0]) == 1:
            region_bounds[0] = np.full(v["graph"].input_shape, region_bounds[0])
        else:
            region_bounds[0] = np.array(region_bounds[0]).reshape(v["graph"].input_shape).astype(v["graph"].input_dtype)
        if len(region_bounds[1]) == 1:
            region_bounds[1] = np.full(v["graph"].input_shape, region_bounds[1])
        else:
            region_bounds[1] = np.array(region_bounds[1]).reshape(v["graph"].input_shape).astype(v["graph"].input_dtype)

        v["initial_region"] = (region_bounds[0], region_bounds[1])
    elif load is not None:
        v["load"] = load

    # Compute safety_predicate
    initial_point = args.get("initial_point")
    constraint_file = args.get("constraint_file")
    label = args.get("label")

    if initial_point is None and constraint_file is None and label is None:
        raise ValueError("Must set one of constraint_file, label, or initial_point")
    elif constraint_file is not None and label is not None:
        raise ValueError("Cannot set both constraint_file and label")

    if label is not None:
        constraints = Constraints.from_label(label)
    elif constraint_file is None:
        constraints = Constraints.from_constraint_file(constraint_file)
    else:
        constraints = Constraints.from_label(np.argmax(v["graph"].evaluate(v["initial_point"])))

    v["safety_predicate"] = SafetyPredicate(
        np.prod(v["graph"].output_shape),
        v["graph"].evaluate,
        constraints=constraints)

    # Compute gradient_function
    v["gradient_function"] = lambda point: v["graph"].gradient_of_loss_wrt_input(point, v["safety_predicate"].best_case_scenario(point))

    # Reshape and compute each strategy's optional arguments
    for strategy in v["strategy"]:
        strategy.evaluate_args(args, v)

    # Remove radius
    del v["radius"]

    return v
    
