import numpy as np
from verapak import snap
from verapak.constraints import SafetyPredicate
from verapak.model_tools.model_base import load_graph_by_type

def Config(config_in):
    graph = load_graph_by_type(config_in["graph"], 'ONNX')
    
    initial_point = None
    label = None
    safety_predicate = SafetyPredicate(np.prod(graph.output_shape), graph.evaluate)
    if config_in["constraint_file"] is not None:
        with open(config_in["constraint_file"], 'r') as f:
            for line in f.readlines():
                parts = line.split(" ")
                labels = []
                i = 0
                while parts[i] not in [">", "<", "min", "max", "notmin", "notmax", "<="]:
                    labels.append(parts[i])
                    i += 1
                if len(parts) == i + 1: # constraint is last element
                    safety_predicate.add_constraint(labels, constraint)
                else:
                    safety_predicate.add_constraint(labels, constraint, parts[i])
    elif config_in["initial_point"] is not None:
        initial_point = np.array( \
                config_in["initial_point"], \
                dtype=graph.input_dtype) \
            .reshape(graph.input_shape)
        if config_in["label"] is None:
            labels = graph.evaluate(initial_point.flatten())
            n = np.argmax(label)
            label = np.zeros_like(label)
            label[n] = 1
            safety_predicate.add_constraint(n, "max")
        else:
            safety_predicate.add_constraint(config_in["label"], "max")
    elif config_in["label"] is None:
        label = config_in["label"]
        safety_predicate.add_constraint(label, "max")
    else:
        raise ValueError("Config must include one of: initial_point, label, constraint_file")

    gradient_function = lambda point: graph.gradient_of_loss_wrt_input(point, safety_predicate.best_case_scenario(point))

    radius = np.full(graph.input_shape, config_in["radius"][0]) if len(config_in["radius"]) == 1 \
            else np.array(config_in["radius"]).reshape(graph.input_shape).astype(graph.input_dtype)

    abstraction_strategy = config_in["abstraction_strategy"]()
    partitioning_strategy = config_in["partitioning_strategy"]()
    verification_strategy = config_in["verification_strategy"]()

    num_abstractions = config_in["num_abstractions"]

    if len(config_in["domain_upper_bound"]) == 1:
        dub = np.full_like(initial_point, config_in["domain_upper_bound"][0])
    else:
        dub = np.array(config_in["domain_upper_bound"]).reshape(graph.input_shape).astype(graph.input_dtype)

    if len(config_in["domain_lower_bound"]) == 1:
        dlb = np.full_like(initial_point, config_in["domain_lower_bound"][0])
    else:
        dlb = np.array(config_in["domain_lower_bound"]).reshape(graph.input_shape).astype(graph.input_dtype)
    domain = [dlb, dub]
    initial_region = snap.region_to_domain([initial_point - radius, initial_point + radius], domain)
    
    return {
        "graph": graph,
        "initial_point": initial_point,
        "label": label,
        "safety_predicate": safety_predicate,
        "radius": radius,
        "gradient_function": gradient_function,
        "strategy": {
            "abstraction": abstraction_strategy,
            "partitioning": partitioning_strategy,
            "verification": verification_strategy,
        },
        "num_abstractions": num_abstractions,
        "domain": domain,
        "initial_region": initial_region,
        "output_dir": config_in["output_dir"],
        "timeout": config_in["timeout"],
        "halt_on_first": config_in["halt_on_first"],
        "raw": config_in
    }

