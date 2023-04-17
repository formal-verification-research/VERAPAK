import numpy as np
from verapak import snap
from verapak.model_tools.model_base import load_graph_by_type

def Config(config_in):
    graph = load_graph_by_type(config_in["graph"], 'ONNX')
    initial_point = np.array( \
            config_in["initial_point"], \
            dtype=graph.input_dtype) \
        .reshape(graph.input_shape)
    label = config_in["label"] if config_in["label"] is not None \
            else np.argmax(graph.evaluate(initial_point).flatten())
    radius = np.full_like(initial_point, config_in["radius"][0]) if len(config_in["radius"]) == 1 \
            else np.array(config_in["radius"]).reshape(graph.input_shape).astype(graph.input_dtype)
    gradient_function = lambda point: graph.gradient_of_loss_wrt_input(point, label)

    abstraction_strategy = config_in["abstraction_strategy"]()
    partitioning_strategy = config_in["partitioning_strategy"]()
    verification_strategy = config_in["verification_strategy"]()

    output_dir = config_in["output_dir"]

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
    
    timeout = config_in["timeout"]

    return {
        "graph": graph,
        "initial_point": initial_point,
        "label": label,
        "radius": radius,
        "gradient_function": gradient_function,
        "strategy": {
            "abstraction": abstraction_strategy,
            "partitioning": partitioning_strategy,
            "verification": verification_strategy,
        },
        "domain": domain,
        "initial_region": initial_region,
        "output_dir": output_dir,
        "timeout": timeout,
        "raw": config_in
    }

