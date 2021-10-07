import tensorflow as tf

def load_graph(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph

def get_inputs(graph_def):
    return [node for node in graph_def.node if node.op == "Placeholder"]
def guess_input(graph_def):
    inputs = get_inputs(graph_def)
    if len(inputs) == 1:
        return inputs[0]
    elif len(inputs) == 0:
        raise ValueError("Input node could not be guessed: No nodes of type \"Placeholder\" found")
    else:
        inputs_str = ", ".join([node.name for node in inputs])
        raise ValueError("Input node could not be guessed: Too many possibilities (" + inputs_str + ").")

def get_outputs(graph_def):
    node_dict = {}
    non_leaf_names = []
    for node in graph_def.node:
        node_dict[node.name] = [node, True] # True --> isLeaf
    for node in graph_def.node:
        for node_input in node.input:
            node_dict[node_input.split(":")[0]][1] = False # Outputs to something else
        if len(node.input) == 0:
            node_dict[node.name.split(":")[0]][1] = False # We don't want any detached nodes
    leaf_nodes = []
    for node_pair in node_dict.values():
        if node_pair[1]:
            leaf_nodes.append(node_pair[0])
    return leaf_nodes
def guess_output(graph_def):
    outputs = get_outputs(graph_def)
    if len(outputs) == 1:
        return outputs[0]
    elif len(outputs) == 0:
        raise ValueError("Output node could not be guessed: Network is completely cyclical (i.e. no leaf nodes)")
    else:
        outputs_str = ", ".join([node.name for node in outputs])
        raise ValueError("Output node could not be guessed: Too any possibilities (" + outputs_str + ").")

def get_node_shape(graph, node_name):
    with graph.as_default():
        node = graph.get_tensor_by_name(node_name + ":0")
        return node.shape

def modified(graph, input_name, output_name, write_path=None):
    with graph.as_default():
        # Get tensors for each node name
        output_tensor = graph.get_tensor_by_name(output_name + ":0")
        input_tensor = graph.get_tensor_by_name(input_name + ":0")

        output_shape = output_tensor.shape

        # Make a placeholder for the labels
        labels_placeholder = tf.compat.v1.placeholder(tf.float32, shape=output_tensor.shape, name="y_label")

        # Create a loss function
        loss_function = tf.nn.softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=output_tensor, name="loss_fun")

        # Create a whole bunch of nodes for finding the gradient of the loss function with respect to the input
        gradient = tf.gradients(loss_function, input_tensor)

        # Grab whichever node is designated as the "output" of the gradient, and route it to a node named "gradient_out"
        gradient_out = tf.identity(gradient, name="gradient_out")

        if write_path is not None:
            tf.io.write_graph(graph, ".", write_path, as_text=False) # Write to file if a path is given
        return graph # Return the modified graph

