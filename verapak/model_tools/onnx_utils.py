import warnings

import tensorflow as tf
import onnx
import numpy as np


def onnx_dtype_to_tf_dtype(onnx_dtype):
    str_val = onnx.TensorProto.DataType.keys()[onnx_dtype]
    if str_val == 'UINT8':
        return np.uint8
    if str_val == 'INT8':
        return np.int8
    if str_val == 'UINT16':
        return np.uint16
    if str_val == 'INT16':
        return np.int16
    if str_val == 'INT32':
        return np.int32
    if str_val == 'INT64':
        return np.int64
    if str_val == 'STRING':
        return np.string
    if str_val == 'BOOL':
        return np.bool
    if str_val == 'FLOAT16':
        return np.float16
    if str_val == 'DOUBLE':
        return np.double
    if str_val == 'UINT32':
        return np.uint32
    if str_val == 'UINT64':
        return np.uint64
    if str_val == 'COMPLEX64':
        return np.complex64
    if str_val == 'COMPLEX128':
        return np.complex128
    if str_val == 'BFLOAT16':
        return np.bfloat16
    return np.float32


def get_onnx_node_shape(node):
    tensor_type = node.type.tensor_type
    if not tensor_type.HasField('shape'):
        return None
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.HasField('dim_value'):
            shape.append(dim.dim_value)
        else:
            shape.append(-1)
    return shape


def get_onnx_node_dtype(node):
    return onnx_dtype_to_tf_dtype(node.type.tensor_type.elem_type)


def get_onnx_input_shape(onnx_model):
    return get_onnx_node_shape(get_onnx_input_node(onnx_model))


def get_onnx_output_shape(onnx_model):
    return get_onnx_node_shape(get_onnx_output_node(onnx_model))


def get_onnx_input_dtype(onnx_model):
    return get_onnx_node_dtype(get_onnx_input_node(onnx_model))


def get_onnx_output_dtype(onnx_model):
    return get_onnx_node_dtype(get_onnx_output_node(onnx_model))

def select_onnx_node(onnx_model, to_select, qualifier):
    if len(to_select) == 1:
        return to_select
    nodes = onnx_model.graph.node
    for node in nodes:
        to_select = filter(lambda s: qualifier(s, node), to_select)
    return list(to_select)

def get_onnx_input_node(onnx_model):
    inputs = select_onnx_node(onnx_model, onnx_model.graph.input, lambda input, node: input.name not in node.output)
    if len(inputs) > 1:
        warnings.warn(f'multiple input nodes... choosing the last, named "{inputs[-1].name}"', ResourceWarning, source=onnx_model.graph.name)
    return inputs[-1]

def get_onnx_output_node(onnx_model):
    outputs = select_onnx_node(onnx_model, onnx_model.graph.output, lambda output, node: output.name not in node.input)
    if len(outputs) > 1:
        warnings.warn(f'multiple output nodes... choosing the last, named "{outputs[-1].name}"', ResourceWarning, source=onnx_model.graph.name)
    return outputs[-1]
