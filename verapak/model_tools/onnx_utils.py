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
    if len(onnx_model.graph.input) > 1:
        print('multiple input nodes... chosing the first')
    node = onnx_model.graph.input[0]
    return get_onnx_node_shape(node)


def get_onnx_output_shape(onnx_model):
    if len(onnx_model.graph.output) > 1:
        print('multiple output nodes... chosing the first')
    node = onnx_model.graph.output[0]
    return get_onnx_node_shape(node)


def get_onnx_input_dtype(onnx_model):
    node = onnx_model.graph.input[0]
    return get_onnx_node_dtype(node)


def get_onnx_output_dtype(onnx_model):
    node = onnx_model.graph.output[0]
    return get_onnx_node_dtype(node)
